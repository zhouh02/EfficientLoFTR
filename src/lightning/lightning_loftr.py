
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather, get_rank, get_world_size, is_main_process
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler

from torch.profiler import profile
import os
import json

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'], profiler=self.profiler)
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.timed_sample_count = 0  # Track actual timed samples

        # SMNN debug metrics aggregation
        self.smnn_debug_sums = {}
        self.smnn_debug_counts = {}

        # Peak memory tracking
        self._test_peak_memory_reset = False
        self.peak_mem_bytes = 0

        # Dump directory for JSON output
        self.dump_dir = dump_dir

        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.contiguous()

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_fine(batch, self.config, self.logger)
            
        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'num_matches': [batch['mconf'].shape[0]], # batch size = 1 only
            }

        # Collect SMNN debug metrics if present
        smnn_debug_keys = [
            'smnn_heatmap0_max', 'smnn_heatmap1_max', 'smnn_mutual_prob_max',
            'smnn_heatmap0_entropy', 'smnn_heatmap1_entropy',
            'smnn_offset_mag0', 'smnn_offset_mag1'
        ]
        for key in smnn_debug_keys:
            if key in batch:
                metrics[key] = batch[key].cpu().numpy()

        ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # figures
            ##if self.config.TRAINER.ENABLE_PLOTTING:
            ##    compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            ##    figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
            ##    for k, v in figures.items():
            ##        self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.matcher.fine_matching.validate = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        ##val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        ##figures = {self.config.TRAINER.PLOT_MODE: []}
        ##if batch_idx % val_plot_interval == 0:
        ##    figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
        figures = {}
        
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.matcher.fine_matching.validate = False
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                ##for k, v in figures.items():
                ##    if self.trainer.global_rank == 0:
                ##        for plot_idx, fig in enumerate(v):
                ##            self.logger.experiment.add_figure(
                ##                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        if (self.config.LOFTR.BACKBONE_TYPE == 'RepVGG') and not self.reparameter:
            self.matcher = reparameter(self.matcher)
            if self.config.LOFTR.HALF:
                self.matcher = self.matcher.eval().half()
            self.reparameter = True

        if not self.warmup:
            if self.config.LOFTR.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

            # Reset peak memory after warmup, only once per test run
            if torch.cuda.is_available() and not self._test_peak_memory_reset:
                torch.cuda.reset_peak_memory_stats()
                self._test_peak_memory_reset = True

        if self.config.LOFTR.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
            self.timed_sample_count += batch['image0'].size(0)
        else:
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)
                self.timed_sample_count += batch['image0'].size(0)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # Aggregate SMNN debug metrics if present
        smnn_debug_keys = [
            'smnn_heatmap0_max', 'smnn_heatmap1_max', 'smnn_mutual_prob_max',
            'smnn_heatmap0_entropy', 'smnn_heatmap1_entropy',
            'smnn_offset_mag0', 'smnn_offset_mag1'
        ]
        smnn_debug_means = {}
        for key in smnn_debug_keys:
            if key in metrics:
                values = np.concatenate([v.reshape(-1) for v in metrics[key]])
                if len(values) > 0:
                    smnn_debug_means[f'{key}_mean'] = float(np.mean(values))
                else:
                    smnn_debug_means[f'{key}_mean'] = float('nan')

        # Aggregate timing / peak-memory across ranks.
        # NCCL only supports CUDA tensors, so the reduce device must follow the active backend.
        dist_ready = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        world_size = torch.distributed.get_world_size() if dist_ready else 1

        total_ms_value = float(self.total_ms)
        timed_sample_count_value = float(self.timed_sample_count)

        # Peak memory: each rank reports its peak, take max across ranks
        if torch.cuda.is_available():
            peak_mem_bytes_value = float(torch.cuda.max_memory_allocated())
        else:
            peak_mem_bytes_value = 0.0

        # Only run collectives when there is more than one rank.
        if dist_ready and world_size > 1:
            backend = torch.distributed.get_backend()
            # NCCL requires CUDA tensors; other backends (e.g. gloo) may use CPU.
            if backend == "nccl":
                reduce_device = next(self.parameters()).device
            else:
                reduce_device = torch.device("cpu")

            # timing + count aggregated together with SUM
            timing_stats = torch.tensor(
                [total_ms_value, timed_sample_count_value],
                dtype=torch.float64,
                device=reduce_device,
            )
            torch.distributed.all_reduce(timing_stats, op=torch.distributed.ReduceOp.SUM)
            total_ms_value = float(timing_stats[0].item())
            timed_sample_count_value = float(timing_stats[1].item())

            # peak memory aggregated with MAX
            peak_mem_tensor = torch.tensor(
                peak_mem_bytes_value,
                dtype=torch.float64,
                device=reduce_device,
            )
            torch.distributed.all_reduce(peak_mem_tensor, op=torch.distributed.ReduceOp.MAX)
            peak_mem_bytes_value = float(peak_mem_tensor.item())

        peak_mem_mb = (
            peak_mem_bytes_value / (1024.0 ** 2) if torch.cuda.is_available() else None
        )

        # [{key: [{...}, *#bs]}, *#batch]
        if is_main_process():
            # Aggregate metrics
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)

            # Add SMNN debug metrics
            val_metrics_4tb.update(smnn_debug_means)

            # Compute timing metric
            avg_inference_ms_per_pair = total_ms_value / max(timed_sample_count_value, 1.0)
            val_metrics_4tb['avg_inference_ms_per_pair'] = avg_inference_ms_per_pair

            # Add peak memory
            val_metrics_4tb['peak_mem_mb'] = peak_mem_mb

            logger.info('\n' + pprint.pformat(val_metrics_4tb))

            # Export to JSON
            if self.dump_dir is not None:
                os.makedirs(self.dump_dir, exist_ok=True)
                json_path = os.path.join(self.dump_dir, 'smnn_metrics.json')

                # Safe conversion helper
                def to_json_serializable(obj):
                    if isinstance(obj, torch.Tensor):
                        if obj.numel() == 1:
                            return obj.item()
                        else:
                            return obj.cpu().tolist()
                    elif isinstance(obj, np.ndarray):
                        if obj.size == 1:
                            return obj.item()
                        else:
                            return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    elif isinstance(obj, dict):
                        return {k: to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [to_json_serializable(item) for item in obj]
                    else:
                        return str(obj)

                serializable_metrics = to_json_serializable(val_metrics_4tb)
                with open(json_path, 'w') as f:
                    json.dump(serializable_metrics, f, indent=2)
                logger.info(f"Saved metrics to {json_path}")