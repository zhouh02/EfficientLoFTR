import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            # 1-stage match
            feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
            # 2-stage 3*3 match
            feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
            # 归一化
            feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
            # 相关
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)
        # dual-softmax 抑制一对多 
        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        # 因为有图是W+2，中心裁剪一圈回到W*W
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'sim_matrix_ff': conf_matrix_ff})
            data.update({'conf_matrix_f': softmax_matrix_f})

        # compute pixel-level absolute kpt coords 
        # 1-stage output
        self.get_fine_ds_match(softmax_matrix_f, data)
        # data['stats_stage1_n'] = int(da ta['mkpts0_c'].shape[0])

        # generate 2-stage 3x3 grid
        # 阶段1得到的左右窗口内索引
        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)

        # 根据mconf对有效匹配子集作细化
        m_ids = m_ids[:len(data['mconf'])]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = (
            m_ids.reshape(-1),
            idx_l.reshape(-1),
            idx_r_iids.reshape(-1),
            idx_r_jids.reshape(-1)
        )

        # 构造 3×3 邻域的索引模板
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long)  # [1, 3, 3, 2]

        m_ids = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l = idx_l[..., None, None].expand(-1, 3, 3)  # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            data['stats_coarse_n'] = int(data['mkpts0_c'].shape[0])
            return

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W + 2, self.W + 2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]

        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale1 = (
                scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:, None, :]
                .expand(-1, -1, 2)
                .reshape(-1, 2)
                if 'scale0' in data else scale
            )

        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(coords_normalized, data, scale1)


    # 2-stage
    def get_fine_match_local(self, coords_normed, data, scale1):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

        # mkpts0_f and mkpts1_f
        # 左侧不动，阶段2仅围绕右窗进行细化
        mkpts0_f = mkpts0_c
        # 将DSNT的归一化坐标映射成像素偏移并放大到原图
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
        # data['stats_stage2_n'] = int(data['mkpts0_f'].shape[0]) 
         

    # 1-stage
    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape
        
        # 展平为长度W^2*W^2的相关向量，并取有效匹配数量
        conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']),...]
        # 每对窗口取top-1
        val, idx = torch.max(conf_matrix, dim = -1)
        idx = idx[:,None]
        # 将top1的线性索引还原为左右窗口的像素坐标索引
        idx_l, idx_r = idx // WW, idx % WW

        # 保存索引，供stage-2的索引3*3
        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        if self.fp16:
            #grid = create_meshgrid(W, W, False, conf_matrix.device)
            #grid = grid.to(torch.float16) - (W // 2) + 0.5
            # 构造以窗口中心为原点的像素级偏移网络，去除其中左右两侧的二维偏移量
            grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16)  - W // 2 + 0.5
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        # 将窗口内偏移按金字塔映射会原图坐标，得到像素级别的匹配点
        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        # 写会像素级别细化坐标
        if torch.is_tensor(scale0) and scale0.numel() > 1: # scale0 is a tensor
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
        else: # scale0 is a float
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)
        
        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f
        })
