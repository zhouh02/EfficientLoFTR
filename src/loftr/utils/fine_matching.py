import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm - Enhanced with CoMatch bilateral refinement"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C] where WW = (W+2)^2
            feat1 (torch.Tensor): [M, WW, C] where WW = (W+2)^2
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW)) - 2  # 去除 padding，得到实际窗口大小
        WW_inner = W ** 2  # 内部窗口大小
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW_inner, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW_inner, WW_inner, device=feat_0.device),
                'mkpts0_f': data['all_mkpts0_c'][:len(data['mconf']),...],
                'mkpts1_f': data['all_mkpts1_c'][:len(data['mconf']),...],
                'all_mkpts0_f': data['all_mkpts0_c'],
                'all_mkpts1_f': data['all_mkpts1_c'],
            })
            return

        # 分离匹配特征和回归特征
        feat_f0 = feat_0[..., :-self.local_regress_slicedim]
        feat_f1 = feat_1[..., :-self.local_regress_slicedim]
        feat_ff0 = feat_0[..., -self.local_regress_slicedim:]
        feat_ff1 = feat_1[..., -self.local_regress_slicedim:]

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            # 归一化匹配特征
            feat_f0_norm = feat_f0 / (feat_f0.shape[-1] ** 0.5)
            feat_f1_norm = feat_f1 / (feat_f1.shape[-1] ** 0.5)
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0_norm, feat_f1_norm)

        # 应用 dual-softmax
        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        
        # 重塑为 5D 张量并去除 padding
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.W+2, self.W+2, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1, 1:-1, 1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'conf_matrix_f': softmax_matrix_f})

        # Stage 1: 计算 pixel-level 的初步匹配
        self.get_fine_ds_match(softmax_matrix_f, data)

        # Stage 2: 双向回归 - CoMatch 的核心改进
        idx_l, idx_r = data['idx_l'], data['idx_r']
        
        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['all_mkpts0_c'][:len(data['mconf']),...],
                'mkpts1_f': data['all_mkpts1_c'][:len(data['mconf']),...],
                'all_mkpts0_f': data['all_mkpts0_c'],
                'all_mkpts1_f': data['all_mkpts1_c'],
            })
            return

        # 生成 3x3 邻域网格
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        idx_l_iids, idx_l_jids = idx_l // W, idx_l % W
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W
        m_ids, idx_l_iids, idx_l_jids, idx_r_iids, idx_r_jids = (
            m_ids.reshape(-1), 
            idx_l_iids.reshape(-1), 
            idx_l_jids.reshape(-1), 
            idx_r_iids.reshape(-1), 
            idx_r_jids.reshape(-1)
        )

        # 创建 3x3 的偏移网格
        delta = create_meshgrid(3, 3, True, softmax_matrix_f.device).to(torch.long)  # [1, 3, 3, 2]
        
        m_ids_expanded = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l_iids_grid = idx_l_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_l_jids_grid = idx_l_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]
        idx_r_iids_grid = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids_grid = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        # 提取回归特征并重塑
        feat_ff0 = feat_ff0.reshape(M, self.W+2, self.W+2, self.local_regress_slicedim)
        feat_ff1 = feat_ff1.reshape(M, self.W+2, self.W+2, self.local_regress_slicedim)
        
        # 在 3x3 邻域内提取特征
        feat_ff0_local = feat_ff0[m_ids_expanded, idx_l_iids_grid, idx_l_jids_grid].view(-1, 9, self.local_regress_slicedim)
        feat_ff1_local = feat_ff1[m_ids_expanded, idx_r_iids_grid, idx_r_jids_grid].view(-1, 9, self.local_regress_slicedim)

        # CoMatch 核心：使用中心点特征的平均值作为查询向量
        feat_ff0_center = feat_ff0_local[:, 4, :]  # 中心点 index = 4 (9//2)
        feat_ff1_center = feat_ff1_local[:, 4, :]
        avg_feat_center = (feat_ff0_center + feat_ff1_center) / 2.0

        # 双向计算 confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            # 归一化
            feat_ff0_local_norm = feat_ff0_local / (self.local_regress_slicedim ** 0.5)
            feat_ff1_local_norm = feat_ff1_local / (self.local_regress_slicedim ** 0.5)
            
            # 对左图的 3x3 邻域计算相似度
            conf_matrix_ff0 = torch.einsum('mc,mrc->mr', avg_feat_center, feat_ff0_local_norm)
            # 对右图的 3x3 邻域计算相似度
            conf_matrix_ff1 = torch.einsum('mc,mrc->mr', avg_feat_center, feat_ff1_local_norm)

        # 应用温度参数并 softmax
        conf_matrix_ff0 = conf_matrix_ff0.reshape(-1, 9)
        conf_matrix_ff1 = conf_matrix_ff1.reshape(-1, 9)
        heatmap0 = F.softmax(conf_matrix_ff0 / self.local_regress_temperature, -1).reshape(-1, 3, 3)
        heatmap1 = F.softmax(conf_matrix_ff1 / self.local_regress_temperature, -1).reshape(-1, 3, 3)

        # 从 heatmap 计算归一化坐标 (使用 DSNT)
        coords_normalized0 = dsnt.spatial_expectation2d(heatmap0[None], True)[0]
        coords_normalized1 = dsnt.spatial_expectation2d(heatmap1[None], True)[0]
        
        # 处理 scale
        if data['bs'] == 1:
            scale0 = scale * data['scale0'] if 'scale0' in data else scale
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale0 = scale * data['scale0'][data['b_ids']][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale
            scale1 = scale * data['scale1'][data['b_ids']][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

        # 计算最终的亚像素级坐标
        self.get_fine_match_bilateral(coords_normalized0, coords_normalized1, data, scale0, scale1)

    def get_fine_match_bilateral(self, coords_normed0, coords_normed1, data, scale0, scale1):
        """
        双向更新匹配点坐标 - CoMatch 的核心改进
        
        Args:
            coords_normed0: 左图的归一化偏移 [-1, 1]
            coords_normed1: 右图的归一化偏移 [-1, 1]
            scale0, scale1: 缩放因子
        """
        all_mkpts0_c = data['all_mkpts0_c']
        all_mkpts1_c = data['all_mkpts1_c']
        
        # 将归一化坐标转换为实际偏移（3x3 网格，所以范围是 [-1, 1] 对应 [-1, 1] 个像素）
        # coords_normed 在 [-1, 1] 范围内，对应 3x3 网格的 [-1, 1] 像素偏移
        offset0 = coords_normed0 * (3 // 2) * scale0  # (3//2) = 1，表示最大偏移 1 个像素
        offset1 = coords_normed1 * (3 // 2) * scale1
        
        # 双向更新：左图和右图都根据各自的 heatmap 更新
        all_mkpts0_f = all_mkpts0_c + offset0
        all_mkpts1_f = all_mkpts1_c + offset1

        data.update({
            "mkpts0_f": all_mkpts0_f[:len(data['mconf']),...],
            "mkpts1_f": all_mkpts1_f[:len(data['mconf']),...],
            "all_mkpts0_f": all_mkpts0_f,
            "all_mkpts1_f": all_mkpts1_f
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        """
        从 confidence matrix 中获取初步匹配
        """
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        # 找到最大值的索引
        conf_matrix_flat = conf_matrix.reshape(m, -1)
        _, idx = torch.max(conf_matrix_flat, dim=-1)
        idx = idx[:, None]
        idx_l, idx_r = idx // WW, idx % WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        # 创建网格用于计算偏移
        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        # 处理 scale
        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        # 计算粗略的匹配点坐标
        if torch.is_tensor(scale0) and scale0.numel() > 1:
            all_mkpts0_f = (data['all_mkpts0_c'][:, None, :] + (delta_l * scale0[:, None, :])).reshape(-1, 2)
            all_mkpts1_f = (data['all_mkpts1_c'][:, None, :] + (delta_r * scale1[:, None, :])).reshape(-1, 2)
        else:
            all_mkpts0_f = (data['all_mkpts0_c'][:, None, :] + (delta_l * scale0)).reshape(-1, 2)
            all_mkpts1_f = (data['all_mkpts1_c'][:, None, :] + (delta_r * scale1)).reshape(-1, 2)
        
        # 更新 data（这些是粗略匹配，会被双向回归进一步优化）
        data.update({
            "mkpts0_c": all_mkpts0_f[:len(data['mconf']),...],
            "mkpts1_c": all_mkpts1_f[:len(data['mconf']),...],
            "all_mkpts0_c": all_mkpts0_f,
            "all_mkpts1_c": all_mkpts1_f
        })