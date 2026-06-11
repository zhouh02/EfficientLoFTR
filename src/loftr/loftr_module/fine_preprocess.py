import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from loguru import logger

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    """
    Fine-level feature preprocessing with CoMatch-style unified patch size
    使用统一的 W+2 尺寸来支持双向回归
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        block_dims = config['backbone']['block_dims']
        self.W = self.config['fine_window_size']
        self.fine_d_model = block_dims[0]

        # FPN 风格的特征融合网络
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1, stride):
        """
        特征金字塔网络，自底向上融合多尺度特征
        
        Args:
            feat_c: coarse feature (1/8 resolution)
            x2: intermediate feature (1/4 resolution)
            x1: fine feature (1/2 resolution)
            stride: downsample stride
        """
        # Layer 3 -> Layer 2
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2 + feat_c)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        # Layer 2 -> Layer 1
        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1 + x2)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        
        return x1
    
    def forward(self, feat_c0, feat_c1, data):
        """
        Args:
            feat_c0, feat_c1: coarse-level features
            data: data dictionary
        
        Returns:
            feat_f0, feat_f1: fine-level features with shape [n, (W+2)^2, c]
        """
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        
        if data['b_ids'].shape[0] == 0:
            # 空匹配情况
            feat0 = torch.empty(0, (self.W+2)**2, self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, (self.W+2)**2, self.fine_d_model, device=feat_c0.device)
            return feat0, feat1

        if data['hw0_i'] == data['hw1_i']:
            # 相同输入尺寸的情况
            feat_c = rearrange(
                torch.cat([feat_c0, feat_c1], 0), 
                'b (h w) c -> b c h w', 
                h=data['hw0_c'][0]
            )  # 1/8 feat
            x2 = data['feats_x2']  # 1/4 feat
            x1 = data['feats_x1']  # 1/2 feat
            del data['feats_x2'], data['feats_x1']

            # 1. FPN 特征提取
            x1 = self.inter_fpn(feat_c, x2, x1, stride)
            feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

            # 2. 使用 W+2 的统一尺寸进行 unfold（关键改进）
            # padding=1 确保边界信息不丢失，支持双向回归
            feat_f0 = F.unfold(feat_f0, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
            
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. 选择预测的匹配
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, (W+2)^2, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

            return feat_f0, feat_f1
            
        else:
            # 处理不同输入尺寸的情况
            feat_c0 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0])
            feat_c1 = rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0])
            
            x2_0, x2_1 = data['feats_x2_0'], data['feats_x2_1']  # 1/4 feat
            x1_0, x1_1 = data['feats_x1_0'], data['feats_x1_1']  # 1/2 feat
            del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']

            # 1. 分别进行 FPN 特征提取
            feat_f0 = self.inter_fpn(feat_c0, x2_0, x1_0, stride)
            feat_f1 = self.inter_fpn(feat_c1, x2_1, x1_1, stride)

            # 2. 使用 W+2 的统一尺寸进行 unfold
            feat_f0 = F.unfold(feat_f0, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=(W+2)**2)
            
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. 选择预测的匹配
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, (W+2)^2, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

            return feat_f0, feat_f1