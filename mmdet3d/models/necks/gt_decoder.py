import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn
from typing import List
from mmdet.models import HEADS
from mmdet3d.ops import three_interpolate, three_nn

@HEADS.register_module()
class Gt2para(nn.Module):
    """Point feature propagation module used in PointNets.

    Propagate the features from one set to another.

    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: dict = dict(type='BN2d'),
                 init_cfg=None):
        super().__init__()
        self.fp16_enabled = False
        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            self.mlps.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_channels[i],
                    mlp_channels[i + 1],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg))

    @force_fp32()
    def forward(self, features):
        """forward.
        """

        new_features = self.mlps(features)

        return new_features