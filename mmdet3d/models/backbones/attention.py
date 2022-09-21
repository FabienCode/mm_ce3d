import copy
import math

import torch
import torch.nn.functional as F
from mmdet.models import BACKBONES
from torch import nn as nn
import numpy as np
import torch

from torch.nn import functional as F

from mmdet3d.models.losses import chamfer_distance

from mmdet.core import multi_apply
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler
from mmcv.runner.dist_utils import master_only
from mmcv.utils.logging import get_logger, logger_initialized, print_log


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


@BACKBONES.register_module()
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]
        # debug_query = self.linears[0](query)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


@BACKBONES.register_module()
class teacher_attention(nn.Module):
    def __init__(self, input_channel, ratio=8):
        super(teacher_attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel // ratio, kernel_size=1,
                               bias=False)
        self.attention_bn1 = nn.BatchNorm1d(input_channel // ratio)

        self.conv2 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel // ratio, kernel_size=1,
                               bias=False)
        self.attention_bn2 = nn.BatchNorm1d(input_channel // ratio)

        self.conv3 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=1, bias=False)
        self.attention_bn3 = nn.BatchNorm1d(input_channel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        b, c, n = q.shape

        a = F.relu(self.attention_bn1(self.conv1(k))).permute(0, 2, 1)

        b = F.relu(self.attention_bn2(self.conv2(q)))  # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b)) / math.sqrt(c)  # b,n,n

        d = F.relu(self.attention_bn3(self.conv3(v)))  # b,c,n
        out = q + torch.bmm(d, s.permute(0, 2, 1))

        # mask = s

        results = out

        return results
@BACKBONES.register_module()
class pointnet(nn.Module):
    def __init__(self, input_channel, init_cfg=None):
        super(pointnet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        # n_pts = x.size()[1]
        x_trans = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x_trans)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x.transpose(2, 1)

@BACKBONES.register_module()
class pointnet_feat(nn.Module):
    def __init__(self, input_channel, init_cfg=None):
        super(pointnet_feat, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[1]
        x_trans = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x_trans)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        pointfeat = torch.cat([x, pointfeat], 1)
        global_feat = pointfeat
        pointfeat = self.conv4(pointfeat)
        pointfeat = pointfeat.transpose(2, 1)
        return global_feat, pointfeat


@BACKBONES.register_module()
class pointnet_cls(nn.Module):
    def __init__(self, input_channel=27, k=18, feature_transform=False):
        super(pointnet_cls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = pointnet_feat(input_channel=input_channel)
        self.feature_extra = pointnet(input_channel=input_channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        final_feature = self.feature_extra(x)
        x_feat, point_feature = self.feat(x)
        point_feature_per = x_feat
        x = F.relu(self.bn1(self.conv1(point_feature_per)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, final_feature


class ConvBNPositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): input features dim.
        num_pos_feats (int): output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) the embeded position features.
        """
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
