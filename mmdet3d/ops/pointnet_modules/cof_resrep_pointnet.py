# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from torch import nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32
from mmcv.ops import three_interpolate, three_nn
from mmdet3d.ops import PAConv
from .builder import SA_MODULES
# from mmdet3d.core.utils.cof_attention import BasicLayer
from typing import List
from mmdet3d.core.utils.cof_attention import BasicLayer
from mmdet3d.ops.pointnet_modules.rr_conv import RR_ConvModule


class cof_rr_BasePointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.
    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional):
            Range of points to apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        grouper_return_grouped_xyz (bool, optional): Whether to return
            grouped xyz in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool, optional): Whether to return
            grouped idx in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(cof_rr_BasePointSAModule, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

        self.cof_attentions1 = BasicLayer(mlp_channels[0][1], 2, 32)
        self.cof_attentions2 = BasicLayer(mlp_channels[0][2], 2, 32)
        self.cof_attentions3 = BasicLayer(mlp_channels[0][3], 2, 32)

        self.cof_channels_norm = []
        self.ori_layer_features = []
        self.cof_layer_features = []

    def _sample_points(self, points_xyz, features, indices, target_xyz):
        """Perform point sampling based on inputs.
        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.
        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.
        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        return new_xyz, indices

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.
        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.
        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(
            self,
            points_xyz,
            features=None,
            indices=None,
            target_xyz=None,
    ):
        """forward.
        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) features of each point.
                Default: None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor, optional): (B, M, 3) new coords of the outputs.
                Default: None.
        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        if self.ori_layer_features is not None:
            del self.ori_layer_features
            del self.cof_layer_features
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)

        # ori_layer_features = []
        self.cof_layer_features = []
        self.ori_layer_features = []
        self.cof_channels_norm = []
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)

            # (B, mlp[-1], num_point, nsample)
            for j in range(len(self.mlps[i])):
                tmp_new_feature = self.mlps[i][j](grouped_results)
                ori_layer_feature = self._pool_features(tmp_new_feature)
                self.ori_layer_features.append(ori_layer_feature)
                grouped_results = tmp_new_feature
            # for j in range(len(ori_layer_features)):
            #     cof_layer_feature = self.cof_attentions[i][j](new_xyz,
            #                                                   ori_layer_feature.permute(0, 2, 1).contiguous(),
            #                                                   new_xyz.shape[1])
            #     ln_cof_feature = F.layer_norm(cof_layer_feature, (cof_layer_feature.shape[-1],))
            #     cof_layer_norm = torch.sqrt(torch.sum(ln_cof_feature ** 2, dim=(0, 1)))
            #     self.cof_channels_norm.append(cof_layer_norm)
            # self.cof_channels_norm.append()
            cof_layer1_feature = self.cof_attentions1(new_xyz,
                                                      self.ori_layer_features[0].permute(0, 2, 1).contiguous(),
                                                      self.ori_layer_features[0].shape[1])
            self.cof_layer_features.append(cof_layer1_feature)
            ln_layer1_cof_feature = F.layer_norm(cof_layer1_feature, (cof_layer1_feature.shape[-1],))
            cof_layer1_norm = torch.sqrt(torch.sum(ln_layer1_cof_feature ** 2, dim=(0, 1)))
            self.cof_channels_norm.append(cof_layer1_norm)

            cof_layer2_feature = self.cof_attentions2(new_xyz,
                                                      self.ori_layer_features[1].permute(0, 2, 1).contiguous(),
                                                      self.ori_layer_features[1].shape[1])
            self.cof_layer_features.append(cof_layer2_feature)
            ln_layer2_cof_feature = F.layer_norm(cof_layer2_feature, (cof_layer2_feature.shape[-1],))
            cof_layer2_norm = torch.sqrt(torch.sum(ln_layer2_cof_feature ** 2, dim=(0, 1)))
            self.cof_channels_norm.append(cof_layer2_norm)

            cof_layer3_feature = self.cof_attentions3(new_xyz,
                                                      self.ori_layer_features[2].permute(0, 2, 1).contiguous(),
                                                      self.ori_layer_features[2].shape[1])

            self.cof_layer_features.append(cof_layer3_feature)
            ln_layer3_cof_feature = F.layer_norm(cof_layer3_feature, (cof_layer3_feature.shape[-1],))
            cof_layer3_norm = torch.sqrt(torch.sum(ln_layer3_cof_feature ** 2, dim=(0, 1)))
            self.cof_channels_norm.append(cof_layer3_norm)



            new_features_list.append(self.ori_layer_features[-1])
        return new_xyz, torch.cat(new_features_list, dim=1), indices, self.ori_layer_features, self.cof_layer_features


@SA_MODULES.register_module()
class cof_rr_PointSAModuleMSG(cof_rr_BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.
    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional): Range of points to
            apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(cof_rr_PointSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3

            mlp = nn.Sequential()
            cof_attention = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                if i == 0:
                    mlp.add_module(
                        f'layer{i}',
                        ConvModule(
                            mlp_channel[i],
                            mlp_channel[i + 1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=bias))
                else:
                    mlp.add_module(
                        f'layer{i}',
                        RR_ConvModule(
                            mlp_channel[i],
                            mlp_channel[i + 1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=bias))
                # cof_attention.add_module(f'layer{i}',
                #                          BasicLayer(mlp_channel[i + 1], 2, 32))
            self.mlps.append(mlp)
            # self.cof_attentions.append(cof_attention)


@SA_MODULES.register_module()
class cof_rr_PointSAModule(cof_rr_PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.
    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int, optional): Number of points.
            Default: None.
        radius (float, optional): Radius to group with.
            Default: None.
        num_sample (int, optional): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int], optional): Range of points
            to apply FPS. Default: [-1].
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False):
        super(cof_rr_PointSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)


class cof_rrc_PointFPModule(BaseModule):
    """Point feature propagation module used in PointNets.
    Propagate the features from one set to another.
    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: dict = dict(type='BN2d'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # self.mlp_channels = mlp_channels
        self.fp16_enabled = False
        self.mlps = nn.Sequential()
        # self.xyz_embeddings = nn.Linear(3, mlp_channels[-1])
        # self.cof_attention = BasicLayer(mlp_channels[-1], 2, 32)
        self.cof_attentions = nn.Sequential()
        for i in range(len(mlp_channels) - 1):
            if i == 0:
                self.mlps.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channels[i],
                        mlp_channels[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg))
            else:
                self.mlps.add_module(
                    f'layer{i}',
                    RR_ConvModule(
                        mlp_channels[i],
                        mlp_channels[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg))
            self.cof_attentions.add_module(f'layer{i}',
                                           BasicLayer(mlp_channels[i + 1], 2, 32))
        self.cof_channels_norm = []
        self.ori_layer_features = []
        self.cof_layer_features = []

    @force_fp32()
    def forward(self, target: torch.Tensor, source: torch.Tensor,
                target_feats: torch.Tensor,
                source_feats: torch.Tensor) -> torch.Tensor:
        """forward.
        Args:
            target (Tensor): (B, n, 3) tensor of the xyz positions of
                the target features.
            source (Tensor): (B, m, 3) tensor of the xyz positions of
                the source features.
            target_feats (Tensor): (B, C1, n) tensor of the features to be
                propagated to.
            source_feats (Tensor): (B, C2, m) tensor of features
                to be propagated.
        Return:
            Tensor: (B, M, N) M = mlp[-1], tensor of the target features.
        """
        # if self.ori_layer_features is not None:
        #     del self.ori_layer_features
        #     del self.cof_layer_features
        self.ori_layer_features = []
        self.cof_layer_features = []
        self.cof_channels_norm = []
        if source is not None:
            dist, idx = three_nn(target, source)
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm

            interpolated_feats = three_interpolate(source_feats, idx, weight)
        else:
            interpolated_feats = source_feats.expand(*source_feats.size()[0:2],
                                                     target.size(1))

        if target_feats is not None:
            new_features = torch.cat([interpolated_feats, target_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats


        # test_layer_fetures = []
        new_features = new_features.unsqueeze(-1)
        for i in range(len(self.mlps)):
            ori_layer_feature = self.mlps[i](new_features)
            # test_layer_fetures.append(ori_layer_feature.squeeze(-1))
            self.ori_layer_features.append(ori_layer_feature.squeeze(-1))
            cof_layer_feature = self.cof_attentions[i](target,
                                                       ori_layer_feature.squeeze(-1).permute(0, 2, 1).contiguous(),
                                                       target.shape[1])
            self.cof_layer_features.append(cof_layer_feature)
            ln_cof_feature = F.layer_norm(cof_layer_feature, (cof_layer_feature.shape[-1],))
            cof_layer_norm = torch.sqrt(torch.sum(ln_cof_feature ** 2, dim=(0, 1)))
            self.cof_channels_norm.append(cof_layer_norm)
            new_features = ori_layer_feature



        return self.ori_layer_features[-1], self.ori_layer_features, self.cof_layer_features

class cof_rr_PointFPModule(BaseModule):
    """Point feature propagation module used in PointNets.
    Propagate the features from one set to another.
    Args:
        mlp_channels (list[int]): List of mlp channels.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 mlp_channels: List[int],
                 norm_cfg: dict = dict(type='BN2d'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # self.mlp_channels = mlp_channels
        self.fp16_enabled = False
        self.mlps = nn.Sequential()
        # self.xyz_embeddings = nn.Linear(3, mlp_channels[-1])
        # self.cof_attention = BasicLayer(mlp_channels[-1], 2, 32)
        self.cof_attentions = nn.Sequential()
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

            self.cof_attentions.add_module(f'layer{i}',
                                           BasicLayer(mlp_channels[i + 1], 2, 32))
        self.cof_channels_norm = []
        self.ori_layer_features = []
        self.cof_layer_features = []

    @force_fp32()
    def forward(self, target: torch.Tensor, source: torch.Tensor,
                target_feats: torch.Tensor,
                source_feats: torch.Tensor) -> torch.Tensor:
        """forward.
        Args:
            target (Tensor): (B, n, 3) tensor of the xyz positions of
                the target features.
            source (Tensor): (B, m, 3) tensor of the xyz positions of
                the source features.
            target_feats (Tensor): (B, C1, n) tensor of the features to be
                propagated to.
            source_feats (Tensor): (B, C2, m) tensor of features
                to be propagated.
        Return:
            Tensor: (B, M, N) M = mlp[-1], tensor of the target features.
        """
        # if self.ori_layer_features is not None:
        #     del self.ori_layer_features
        #     del self.cof_layer_features
        self.ori_layer_features = []
        self.cof_layer_features = []
        self.cof_channels_norm = []
        if source is not None:
            dist, idx = three_nn(target, source)
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm

            interpolated_feats = three_interpolate(source_feats, idx, weight)
        else:
            interpolated_feats = source_feats.expand(*source_feats.size()[0:2],
                                                     target.size(1))

        if target_feats is not None:
            new_features = torch.cat([interpolated_feats, target_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats


        # test_layer_fetures = []
        new_features = new_features.unsqueeze(-1)
        for i in range(len(self.mlps)):
            ori_layer_feature = self.mlps[i](new_features)
            # test_layer_fetures.append(ori_layer_feature.squeeze(-1))
            self.ori_layer_features.append(ori_layer_feature.squeeze(-1))
            cof_layer_feature = self.cof_attentions[i](target,
                                                       ori_layer_feature.squeeze(-1).permute(0, 2, 1).contiguous(),
                                                       target.shape[1])
            self.cof_layer_features.append(cof_layer_feature)
            ln_cof_feature = F.layer_norm(cof_layer_feature, (cof_layer_feature.shape[-1],))
            cof_layer_norm = torch.sqrt(torch.sum(ln_cof_feature ** 2, dim=(0, 1)))
            self.cof_channels_norm.append(cof_layer_norm)
            new_features = ori_layer_feature



        return self.ori_layer_features[-1], self.ori_layer_features, self.cof_layer_features
