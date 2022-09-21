# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector
from .label_prior import LabelPrior

import torch.nn.functional as F
from mmdet3d.core.utils.cof_attention import BasicLayer
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class VoxelNet_pcpa(LabelPrior):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 label_backbone=None,
                 gt_backbone=None,
                 # gt_attention=None,
                 neck=None,
                 label_neck = None,
                 gt_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet_pcpa, self).__init__(
            backbone=backbone,
            label_backbone=label_backbone,
            gt_backbone=gt_backbone,
            # gt_attention=gt_attention,
            neck=neck,
            label_neck=label_neck,
            gt_neck=gt_neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.label_voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.label_middle_encoder = builder.build_middle_encoder(middle_encoder)


        self.gt_voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.gt_middle_encoder = builder.build_middle_encoder(middle_encoder)
        # self.label_neck = build_neck(label_neck)
        # self.gt_neck = build_neck(gt_neck)



        self.gt_attention = BasicLayer(384, 2, 32)
        for p in self.label_backbone.parameters():
            p.requires_grad = False
        #
        for p in self.gt_backbone.parameters():
            p.requires_grad = False

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_teacher_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.label_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.label_middle_encoder(voxel_features, coors, batch_size)
        x = self.label_backbone(x)
        if self.with_neck:
            x = self.label_neck(x)
        return x

    def extract_gt_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.gt_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.gt_middle_encoder(voxel_features, coors, batch_size)
        x = self.gt_backbone(x)
        if self.with_neck:
            x = self.gt_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      gt_points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        B, C, H, W = x[0].shape

        # with torch.no_grad():
        gt_pc_x = self.extract_gt_feat(gt_points, img_metas)

        t_x = self.extract_teacher_feat(points, img_metas)
        #
        gt_a = gt_pc_x[0].reshape(B, C, -1)
        t_a = t_x[0].reshape(B, C, -1)

        t_a = self.gt_attention(t_a.permute(0, 2, 1).contiguous(), gt_a.permute(0, 2, 1).contiguous(),
                                t_a.shape[2]).permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        t_a = t_a.reshape(B, C, H, W)
        tx = [t_x[0] + t_a]
        # tx = gt_pc_x
        outs = self.bbox_head(tx)
        t_loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        t_losses = self.bbox_head.loss(
            *t_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for k, v in t_losses.items():
            losses['label_' + k] = v

        label_encodings = x[0].detach()
        features = tx[0]

        losses['dist_feature_loss'] = 1 * F.mse_loss(label_encodings, features)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
