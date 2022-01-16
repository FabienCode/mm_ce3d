import torch
import torch.nn as nn
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .single_stage import SingleStage3DDetector
from .. import builder
import torch.nn.functional as F


@DETECTORS.register_module()
class VoteNet(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 label_backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNet, self).__init__(
            backbone=backbone,
            label_backbone=label_backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

        # self.feature_adapt = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        # )
        # self.layernorm = nn.GroupNorm(num_groups=1, num_channels=256, affine=False)


    def forward_train(self,
                      points,
                      gt_points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_points:
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        ori_feature = self.backbone(points_cat)

        bbox_preds = self.bbox_head(ori_feature, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # label computer
        gt_points_cat = torch.stack(gt_points)
        label_feature = self.label_backbone(gt_points_cat)
        label_bbox_preds = self.bbox_head(label_feature, self.train_cfg.sample_mod)
        label_loss_inputs = (gt_points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        label_losses = self.bbox_head.loss(
            label_bbox_preds, *label_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for k, v in label_losses.items():
            losses['label_' + k] = v
        # calculate distance loss
        # with torch.no_grad():
        #     n = label_feature['fp_features'][-1].size(0)
        #     label_encodings = [self.layernorm(i) for i in label_feature['fp_features']]
        #     label_encodings = torch.cat([i.view(n, 256, -1) for i in label_encodings], dim=2)
        #     label_encodings.detach_()
        # # n = label_feature['fp_features'][-1].size(0)
        # # label_encodings = [self.layernorm(i) for i in label_feature['fp_features']]
        # # label_encodings = torch.cat([i.view(n, 256, -1) for i in label_encodings], dim=2)
        # features = [self.feature_adapt(i.unsqueeze(-1)) for i in ori_feature['fp_features']]
        # features = [self.layernorm(i.squeeze(-1)) for i in features]
        # # # features = [self.layernorm((self.feature_adapt(i.unsqueeze(-1))).squeeze(-1) for i in ori_feature['fp_features'])]
        # features = torch.cat([i.view(n, 256, -1) for i in features], dim=2)
        #
        # for i in range(len(label_feature['fp_features'])):
        #     losses['dict_fp_loss_' + str(i)] = F.mse_loss(label_feature['fp_features'][i],
        #                                                   ori_feature['fp_features'][i])
        # for i in range(len(label_feature['fp_features'])):
        #     losses['dict_fp_loss_' + str(i)] = 0.1 * i * (F.mse_loss(label_feature['fp_features'][i], ori_feature['fp_features'][i]))
        # for i in range(len(label_feature['sa_features'])):
        #     losses['dict_sa_loss_' + str(i)] = 0.1 * i * (F.mse_loss(label_feature['sa_features'][i], ori_feature['sa_features'][i]))
        label_encodings = label_feature['fp_features'][-1]
        label_encodings.detach()
        features = ori_feature['fp_features'][-1]
        losses['dict_fp_loss'] = F.mse_loss(label_encodings, features)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.backbone(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
