import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .label_prior import LabelPrior
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@DETECTORS.register_module()
class Votepcpa(LabelPrior):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 label_backbone=None,
                 gt_backbone=None,
                 gt_attention=None,
                 pointnet=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(Votepcpa, self).__init__(
            backbone=backbone,
            label_backbone=label_backbone,
            gt_backbone=gt_backbone,
            gt_attention=gt_attention,
            pointnet=pointnet,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

        # for p in self.label_backbone.parameters():
        #     p.requires_grad = False
        # #
        # for p in self.gt_backbone.parameters():
        #     p.requires_grad = False
        # #
        # for p in self.pointnet.parameters():
        #     p.requires_grad = False
        #
        # for p in self.gt_attention.parameters():
        #     p.requires_grad = False

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
        points_cat = torch.stack(points)  # (B,N,4)

        # x = self.extract_feat(points_cat)
        student_feature = self.backbone(points_cat)  # ['fp_features']:(B,C,N)
        bbox_preds = self.bbox_head(student_feature, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # gt_para
        # gt_para = self.extract_para_sunrgbd(gt_bboxes_3d, gt_labels_3d).cuda()
        # gt_para = self.extract_para_scannet(gt_bboxes_3d, gt_labels_3d).cuda()
        gt_para = self.extract_para_kitti(gt_bboxes_3d, gt_labels_3d).cuda()
        gt_para_feature = self.pointnet(gt_para)

        # gt_pc
        gt_points_cat = torch.stack(gt_points)
        gt_feature = self.gt_backbone(gt_points_cat)

        # gt_pc & gt_para attention
        gt_feature['sa_features'][-1] = self.gt_attention(gt_feature['sa_features'][-1],
                                                          gt_para_feature,
                                                          gt_para_feature)

        # teacher compute
        teacher_feature = self.label_backbone(points_cat)  # label branch ori points feature

        teacher_feature['sa_features'][-1] = self.gt_attention(teacher_feature['sa_features'][-1],
                                                               gt_feature['sa_features'][-1],
                                                               gt_feature['sa_features'][-1])
        label_bbox_pred = self.bbox_head(teacher_feature, self.train_cfg.sample_mod)
        label_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask, img_metas)
        label_losses = self.bbox_head.loss(label_bbox_pred, *label_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for k, v in label_losses.items():
            losses['label_' + k] = v

        label_encodings = teacher_feature['sa_features'][-1].detach()

        features = student_feature['sa_features'][-1]

        losses['dist_feature_loss'] = 1 * F.mse_loss(label_encodings, features)

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

        x = self.extract_feat(points_cat)
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

    def extract_para_scannet(self, gt_bboxes_3d, gt_labels_3d):
        mean_sizes = [[0.76966727, 0.8116021, 0.92573744],
                      [1.876858, 1.8425595, 1.1931566],
                      [0.61328, 0.6148609, 0.7182701],
                      [1.3955007, 1.5121545, 0.83443564],
                      [0.97949594, 1.0675149, 0.6329687],
                      [0.531663, 0.5955577, 1.7500148],
                      [0.9624706, 0.72462326, 1.1481868],
                      [0.83221924, 1.0490936, 1.6875663],
                      [0.21132214, 0.4206159, 0.5372846],
                      [1.4440073, 1.8970833, 0.26985747],
                      [1.0294262, 1.4040797, 0.87554324],
                      [1.3766412, 0.65521795, 1.6813129],
                      [0.6650819, 0.71111923, 1.298853],
                      [0.41999173, 0.37906948, 1.7513971],
                      [0.59359556, 0.5912492, 0.73919016],
                      [0.50867593, 0.50656086, 0.30136237],
                      [1.1511526, 1.0546296, 0.49706793],
                      [0.47535285, 0.49249494, 0.5802117]]
        bbox_info = list()
        for i in range(len(gt_bboxes_3d)):
            size_class_target = gt_labels_3d[i]

            size_res_target = gt_bboxes_3d[i].dims - gt_bboxes_3d[i].tensor.new_tensor(
                mean_sizes)[size_class_target]
            # log_size_res_target = torch.zeros((size_res_target.shape[0], 3))
            # for a in range(size_res_target.shape[0]):
            #     for b in range(3):
            #         log_size_res_target[a][b] = torch.log(size_res_target[a][b].abs())

            # log_size_res_target = torch.log(size_res_target)

            # (dir_class_target, dir_res_target) = self.angle2class(gt_bboxes_3d[i].yaw) # sunrgbd
            box_num = gt_labels_3d[i].shape[0]
            dir_class_target = gt_labels_3d[i].new_zeros(box_num)  # scannet
            dir_res_target = gt_bboxes_3d[i].tensor.new_zeros(box_num)  # scannet
            size_class_target = size_class_target.reshape(box_num, 1)
            dir_class_target = dir_class_target.reshape(box_num, 1)
            dir_res_target = dir_res_target.reshape(box_num, 1)
            new_center = gt_bboxes_3d[i].gravity_center.cuda()
            # rx = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # ry = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # rz = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # center_aug = torch.cat((rx, ry, rz)).repeat(gt_bboxes_3d[i].gravity_center.shape[0], 1)
            # center_aug = center_aug * gt_bboxes_3d[i].dims
            # new_center = new_center + center_aug
            # new_center = gt_bboxes_3d[i].gravity_center

            tmp_info = torch.cat(
                (new_center.cuda(), size_res_target.cuda(), size_class_target,  # log dim
                 dir_class_target.cuda(), dir_res_target.cuda()),
                1)
            bbox_info.append(tmp_info)
        # # bbox_info = DC(gt_bboxes_3d.tensor)
        # bbox_info_cat = torch.stack(bbox_info)
        label_one_hot = list()
        for i in range(len(gt_labels_3d)):
            # tmp_label_one_hot = to_one_hot(gt_labels_3d[i], 10)
            gt_labels_3d_tmp = gt_labels_3d[i].reshape(gt_labels_3d[i].shape[0], 1).cuda()
            tmp_label_one_hot = torch.zeros(gt_labels_3d_tmp.shape[0], 18).cuda().scatter_(1, gt_labels_3d_tmp, 1)
            tmp_label_one_hot = F.softmax(tmp_label_one_hot, dim=-1)
            # tmp_label_one_hot = self.embedding(gt_labels_3d[i].cuda())

            label_one_hot.append(tmp_label_one_hot)
        # bbox_label = torch.stack(gt_labels_3d)

        gt_para = list()
        for i in range(len(gt_labels_3d)):
            gt_para.append(torch.zeros((256, 27)))
            tmp = torch.cat((bbox_info[i].cuda(), label_one_hot[i].cuda()), 1)
            for j in range(tmp.shape[0]):
                gt_para[i][j] = tmp[j]

        gt_para_cat = torch.stack(gt_para)
        return gt_para_cat

    def extract_para_sunrgbd(self, gt_bboxes_3d, gt_labels_3d):
        mean_sizes = [
            [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
            [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
            [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
            [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
            [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
        ]
        bbox_info = list()
        for i in range(len(gt_bboxes_3d)):
            size_class_target = gt_labels_3d[i]

            size_res_target = gt_bboxes_3d[i].dims - gt_bboxes_3d[i].tensor.new_tensor(
                mean_sizes)[size_class_target]
            # log_size_res_target = torch.zeros((size_res_target.shape[0], 3))
            # for a in range(size_res_target.shape[0]):
            #     for b in range(3):
            #         log_size_res_target[a][b] = torch.log(size_res_target[a][b].abs())

            # log_size_res_target = torch.log(size_res_target)

            # (dir_class_target, dir_res_target) = self.angle2class(gt_bboxes_3d[i].yaw) # sunrgbd
            box_num = gt_labels_3d[i].shape[0]
            dir_class_target = gt_labels_3d[i].new_zeros(box_num)  # scannet
            dir_res_target = gt_bboxes_3d[i].tensor.new_zeros(box_num)  # scannet
            size_class_target = size_class_target.reshape(box_num, 1)
            dir_class_target = dir_class_target.reshape(box_num, 1)
            dir_res_target = dir_res_target.reshape(box_num, 1)
            new_center = gt_bboxes_3d[i].gravity_center
            # rx = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # ry = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # rz = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # center_aug = torch.cat((rx, ry, rz)).repeat(gt_bboxes_3d[i].gravity_center.shape[0], 1)
            # center_aug = center_aug * gt_bboxes_3d[i].dims
            # new_center = new_center + center_aug
            # new_center = gt_bboxes_3d[i].gravity_center

            tmp_info = torch.cat(
                (new_center.cuda(), size_res_target.cuda(), size_class_target,  # log dim
                 dir_class_target.cuda(), dir_res_target.cuda()),
                1)
            bbox_info.append(tmp_info)
        # # bbox_info = DC(gt_bboxes_3d.tensor)
        # bbox_info_cat = torch.stack(bbox_info)
        label_one_hot = list()
        for i in range(len(gt_labels_3d)):
            # tmp_label_one_hot = to_one_hot(gt_labels_3d[i], 10)
            gt_labels_3d_tmp = gt_labels_3d[i].reshape(gt_labels_3d[i].shape[0], 1).cuda()
            tmp_label_one_hot = torch.zeros(gt_labels_3d_tmp.shape[0], 10).cuda().scatter_(1, gt_labels_3d_tmp, 1)
            tmp_label_one_hot = F.softmax(tmp_label_one_hot, dim=-1)
            # tmp_label_one_hot = self.embedding(gt_labels_3d[i].cuda())

            label_one_hot.append(tmp_label_one_hot)
        # bbox_label = torch.stack(gt_labels_3d)

        gt_para = list()
        for i in range(len(gt_labels_3d)):
            gt_para.append(torch.zeros((256, 19)))
            tmp = torch.cat((bbox_info[i].cuda(), label_one_hot[i].cuda()), 1)
            for j in range(tmp.shape[0]):
                gt_para[i][j] = tmp[j]

        gt_para_cat = torch.stack(gt_para)
        return gt_para_cat

    def extract_para_kitti(self, gt_bboxes_3d, gt_labels_3d):
        # mean_sizes = [
        #     [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
        #     [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
        #     [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
        #     [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
        #     [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
        # ]
        bbox_info = list()
        for i in range(len(gt_bboxes_3d)):
            size_class_target = gt_labels_3d[i]

            size_res_target = gt_bboxes_3d[i].dims
            box_num = gt_labels_3d[i].shape[0]
            dir_class_target = gt_labels_3d[i].new_zeros(box_num)  # scannet
            dir_res_target = gt_bboxes_3d[i].tensor.new_zeros(box_num)  # scannet
            size_class_target = size_class_target.reshape(box_num, 1)
            dir_class_target = dir_class_target.reshape(box_num, 1)
            dir_res_target = dir_res_target.reshape(box_num, 1)
            new_center = gt_bboxes_3d[i].gravity_center
            # rx = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # ry = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # rz = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # center_aug = torch.cat((rx, ry, rz)).repeat(gt_bboxes_3d[i].gravity_center.shape[0], 1)
            # center_aug = center_aug * gt_bboxes_3d[i].dims
            # new_center = new_center + center_aug
            # new_center = gt_bboxes_3d[i].gravity_center

            tmp_info = torch.cat(
                (new_center.cuda(), size_res_target.cuda(), size_class_target,  # log dim
                 dir_class_target.cuda(), dir_res_target.cuda()),
                1)
            bbox_info.append(tmp_info)
        # # bbox_info = DC(gt_bboxes_3d.tensor)
        # bbox_info_cat = torch.stack(bbox_info)
        # label_one_hot = list()
        # for i in range(len(gt_labels_3d)):
        #     # tmp_label_one_hot = to_one_hot(gt_labels_3d[i], 10)
        #     gt_labels_3d_tmp = gt_labels_3d[i].reshape(gt_labels_3d[i].shape[0], 1).cuda()
        #     tmp_label_one_hot = torch.zeros(gt_labels_3d_tmp.shape[0], 10).cuda().scatter_(1, gt_labels_3d_tmp, 1)
        #     tmp_label_one_hot = F.softmax(tmp_label_one_hot, dim=-1)
        #     # tmp_label_one_hot = self.embedding(gt_labels_3d[i].cuda())
        #
        #     label_one_hot.append(tmp_label_one_hot)
        # bbox_label = torch.stack(gt_labels_3d)

        gt_para = list()
        for i in range(len(gt_labels_3d)):
            gt_para.append(torch.zeros((256, 9)))
            # tmp = torch.cat((bbox_info[i].cuda(), label_one_hot[i].cuda()), 1)
            for j in range(bbox_info[i].shape[0]):
                gt_para[i][j] = bbox_info[i][j]

        gt_para_cat = torch.stack(gt_para)
        return gt_para_cat


def KLDivergenceLoss(y, teacher_scores, mask=None, T=1):
    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(y / T, dim=1)[mask]
            q = F.softmax(teacher_scores / T, dim=1)[mask]
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum()
        else:
            loss = torch.Tensor([0]).cuda()
    else:
        p = F.log_softmax(y / T, dim=1)
        q = F.softmax(teacher_scores / T, dim=1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    return loss * T ** 2


def to_one_hot(y, n_class):
    return torch.eye(n_class)[y]


def angle2class(angle):
    """Convert continuous angle to a discrete class and a residual.

    Convert continuous angle to a discrete class and a small
    regression number from class center angle to current angle.

    Args:
        angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
            class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

    Returns:
        tuple: Encoded discrete class and residual.
    """
    angle = angle % (2 * np.pi)
    num_dir_bins = 12
    angle_per_class = 2 * np.pi / float(num_dir_bins)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    angle_cls = shifted_angle // angle_per_class
    angle_res = shifted_angle - (
            angle_cls * angle_per_class + angle_per_class / 2)
    return angle_cls.long(), angle_res


def extract_para_sunrgbd(gt_bboxes_3d, gt_labels_3d):
    mean_sizes = [
        [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
        [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
        [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
        [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
        [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
    ]
    bbox_info = list()
    for i in range(len(gt_bboxes_3d)):
        size_class_target = gt_labels_3d[i]

        size_res_target = gt_bboxes_3d[i].dims - gt_bboxes_3d[i].tensor.new_tensor(
            mean_sizes)[size_class_target]

        (dir_class_target, dir_res_target) = angle2class(gt_bboxes_3d[i].yaw)  # sunrgbd
        box_num = gt_labels_3d[i].shape[0]
        size_class_target = size_class_target.reshape(box_num, 1)
        dir_class_target = dir_class_target.reshape(box_num, 1)
        dir_res_target = dir_res_target.reshape(box_num, 1)
        rx = torch.empty(1, dtype=torch.float32).uniform_(-0.3, 0.3)
        ry = torch.empty(1, dtype=torch.float32).uniform_(-0.3, 0.3)
        rz = torch.empty(1, dtype=torch.float32).uniform_(-0.3, 0.3)
        center_aug = torch.cat((rx, ry, rz))
        gt_bboxes_3d[i].gravity_center = gt_bboxes_3d[i].gravity_center + center_aug

        tmp_info = torch.cat(
            (gt_bboxes_3d[i].gravity_center.cuda(), size_res_target.cuda(), size_class_target,
             dir_class_target.cuda(), dir_res_target.cuda()),
            1)
        bbox_info.append(tmp_info)
    # # bbox_info = DC(gt_bboxes_3d.tensor)
    # bbox_info_cat = torch.stack(bbox_info)
    label_one_hot = list()
    for i in range(len(gt_labels_3d)):
        # tmp_label_one_hot = to_one_hot(gt_labels_3d[i], 10)
        gt_labels_3d_tmp = gt_labels_3d[i].reshape(gt_labels_3d[i].shape[0], 1).cuda()
        tmp_label_one_hot = torch.zeros(gt_labels_3d_tmp.shape[0], 10).cuda().scatter_(1, gt_labels_3d_tmp, 1)
        tmp_label_one_hot = F.softmax(tmp_label_one_hot, dim=-1)

        label_one_hot.append(tmp_label_one_hot)
    # bbox_label = torch.stack(gt_labels_3d)

    gt_para = list()
    for i in range(len(gt_labels_3d)):
        gt_para.append(torch.zeros((256, 19)))
        tmp = torch.cat((bbox_info[i].cuda(), label_one_hot[i].cuda()), 1)
        for j in range(tmp.shape[0]):
            gt_para[i][j] = tmp[j]

    gt_para_cat = torch.stack(gt_para)
    return gt_para_cat
