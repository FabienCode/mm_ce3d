import mmcv
import numpy as np
from concurrent import futures as futures
from os import path as osp
from scipy import io as sio


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random sampling.

    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.
        num_points (int): The number of samples.
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1 * heading_angle)
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


class SUNRGBDInstance(object):

    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1]
        self.ymin = data[2]
        self.xmax = data[1] + data[3]
        self.ymax = data[2] + data[4]
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.centroid = np.array([data[5], data[6], data[7]])
        self.w = data[8]
        self.l = data[9]  # noqa: E741
        self.h = data[10]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1],
                                             self.orientation[0])
        self.box3d = np.concatenate([
            self.centroid,
            np.array([self.l * 2, self.w * 2, self.h * 2, self.heading_angle])
        ])


class SUNRGBDData(object):
    """SUNRGBD data.

    Generate scannet infos for sunrgbd_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
        use_v1 (bool): Whether to use v1. Default: False.
    """

    def __init__(self, root_path, split='train', use_v1=False):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path, 'sunrgbd_trainval')
        self.classes = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
            'night_stand', 'bookshelf', 'bathtub'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {
            label: self.classes[label]
            for label in range(len(self.classes))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.split_dir, f'{split}_data_idx.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = map(int, mmcv.list_from_file(split_file))
        self.image_dir = osp.join(self.split_dir, 'image')
        self.calib_dir = osp.join(self.split_dir, 'calib')
        self.depth_dir = osp.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = osp.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = osp.join(self.split_dir, 'label')

    def __len__(self):
        return len(self.sample_id_list)

    def get_image(self, idx):
        img_filename = osp.join(self.image_dir, f'{idx:06d}.jpg')
        return mmcv.imread(img_filename)

    def get_image_shape(self, idx):
        image = self.get_image(idx)
        return np.array(image.shape[:2], dtype=np.int32)

    def get_depth(self, idx):
        depth_filename = osp.join(self.depth_dir, f'{idx:06d}.mat')
        depth = sio.loadmat(depth_filename)['instance']
        return depth

    def get_calibration(self, idx):
        calib_filepath = osp.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rt = np.array([float(x) for x in lines[0].split(' ')])
        Rt = np.reshape(Rt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        return K, Rt

    def get_label_objects(self, idx):
        label_filename = osp.join(self.label_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [SUNRGBDInstance(line) for line in lines]
        return objects

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            # convert depth to points
            SAMPLE_NUM = 50000
            GT_SAMPLE_NUM = 20000
            # TODO: Check whether can move the point
            #  sampling process during training.
            pc_upright_depth = self.get_depth(sample_idx)
            pc_upright_depth_subsampled = random_sampling(
                pc_upright_depth, SAMPLE_NUM)

            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'gt_points'))
            # print('heiheiheihei {}'.format(pc_upright_depth_subsampled.shape))
            pc_upright_depth_subsampled.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx:06d}.bin'))


            info['pts_path'] = osp.join('points', f'{sample_idx:06d}.bin')
            info['gt_pts_path'] = osp.join('gt_points', f'gt_{sample_idx:06d}.npy')
            img_name = osp.join(self.image_dir, f'{sample_idx:06d}')
            img_path = osp.join(self.image_dir, img_name)
            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx),
                'image_path': img_path
            }
            info['image'] = image_info

            K, Rt = self.get_calibration(sample_idx)
            calib_info = {'K': K, 'Rt': Rt}
            info['calib'] = calib_info

            # --------------Gt_pts
            obj_list = self.get_label_objects(sample_idx)
            point_inner_bbox = np.zeros((SAMPLE_NUM, 6))
            # print(f"ori pts_inner_bbox shape is {point_inner_bbox.shape[0]}!!!")
            count_pts_inner_bbox = 0
            box_count = 0
            reshape_pc_upright_depth_subsampled = pc_upright_depth_subsampled.reshape(-1, 6)
            # if box_count != 0:
            for obj in obj_list:
                if obj.classname not in self.cat2label.keys(): continue
                if obj.classname in self.cat2label.keys():
                    begin_count = count_pts_inner_bbox
                    box_count += 1
                    try:
                        box3d_pts_3d = my_compute_box_3d(obj.centroid,
                                                         np.array([obj.l, obj.w, obj.h]), obj.heading_angle)
                        pc_in_box3d, inds = extract_pc_in_box3d( \
                            reshape_pc_upright_depth_subsampled[:, 0:3], box3d_pts_3d)
                        # print('box3d_pts_3d is {}, obj.box3d is {}'.format(box3d_pts_3d, obj.box3d))
                        for a in range(pc_in_box3d.shape[0]):
                            point_inner_bbox[count_pts_inner_bbox, 0:3] = pc_in_box3d[a, 0:3]
                            count_pts_inner_bbox += 1
                    except:
                        print(f'{self.split} sample_idx: {sample_idx} {obj.classname} error!!!')
                    print(
                        f'{self.split} sample_idx: {sample_idx} {obj.classname} have {count_pts_inner_bbox - begin_count} gt pts!')
            if box_count == 0:
                point_inner_bbox = random_sampling(pc_upright_depth_subsampled, GT_SAMPLE_NUM)
                # for i in range(point_inner_bbox_sampler.shape[0]):
                #     point_inner_bbox[count_pts_inner_bbox, :] = point_inner_bbox_sampler[i, :]
                #     count_pts_inner_bbox += 1
            elif count_pts_inner_bbox > GT_SAMPLE_NUM:
                point_inner_bbox = random_sampling(point_inner_bbox[0:count_pts_inner_bbox, :], GT_SAMPLE_NUM)
            else:
                point_inner_bbox = point_inner_bbox[0:GT_SAMPLE_NUM, :]


            # if box_count == 0:
            #     point_inner_bbox = random_sampling(reshape_pc_upright_depth_subsampled[:, 0:3], GT_SAMPLE_NUM)
            # elif count_pts_inner_bbox > GT_SAMPLE_NUM:
            #     point_inner_bbox = random_sampling(point_inner_bbox[0:count_pts_inner_bbox, :], GT_SAMPLE_NUM)
            # else:
            #     point_inner_bbox = point_inner_bbox[0:GT_SAMPLE_NUM, :]

            # print('{} sample_idx:{} Total have {} gt pts ! pts_inner_bbox shape is {}'.format(self.split, sample_idx,
            #                                                                                   count_pts_inner_bbox,
            #                                                                                   point_inner_bbox.shape[
            #                                                                                       0]))
            # print(' {} gt_pts shape: {}'.format(sample_idx, point_inner_bbox.shape))
            np.save(osp.join(self.root_dir, 'gt_points', f'gt_{sample_idx:06d}.npy'),
                    point_inner_bbox)
            gt_points = np.load(osp.join(self.root_dir, 'gt_points', f'gt_{sample_idx:06d}.npy'))

            print('Fffffffffff$$$$$$$$###@@@@@ gt_point_file shape is {}'.format(gt_points.shape))

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['bbox'] = np.concatenate([
                        obj.box2d.reshape(1, 4) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                        axis=0)
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                        axis=0)
                    annotations['dimensions'] = 2 * np.array([
                        [obj.l, obj.h, obj.w] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lhw(depth) format
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['index'] = np.arange(
                        len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.keys()
                        ],
                        axis=0)  # (K,8)
                info['annos'] = annotations
            return info


        sample_id_list = sample_id_list if \
            sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
