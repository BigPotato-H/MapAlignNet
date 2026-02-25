import open3d
import torch.utils.data as data
import random
import os.path
import math
import torch
from PIL import Image
from torchvision import transforms

# matplotlib.use('TkAgg')

from data.argoverse import augmentation
import options
from data.argoverse.argo_helper import *
from matplotlib import pyplot as plt


from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.map import EgoViewMapRenderer
from av2.rendering.map import draw_visible_polyline_segments_cv2
import av2.geometry.interpolate as interp_utils
from pathlib import Path

from deeplab.utils import get_label_mask, set_class_values,LABEL_COLORS_LIST,ALL_CLASSES
from scipy.spatial.transform import Rotation as R

import albumentations as A
from av2.geometry.se3 import SE3
from av2.geometry.camera.pinhole_camera import PinholeCamera


def normalize():
    """
    Transform to normalize image.
    """
    transform = A.Compose([
        A.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True
        )
    ])
    return transform

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform


def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def make_argo_dataset(root_path,log_list):
    dataset = []
    for log_id in log_list:
        log_folder = os.path.join(root_path, log_id)
        map_folder = os.path.join(log_folder, 'map')
        img_folder = os.path.join(log_folder, 'sensors/cameras/ring_front_center/')
        img_list = os.listdir(img_folder)
        for img_na in img_list:
            dataset.append((log_id,map_folder,img_folder,img_na))

    return dataset


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)

    # T = P[0:3, 3].reshape(3, 1)
    # r = P[0:3, 0:3]
    # P_pc_homo_np = pc_np + T
    # P_pc_homo_np = np.dot(r, P_pc_homo_np)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


def bgr2gray(bgr):
    # t = np.array([0.299, 0.587, 0.114]).reshape(3,1)
    # gray = np.dot(bgr[..., :3], t)
    #
    # gray[gray[:, :, 0] >= 190] = 255
    # gray[gray[:, :, 0] < 190] = 0
    # col = gray[:, :, 0]
    # gray = np.concatenate([gray, gray,gray], axis=2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    retval, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def get_local_polyline(egoview_renderer, polyline):
    N_INTERP_PTS = 100
    polyline_city_frame = interp_utils.interp_arc(t=N_INTERP_PTS, points=polyline)
    polyline_ego_frame = egoview_renderer.ego_SE3_city.transform_point_cloud(polyline_city_frame)
    uv, points_cam, is_valid_points = egoview_renderer.pinhole_cam.project_ego_to_img(polyline_ego_frame,remove_nan=True)
    if is_valid_points.sum() == 0:
        return None, None

    u = np.round(uv[:, 0][is_valid_points]).astype(np.int32)  # type: ignore
    v = np.round(uv[:, 1][is_valid_points]).astype(np.int32)  # type: ignore

    lane_z = points_cam[:, 2][is_valid_points]
    line_segments_arr = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    return polyline_ego_frame, line_segments_arr


def cull_to_view_frustum(uv,H, W, points_cam):
    is_valid_x = np.logical_and(0 <= uv[0,:], uv[0,:] < W - 1)
    is_valid_y = np.logical_and(0 <= uv[1,:], uv[1,:] < H - 1)
    #is_valid_z = points_cam[:, 2] > 0
    is_valid_z = np.logical_and(0 < points_cam[2,:], points_cam[2,:] < 60)
    is_valid_points = np.logical_and(np.logical_and(is_valid_x, is_valid_y), is_valid_z) #bool
    return is_valid_points


def get_project_points(P,pc_np, img, K):
    pc_np_homo = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]))), axis=0)  # 4xN
    pc_np_recovered_homo = np.dot(P, pc_np_homo)
    KP_hd_pxpy = np.dot(K, pc_np_recovered_homo)  # 3xN
    KP_hd_pxpy = KP_hd_pxpy / KP_hd_pxpy[2:, :]# 3xN
    KP_hd_pxpy[2,:] = pc_np_recovered_homo[2,:]

    H, W = img.shape[0], img.shape[1]
    is_valid_points = cull_to_view_frustum(KP_hd_pxpy, H, W, pc_np_recovered_homo)

    mask = np.zeros_like(KP_hd_pxpy)
    KP_hd_pxpy = np.where(is_valid_points, KP_hd_pxpy, mask)
    return KP_hd_pxpy


def save_aug_hd_img(P, pc_np, img, K, save_dir):
    KP_hd_pxpy = get_project_points(P,pc_np, img, K)

    view_img = img.copy()
    y_indices = KP_hd_pxpy[1, :].astype(int)
    x_indices = KP_hd_pxpy[0, :].astype(int)
    view_img[y_indices, x_indices,0] = 0
    view_img[y_indices, x_indices,1] = 0
    view_img[y_indices, x_indices,2] = 255
    #view_img = view_img.transpose(1, 2, 0)

    cv2.imwrite(save_dir, view_img)


class argoLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(argoLoader, self).__init__()
        #self.root = root + str(mode)
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # store the calibration matrix for each sequence
        self.log_list = []
        if len(opt.log_id) == 0:
            self.log_list = os.listdir(self.root)
            # log_list = log_list[0:1]
            ''' if mode == 'train':
                #log_list = log_list[0:3]
                log_list =['00a6ffc1-6ce9-3bc3-a060-6006e9893a1a',
                           '0a8a4cfa-4902-3a76-8301-08698d6290a2',
                           '0a524e66-ee33-3b6c-89ef-eac1985316db']
            if mode == 'val':
                log_list = ['0a132537-3aec-35bb-af13-7faa0811000d']'''
            # log_list = log_list[-1:]
        else:
            self.log_list = [opt.log_id]
        self.calib_helper = argoCalibHelper(Path(self.root),self.log_list)
        # print(self.calib_helper.calib_matrix_dict)

        # list of (pc_path, img_path, seq, i, img_key)
        self.dataset = make_argo_dataset(self.root, self.log_list)

        self.label_colors_list = LABEL_COLORS_LIST
        self.all_classes = ALL_CLASSES
        self.classes_to_train = ALL_CLASSES
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )
        if mode == 'train':
            self.tfms = train_transforms(256)
        else:
            self.tfms = valid_transforms(256)

        self.norm_tfms = normalize()
        self.save_label_img = False
        self.file = self.opt.decalib_file
        if self.file is not None:
            self.perturb_arr = np.loadtxt(self.file, dtype=np.float32, delimiter=',')

    def augment_pc(self, pc_np, intensity_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np

    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def cal_rotate_matrix(self, angles):
        axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        r_x = angles[0]
        r_y = angles[1]
        r_z = angles[2]
        rot_x = linalg.expm(np.cross(np.eye(3), axis_x / linalg.norm(axis_x) * r_x))
        rot_y = linalg.expm(np.cross(np.eye(3), axis_y / linalg.norm(axis_y) * r_y))
        rot_z = linalg.expm(np.cross(np.eye(3), axis_z / linalg.norm(axis_z) * r_z))

        rot = np.dot(np.dot(rot_x, rot_y), rot_z)
        return rot

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        # angles = [random.uniform(math.pi / 2 - P_Rx_amplitude, math.pi / 2 + P_Rx_amplitude),
        #           random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
        #           random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        #rotation_mat = augmentation.angles2rotation_matrix(angles)
        rotation_mat = self.cal_rotate_matrix(angles)
        P_random = np.identity(4, dtype=float)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def downsample_np(self, pc_np, intensity_np):
        if pc_np.shape[1] >= self.opt.input_pt_num:
            choice_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.opt.input_pt_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]

        return pc_np, intensity_np

    def get_local_map(self, map_folder,img, img_na, city_SE3_ego,pinhole_cam):

        avm = ArgoverseStaticMap.from_map_dir(Path(map_folder), build_raster=False)
        lane_segments = avm.get_scenario_lane_segments()
        ped_crossings = avm.get_scenario_ped_crossings()
        #ped_crossings = []
        egoview_renderer = EgoViewMapRenderer(
            depth_map=None, city_SE3_ego=city_SE3_ego, pinhole_cam=pinhole_cam, avm=avm
        )

        map_3d_local = []
        map_2d = []
        if self.save_label_img:
            img_bgr = img.copy()
        else:
            #img_bgr = np.zeros((img.shape[0], img.shape[1],1), dtype=np.uint8)
            img_bgr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for lane_segment in lane_segments:
            if lane_segment.is_intersection:
                continue
            polyline_ego_frame, line_segments_arr = \
                get_local_polyline(egoview_renderer,lane_segment.right_lane_boundary.xyz)
            if line_segments_arr is None:
                continue

            map_3d_local.append(polyline_ego_frame)
            map_2d.append(line_segments_arr)
            not_occluded = np.ones(line_segments_arr.shape[0], dtype=bool)
            draw_visible_polyline_segments_cv2(line_segments_arr,
                valid_pts_bool=not_occluded,image=img_bgr,
               color=(255, 255, 255),
               #color=255,
                thickness_px=10)

            get_left = True
            if get_left:
                polyline_ego_frame, line_segments_arr = \
                    get_local_polyline(egoview_renderer, lane_segment.left_lane_boundary.xyz)
                if line_segments_arr is None:
                    continue

                map_3d_local.append(polyline_ego_frame)
                map_2d.append(line_segments_arr)
                not_occluded = np.ones(line_segments_arr.shape[0], dtype=bool)
                draw_visible_polyline_segments_cv2(line_segments_arr,
                                                   valid_pts_bool=not_occluded,
                                                   image=img_bgr,
                                                   color=(255, 255, 255),
                                                   # color=255,
                                                   thickness_px=10)
        for pc in ped_crossings:
            polyline_ego_frame, line_segments_arr = \
                get_local_polyline(egoview_renderer,pc.polygon)
            if line_segments_arr is None:
                continue

            map_3d_local.append(polyline_ego_frame)
            map_2d.append(line_segments_arr)
            not_occluded = np.ones(line_segments_arr.shape[0], dtype=bool)
            draw_visible_polyline_segments_cv2(line_segments_arr,
                valid_pts_bool=not_occluded,image=img_bgr,
               color=(255, 255, 255),
               #color=255,
                thickness_px=10)

        #pc_np = npy_data[0:3, :]  # 3xN
        #intensity_np = npy_data[3:4, :]  # 1xN
        #sn_np = npy_data[4:7, :]  # 3xN
        map_3d_local = np.concatenate(map_3d_local).astype(np.float32)
        map_2d = np.concatenate(map_2d)
        type_np = np.full(map_3d_local.shape[0], 13)
        type_np = np.expand_dims(type_np, axis=0)

        if self.save_label_img:
            cv2.imwrite('/mnt/e/data/hn/temp/hd_rgb_label/'+img_na+'.jpg', img_bgr)#draw map on orginal image
            cv2.imwrite('/mnt/e/data/hn/temp/hd_label_binary/' + img_na + '.jpg', img_bgr)#draw mask image
        return map_3d_local.T, type_np, img_bgr


    def get_label_image(self, index, rgb_img, extrinsic, clr=[255,255,255]):
        log_id, map_folder, img_folder, img_na = self.dataset[index]
        pinhole_cam = self.calib_helper.get_Pi_matrix(log_id)
        img_stamp = int(Path(img_na).stem)
        city_SE3_ego = self.calib_helper.get_matrix(log_id, img_stamp)

        #3*4->4*4
        #extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        #init_extrinsic = pinhole_cam.extrinsics
        extrinsic_inv = np.linalg.inv(extrinsic)
        se3 = SE3(rotation=extrinsic_inv[0:3, 0:3], translation=extrinsic_inv[0:3, 3])
        #se3 = SE3(rotation=np.linalg.inv(init_extrinsic[:, 0:3]), translation=-init_extrinsic[0:3, 3])
        phc = PinholeCamera(
            ego_SE3_cam=se3,
            intrinsics=pinhole_cam.intrinsics,
            cam_name=pinhole_cam.cam_name,
        )
        #local_hd_map, class_np, depth_img = get_local_map(map_folder, rgb_img, img_na, city_SE3_ego, pinhole_cam)
        local_hd_map, class_np, depth_img = self.get_local_map(map_folder, rgb_img, img_na, city_SE3_ego, phc)

        img_depth_colored = cv2.merge([depth_img, depth_img, depth_img])
        mask = img_depth_colored > 0
        overlap_img = rgb_img.copy()
        #overlap_img[mask] = img_depth_colored[mask]
        overlap_img[mask[:,:,0]] = np.array(clr, dtype=np.uint8)
        if 0:
            cv2.imwrite(r'E:\data\hn\temp\regnet_results\init_depth\\' + img_na, overlap_img)

        return depth_img, overlap_img


    def get_unaligned_project_points(self, pc_np, img, K):
        H, W = img.shape[0], img.shape[1]
        pc_np_homo = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]))), axis=0)
        #pc_np_homo = np.dot(P, pc_np_homo)[0:3, :]
        pc_np_homo_K = np.dot(K, pc_np_homo)
        KP_pc_pxpy = pc_np_homo_K[0:2, :] / pc_np_homo_K[2:3, :]
        KP_pc_pxpy = KP_pc_pxpy.astype(np.int)

        px = KP_pc_pxpy[0, :]
        py = KP_pc_pxpy[1, :]

        x_inside_mask = np.array((px >= 0) & (px <= W - 1), dtype=bool)
        y_inside_mask = np.array((py >= 0) & (py <= H - 1), dtype=bool)
        z_inside_mask = np.array(pc_np_homo[2] > 0.1, dtype=bool)
        inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask)

        is_valid_points = cull_to_view_frustum(KP_pc_pxpy, H, W, pc_np_homo)
        is_valid_points = is_valid_points.bool().unsqueeze(1)
        is_valid_points = is_valid_points.repeat(1, 2, 1)

        mask = torch.zeros_like(KP_pc_pxpy)
        KP_hd_pxpy = torch.where(is_valid_points, KP_pc_pxpy, mask)

        #intensity
        idx = py * W + px
        inside_idx = idx[inside_mask]
        gray_img = img[:, :, 0].flatten()
        pix = gray_img[inside_idx]
        #inside_mask[inside_mask] = (pix > 0)
        return inside_mask

    def get_label_mask(self,img_label):
        return get_label_mask(img_label, self.class_values, self.label_colors_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        log_id, map_folder, img_folder,img_na= self.dataset[index]

        pinhole_cam = self.calib_helper.get_Pi_matrix(log_id)
        K = pinhole_cam.intrinsics.K
        img_stamp = int(Path(img_na).stem)
        city_SE3_ego = self.calib_helper.get_matrix(log_id,img_stamp)

        ############img#############
        img_path = os.path.join(img_folder, img_na)
        img = cv2.imread(img_path)
        # load point cloud
        pc_np, intensity_np,img_label = self.get_local_map(map_folder,img, img_na, city_SE3_ego, pinhole_cam)
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np)

        #  ------------- apply random transform on points under the NWU coordinate ------------
        Pr = []
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude,
                                                self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude,
                                                self.opt.P_Rz_amplitude)
            # -------------- augmentation ----------------------
            pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            img = self.augment_img(img)
        else:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude,
                                                self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude,
                                                self.opt.P_Rz_amplitude)
        if self.file is not None:
            rt = self.perturb_arr[index]
            rot = R.from_euler('xyz', rt[3:6], degrees=True)
            decalib_rot = rot.as_matrix()

            P_random = np.identity(4, dtype=float)
            P_random[0:3, 0:3] = decalib_rot
            P_random[0:3, 3] = rt[0:3]
            Pr = P_random

        Pr_inv = np.linalg.inv(Pr)

        # perturb 4*4
        P_gt = pinhole_cam.extrinsics
        P = np.dot(Pr, P_gt)

        ###for view
        unaligned_label_image, _ = self.get_label_image(index, img, P)
        #mask = unaligned_label_image > 0
        #img[mask[:, :, 0]] = np.array([255,0,0], dtype=np.uint8)
        #temp_save_dir = '/mnt/e/data/hn/temp/aug_hd_img_unalign/' + img_na
        #cv2.imwrite(temp_save_dir, img)

        if 0:
            _, gt_overlap_image= self.get_label_image(index, img, P_gt, [0, 255, 0])
            temp_save_dir = '/mnt/e/data/hn/temp/aug_hd_img/' + img_na
            cv2.imwrite(temp_save_dir, gt_overlap_image)

            _, unaligned_overlap_image = self.get_label_image(index, img, P, [0, 0, 255])
            temp_save_dir = '/mnt/e/data/hn/temp/aug_hd_img_unalign/' + img_na
            cv2.imwrite(temp_save_dir, unaligned_overlap_image)

        # crop the first few rows, original is 370x1226 now
        K = camera_matrix_cropping(K, dx=0, dy=self.opt.crop_original_top_rows)
        img = img[self.opt.crop_original_top_rows:, :, :]
        img_label = img_label[self.opt.crop_original_top_rows:, :]
        unaligned_label_image = unaligned_label_image[self.opt.crop_original_top_rows:, :]
        # scale
        H = int(round(img.shape[1] * self.opt.img_scale))
        W = int(round(img.shape[0] * self.opt.img_scale))
        img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
        img_label = cv2.resize(img_label, (H, W), interpolation=cv2.INTER_LINEAR)
        unaligned_label_image = cv2.resize(unaligned_label_image, (H, W), interpolation=cv2.INTER_LINEAR)
        K = camera_matrix_scaling(K, self.opt.img_scale)

        # crop into input size
        img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
        img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        img_label = img_label[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W]
        unaligned_label_image = unaligned_label_image[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        # 3*4
        P = P[0:3, :]
        img_bgr = img.copy()
        #img = bgr2gray(img)

        # #################debug visualization of random transformation & augmentation
        if 0:
            temp_save_dir = '/mnt/e/data/hn/temp/aug_hd_img/pt_'+img_na
            save_aug_hd_img(P_gt[0:3, :], pc_np, img_bgr, K, temp_save_dir)

            temp_save_dir = '/mnt/e/data/hn/temp/aug_hd_img_unalign/pt_'+img_na
            save_aug_hd_img(P, pc_np, img_bgr, K, temp_save_dir)

        #unalign_project_points
        upp = get_project_points(P, pc_np, img_bgr, K)

        img = self.norm_tfms(image=img)['image']
        img_label = self.get_label_mask(img_label)
        unaligned_label_image = self.get_label_mask(unaligned_label_image)
        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np.astype(np.float32))  # 3xN

        P_init = torch.from_numpy(P.astype(np.float32))  # 3x4
        P_gt = torch.from_numpy(P_gt[0:3,:].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        #img_label = torch.from_numpy(img_label.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW

        img_label = torch.from_numpy(img_label.astype(np.float32)).contiguous()  # HxW
        K = torch.from_numpy(K.astype(np.float32))  # 3x3c
        upp = torch.from_numpy(upp.astype(np.float32))
        unaligned_label_image = torch.from_numpy(unaligned_label_image.astype(np.float32)).contiguous()
        unaligned_label_image = unaligned_label_image.unsqueeze(0)
        return pc, img, img_label,\
               P_gt, P_init, K,upp,index,img_path,unaligned_label_image


if __name__ == '__main__':
    opt = options.Options()
    argoloader = argoLoader(opt.input_dir, 'train', opt)
    #argoloader = argoLoader(opt.dataroot, 'val', opt)
    argoloader.save_label_img = True

    for i in range(0, len(argoloader), 1):
        print('--- %d ---' % i)
        data = argoloader[i]
        for item in data:
            print(item.size())
