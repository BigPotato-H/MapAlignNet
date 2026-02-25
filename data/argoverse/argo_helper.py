import numpy as np
import os
import cv2
from scipy import linalg
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

def coordinate_cam_to_NWU(pc_np):
    assert pc_np.shape[0] == 3
    pc_nwu_np = np.copy(pc_np)
    pc_nwu_np[0, :] = pc_np[2, :]  # x <- z
    pc_nwu_np[1, :] = -pc_np[0, :]  # y <- -x
    pc_nwu_np[2, :] = -pc_np[1, :]  # z <- -y
    return pc_nwu_np


def coordinate_NWU_to_cam(pc_np):
    assert pc_np.shape[0] == 3
    pc_cam_np = np.copy(pc_np)
    pc_cam_np[0, :] = -pc_np[1, :]  # x <- -y
    pc_cam_np[1, :] = -pc_np[2, :]  # y <- -z
    pc_cam_np[2, :] = pc_np[0, :]  # z <- x
    return pc_cam_np




def read_calib(data_root, log_list):
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    img_matrix_dict = dict()
    pinhole_cam_dict = dict()
    for log_id in log_list:

        cam_name = 'ring_front_center'

        pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)

        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        se3_dict = dict()
        imgs = []
        for i, img_fpath in enumerate(cam_im_fpaths):
            cam_timestamp_ns = int(img_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                print("missing city_SE3_ego")
                continue
            imgs.append(cam_timestamp_ns)
            se3_dict[cam_timestamp_ns] = city_SE3_ego
        img_matrix_dict[log_id] = se3_dict
        pinhole_cam_dict[log_id] = pinhole_cam
    return imgs, img_matrix_dict,pinhole_cam_dict



class argoCalibHelper:
    def __init__(self, root_path, log_list):
        self.root_path = root_path
        self.imgs, self.calib_matrix_dict,self.Pi_matrix_dict = read_calib(root_path, log_list)


    def get_img_dict(self):
        return self.imgs

    def get_matrix(self,log_id: str, matrix_key1: int):
        return self.calib_matrix_dict[log_id][matrix_key1]


    def get_Pi_matrix(self, log_id):
        return self.Pi_matrix_dict[log_id]

    def transform_pc_vel_to_img(self,
                                pc: np.ndarray,
                                Tr: np.ndarray=None):
        """

        :param pc: 3xN
        :param seq: int
        :param img_key: 'P0', 'P1', 'P2', 'P3'
        :return: 3xN
        """
        pc_homo = pc  # 3xN
        T = Tr[0:3, 3].reshape(3,1)
        r = Tr[0:3, 0:3]
        pc_homo = pc_homo + T
        pc_img_homo = np.dot(r, pc_homo)
        return pc_img_homo[0:3, :]

    def transform_pc_img_to_vel(self,
                                pc: np.ndarray,
                                seq: int = 0,
                                img_key: str = 'P2',
                                Pi: np.ndarray=None,
                                Tr: np.ndarray=None):
        """

        :param pc: 3xN
        :param seq: int
        :param img_key: 'P0', 'P1', 'P2', 'P3'
        :return: 3xN
        """
        pc_homo = np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)  # 3xN
        if Pi is None:
            Pi_inv = np.linalg.inv(self.get_matrix(seq, img_key))
        else:
            Pi_inv = np.linalg.inv(Pi)
        if Tr is None:
            Tr_inv = np.linalg.inv(self.get_matrix(seq, 'Tr'))
        else:
            Tr_inv = np.linalg.inv(Tr)
        pc_vel_homo = np.dot(np.dot(Tr_inv, Pi_inv), pc_homo)  # 4x4 * 4x4 * 4xN
        return pc_vel_homo[0:3, :]


def crop_pc_with_img(pc_np, intensity_np, sn_np, img, K):
    """

    :param pc_np:
    :param intensity_np:
    :param sn_np:
    :param img:
    :param K:
    :return:
    """
    H, W = img.shape[0], img.shape[1]

    pc_pixels = np.dot(K, pc_np)  # 3xN
    pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN

    pc_pixels = np.round(pc_pixels)
    pc_mask_x = np.logical_and(pc_pixels[0, :] >= 0, pc_pixels[0, :] <= W - 1)
    pc_mask_y = np.logical_and(pc_pixels[1, :] >= 0, pc_pixels[1, :] <= H - 1)
    pc_mask = np.logical_and(pc_mask_x, pc_mask_y)

    pc_np_img = pc_np[:, pc_mask]
    intensity_np_img = intensity_np[:, pc_mask]
    sn_np_img = sn_np[:, pc_mask]

    return pc_np_img, intensity_np_img, sn_np_img


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


class ProjectiveFarthestSampler:
    def __init__(self):
        self.fps_2d = FarthestSampler(dim=2)

    def sample(self, pts, k, projection_K):
        # 1. project the points onto image with projection K
        pts_2d = np.dot(projection_K, pts)  # 3x3 * 3xN -> 3xN
        pts_2d = pts_2d[0:2, :] / pts_2d[2:, :]  # 2xN

        # 2. FPS on 2d
        nodes_2d, nodes_idx = self.fps_2d.sample(pts_2d, k)

        # 3. get the corresponding 3d points
        nodes_3d = pts[:, nodes_idx]

        return nodes_3d, nodes_idx


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx
