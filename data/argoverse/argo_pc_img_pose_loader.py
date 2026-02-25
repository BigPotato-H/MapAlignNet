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
from data.argoverse import options
from data.argoverse.argo_helper import *
from matplotlib import pyplot as plt


from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.map import EgoViewMapRenderer
from av2.rendering.map import draw_visible_polyline_segments_cv2
import av2.geometry.interpolate as interp_utils
from pathlib import Path

def downsample_with_intensity_sn(pointcloud, intensity, sn, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    intensity_max = np.max(intensity)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0:1] = np.transpose(intensity) / intensity_max

    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    pcd.normals = open3d.utility.Vector3dVector(np.transpose(sn))

    down_pcd = open3d.geometry.voxel_down_sample(pcd, voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    intensity = np.transpose(np.asarray(down_pcd.colors)[:, 0:1]) * intensity_max
    sn = np.transpose(np.asarray(down_pcd.normals))

    return pointcloud, intensity, sn


def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def make_argo_dataset(root_path, mode, opt):
    dataset = []
    log_list = os.listdir(root_path)

    ''' if mode == 'train':
        #log_list = log_list[0:3]
        log_list =['00a6ffc1-6ce9-3bc3-a060-6006e9893a1a',
                   '0a8a4cfa-4902-3a76-8301-08698d6290a2',
                   '0a524e66-ee33-3b6c-89ef-eac1985316db']
    if mode == 'val':
        log_list = ['0a132537-3aec-35bb-af13-7faa0811000d']'''
        #log_list = log_list[-1:]

    for log_id in log_list:
        log_folder = os.path.join(root_path, log_id)
        map_folder = os.path.join(log_folder, 'map')
        img_folder = os.path.join(log_folder, 'sensors\\cameras\\ring_front_center\\')
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
    N_INTERP_PTS = 1000
    polyline_city_frame = interp_utils.interp_arc(t=N_INTERP_PTS, points=polyline)
    polyline_ego_frame = egoview_renderer.ego_SE3_city.transform_point_cloud(polyline_city_frame)
    uv, points_cam, is_valid_points = egoview_renderer.pinhole_cam.project_ego_to_img(polyline_ego_frame)
    if is_valid_points.sum() == 0:
        return None, None

    u = np.round(uv[:, 0][is_valid_points]).astype(np.int32)  # type: ignore
    v = np.round(uv[:, 1][is_valid_points]).astype(np.int32)  # type: ignore

    lane_z = points_cam[:, 2][is_valid_points]
    line_segments_arr = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    return polyline_ego_frame, line_segments_arr


def save_aug_data(P,pc_np,intensity_np,img,K,inside_mask,img_na):
    pc_np_homo = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]))), axis=0)  # 4xN
    pc_np_recovered_homo = np.dot(P, pc_np_homo)
    ####pc_np_recovered_homo = pc_np_homo
    pc_np_recovered_vis = projection_pc_img(pc_np_recovered_homo[0:3, :],
                                            intensity_np, img, K, inside_mask, size=1)
    cv2.imwrite(r'E:\data\hn\temp\aug_hd_img\\' + img_na + '.jpg', pc_np_recovered_vis)


class argoLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(argoLoader, self).__init__()
        self.root = root + str(mode)
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # store the calibration matrix for each sequence
        self.calib_helper = argoCalibHelper(Path(self.root))
        # print(self.calib_helper.calib_matrix_dict)

        # list of (pc_path, img_path, seq, i, img_key)
        self.dataset = make_argo_dataset(self.root, mode, opt)

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
        P_random = np.identity(4, dtype=np.float)
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
        #ped_crossings = avm.get_scenario_ped_crossings()

        egoview_renderer = EgoViewMapRenderer(
            depth_map=None, city_SE3_ego=city_SE3_ego, pinhole_cam=pinhole_cam, avm=avm
        )

        left_neigh_dict = {}
        for lane_seg in lane_segments:
            if lane_seg.left_neighbor_id is not None and \
                    lane_seg.id < lane_seg.left_neighbor_id:  # (用于区分位置重叠方向相反的车道线，只存一次)
                left_neigh_dict[lane_seg.id] = lane_seg.left_neighbor_id


        map_3d_local = []
        map_2d = []
        #img_bgr = img.copy()
        img_bgr = np.zeros((img.shape[0], img.shape[1],1), dtype=np.uint8)
        #img_bgr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for lane_segment in lane_segments:
            polyline_ego_frame, line_segments_arr = \
                get_local_polyline(egoview_renderer,lane_segment.right_lane_boundary.xyz)
            if line_segments_arr is None:
                continue

            map_3d_local.append(polyline_ego_frame)
            map_2d.append(line_segments_arr)
            not_occluded = np.ones(line_segments_arr.shape[0], dtype=bool)
            draw_visible_polyline_segments_cv2(line_segments_arr,
                valid_pts_bool=not_occluded,image=img_bgr,
               #color=(255, 255, 255),
               color=255,
                thickness_px=3)

            if lane_seg.left_neighbor_id is not None:
                left_left_lane_id = left_neigh_dict.get(lane_seg.left_neighbor_id)
                if lane_seg.id == left_left_lane_id:
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
                                                       #color=(255, 255, 255),
                                                       color=255,
                                                       thickness_px=3)

        #pc_np = npy_data[0:3, :]  # 3xN
        #intensity_np = npy_data[3:4, :]  # 1xN
        #sn_np = npy_data[4:7, :]  # 3xN
        map_3d_local = np.concatenate(map_3d_local).astype(np.float32)
        map_2d = np.concatenate(map_2d)
        type_np = np.full(map_3d_local.shape[0], 13)
        type_np = np.expand_dims(type_np, axis=0)

        #cv2.imwrite(r'E:\data\hn\temp\hd_img\\'+img_na+'.jpg', img_bgr)
        #cv2.imwrite(r'E:\data\hn\temp\label_img\\' + img_na + '.jpg', img_bgr)

        #img_bgr = cv2.normalize(img_bgr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return map_3d_local.T, type_np, img_bgr

    def cal_per_point_label(self, pc_np, img, P, K):
        H, W = img.shape[0], img.shape[1]
        pc_np_homo = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]))), axis=0)
        pc_np_homo = np.dot(P, pc_np_homo)[0:3, :]
        pc_np_homo_K = np.dot(K, pc_np_homo)
        KP_pc_pxpy = pc_np_homo_K[0:2, :] / pc_np_homo_K[2:3, :]
        KP_pc_pxpy = KP_pc_pxpy.astype(np.int)

        px = KP_pc_pxpy[0, :]
        py = KP_pc_pxpy[1, :]

        x_inside_mask = np.array((px >= 0) & (px <= W - 1), dtype=bool)
        y_inside_mask = np.array((py >= 0) & (py <= H - 1), dtype=bool)
        z_inside_mask = np.array(pc_np_homo[2] > 0.1, dtype=bool)
        inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask)

        #intensity
        idx = py * W + px
        inside_idx = idx[inside_mask]
        gray_img = img[:, :, 0].flatten()
        pix = gray_img[inside_idx]
        #inside_mask[inside_mask] = (pix > 0)

        #boundary
        binary_img = img[:, :, 0]
        contours,_ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        zero_img = np.zeros(binary_img.shape)
        cv2.drawContours(zero_img, contours, -1, 255, -1)
        cv2.drawContours(zero_img, contours, -1, 255, 5)
        bound_pix = zero_img.flatten()[inside_idx]
        inside_mask[inside_mask] = np.logical_or(pix > 0, bound_pix > 0)
        # bound_idx = np.zeros(img.shape[0] * img.shape[1], dtype=bool)
        # for contour in contours:
        #     c_idx = contour[:, :, 1] * W + contour[:, :, 0]
        #     bound_idx[c_idx] = 1

        #cv2.imwrite('/data/nianh/DeepI2P/cc.jpg', zero_img)
        return inside_mask


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        log_id, map_folder, img_folder,img_na= self.dataset[index]
        #pc_folder, img_folder, pc_na, img_na = self.dataset[index]

        pinhole_cam = self.calib_helper.get_Pi_matrix(log_id)
        K = pinhole_cam.intrinsics.K
        img_stamp = int(Path(img_na).stem)
        city_SE3_ego = self.calib_helper.get_matrix(log_id,img_stamp)

        ############img#############
        img_path = os.path.join(img_folder, img_na)
        img = cv2.imread(img_path)
        # load point cloud
        pc_np, intensity_np,img_label = self.get_local_map(map_folder,img, img_na, city_SE3_ego, pinhole_cam)


        K = camera_matrix_cropping(K, dx=0, dy=self.opt.crop_original_top_rows)

        # crop the first few rows, original is 370x1226 now
        img = img[self.opt.crop_original_top_rows:, :, :]
        img_label = img_label[self.opt.crop_original_top_rows:, :]
        # scale
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.opt.img_scale)),
                          int(round((img.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        img_label = cv2.resize(img_label,
                         (int(round(img_label.shape[1] * self.opt.img_scale)),
                          int(round((img_label.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        K = camera_matrix_scaling(K, self.opt.img_scale)

        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        img_label = img_label[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W]
        #img_label = np.expand_dims(img_label, axis=2)
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

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
            # if random.random() > 0.5:
            #     img = np.flip(img, 1)
            #     P_flip = np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=pc_np.dtype)
            #     Pr = np.dot(Pr, P_flip)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)

        else:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude,
                                                self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude,
                                                self.opt.P_Rz_amplitude)

        Pr_inv = np.linalg.inv(Pr)
        pc_np = transform_pc_np(Pr, pc_np)
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np)

        # assemble P. P * pc will get the point cloud in the camera image coordinate
        P = np.dot(pinhole_cam.extrinsics, Pr_inv)
        #P = Pc
        t_ji = np.array([0,0,0])

        img_bgr = img.copy()
        #img = bgr2gray(img)
        inside_mask = self.cal_per_point_label(pc_np, img, P, K)
        # #################debug visualization of random transformation & augmentation
        #save_aug_data(P,pc_np,intensity_np,img_bgr,K,inside_mask,img_na)

        # ------------ Farthest Point Sampling ------------------
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              self.opt.node_a_num * 8,
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            self.opt.node_b_num * 8,
                                                                            replace=False)],
                                                 k=self.opt.node_b_num)

        # visualize nodes
        # ax = vis_tools.plot_pc(pc_np, size=1)
        # ax = vis_tools.plot_pc(node_a_np, size=10, ax=ax)
        # plt.show()

        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np.astype(np.float32))  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN

        node_a = torch.from_numpy(node_a_np)  # 3xMa
        node_b = torch.from_numpy(node_b_np)  # 3xMb

        P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        #img_label = torch.from_numpy(img_label.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        img_label = torch.from_numpy(img_label.astype(np.float32)).contiguous()  # HxW
        #img_label = torch.tensor(img_label, dtype=torch.long)
        K = torch.from_numpy(K.astype(np.float32))  # 3x3

        t_ji = torch.from_numpy(t_ji.astype(np.float32))  # 3

        inside_mask = torch.from_numpy(inside_mask.astype(np.bool))

        return pc, img, img_label,\
               P, K

#remain old version
def view_per_point_cloud_label(pc, img, gt_Rt, K):
    B, C, H, W = img.shape[0], img.shape[1],img.shape[2],img.shape[3]
    B, C_pc, N = pc.shape[0], pc.shape[1], pc.shape[2]
    pc_homo = torch.cat((pc,
                         torch.ones((B, 1, N), dtype=torch.float32, device=pc.device)),
                        dim=1)  # Bx4xN
    P_pc_homo = torch.matmul(gt_Rt, pc_homo)  # Bx3xN
    KP_pc_homo = torch.matmul(K, P_pc_homo)  # Bx3xN
    KP_pc_pxpy = KP_pc_homo[:, 0:2, :] / KP_pc_homo[:, 2:3, :]  # Bx2xN

    KP_pc_pxpy = KP_pc_pxpy.to(dtype=torch.int)
    px = KP_pc_pxpy[:, 0, :]
    py = KP_pc_pxpy[:, 1, :]
    x_inside_mask = (KP_pc_pxpy[:, 0:1, :] >= 0) \
                    & (KP_pc_pxpy[:, 0:1, :] <= W - 1)  # Bx1xN
    y_inside_mask = (KP_pc_pxpy[:, 1:2, :] >= 0) \
                    & (KP_pc_pxpy[:, 1:2, :] <= H - 1)  # Bx1xN
    z_inside_mask = KP_pc_homo[:, 2:3, :] > 0.1
    inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1) # Bx1xN # BxN

    # #########debug
    if 1:
        print('######')
        imgs_np = img.detach().round() \
            .to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # Bx3xHxW -> BxHxWx3
        px = px.detach().cpu().numpy()
        py = py.detach().cpu().numpy()
        b = 0
        d_img = imgs_np[b, :, :, 0:3]
        H_delta = 0
        W_delta = 0
        H_large = H + int(H_delta * 2)
        W_large = W + int(W_delta * 2)
        img_vis_fine_np = np.zeros((H_large, W_large, 3), dtype=np.uint8) + 255
        img_vis_fine_np[H_delta:H_delta + H, W_delta:W_delta + W] = d_img
        t = 0
        for n in range(N):
            x = px[b, n]
            y = py[b, n]
            #ppx2 = pix[0, py * W + px]
            if inside_mask[b, n]:
                #if self.img[b, 0, y, x] > 0:c
                            #print(n)
                            #t += 1
                cv2.circle(img_vis_fine_np,
                           (x, y),
                           1,
                           (0, 255, 0),
                           1)
            else:
                cv2.circle(img_vis_fine_np,
                           (x, y),
                           1,
                           (0, 0, 255),
                           1)
        print(t)
        cv2.imwrite(r'E:\data\hn\temp\\cc.jpg', img_vis_fine_np)
    return inside_mask


def view_per_point_label(img_label, img, img_idx):
    B, C, H, W = img.shape[0], img.shape[1],img.shape[2],img.shape[3]
    inside_mask = img_label.squeeze(1) # Bx1xN # BxN

    # #########debug
    if 1:
        imgs_np = img.detach().round() \
            .to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # Bx3xHxW -> BxHxWx3
        b = 0 #bantch
        img_vis_fine_np = imgs_np[b,:,:,0:3]

        img_label_np = img_label.detach().long() \
            .to(dtype=torch.uint8).contiguous().cpu().numpy()  # BxHxW -> BxHxW
        img_vis_label_np = img_label_np[b,:,:]
        NCLASSES = 2
        colors =[[0,0,255], [0,255,0]]
        row_ind, col_ind = np.nonzero(img_vis_label_np)
        for i in range(0, len(row_ind)):
            cv2.circle(img_vis_fine_np,
                       (col_ind[i], row_ind[i]),
                       1,
                       colors[1],
                       1)

        cv2.imwrite(r'E:\data\hn\temp\test\\' + str(img_idx) + '.jpg', img_vis_fine_np)
    return inside_mask

if __name__ == '__main__':
    opt = options.Options()
    argoloader = argoLoader(opt.dataroot, 'train', opt)

    for i in range(0, len(argoloader), 1):
        print('--- %d ---' % i)
        data = argoloader[i]
        for item in data:
            print(item.size())
