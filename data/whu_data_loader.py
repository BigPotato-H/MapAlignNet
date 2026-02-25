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
import options_whu
from matplotlib import pyplot as plt

from pathlib import Path

from deeplab.utils import get_label_mask, set_class_values,LABEL_COLORS_LIST,ALL_CLASSES
from scipy.spatial.transform import Rotation as R

import albumentations as A
import numpy as np
import cv2
import pandas as pd
import glob
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import MultiLineString
from shapely import wkt


image_width = 8192
image_height = 4096
#image_width = 8000
#image_height = 4000


#db_host = "localhost"        # Database host
db_host = "172.26.0.1"
db_port = "5433"             # Port (default for PostgreSQL is 5432)
db_name = "whu_hn"    # Replace with your database name
db_user = "postgres"    # Replace with your username
db_password = "postgres"  # Replace with your password
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

def get_range_from_pose(x, y, buffer_distance):
    """
    Given a pose (x, y) and a buffer distance, calculate the geographic range (bounding box).

    Parameters:
        x (float): X coordinate (longitude or easting).
        y (float): Y coordinate (latitude or northing).
        buffer_distance (float): Half the size of the range in the same units as x and y.

    Returns:
        tuple: (min_x, min_y, max_x, max_y) representing the bounding box.
    """
    min_x = x - buffer_distance
    max_x = x + buffer_distance
    min_y = y - buffer_distance
    max_y = y + buffer_distance
    return (min_x, min_y, max_x, max_y)


def random_jitter_6dof(pose):
    translation_jitter = np.random.uniform(-1, 1, 3)  # Translation jitter in range [-1, 1] meters
    rotation_jitter = np.random.uniform(-0.04, 0.04, 3)  # Rotation jitter in range [-0.04, 0.04] radians
    jitter_pose = pose + np.concatenate([translation_jitter, rotation_jitter])
    return jitter_pose

def query_with_geo_range(bounding_box):
    (min_x, min_y, max_x, max_y) = bounding_box
    # Create a bounding box (polygon)
    bbox_query = f"""
        SELECT 
            id,
            geom,
            ST_AsText(ST_Intersection(geom, ST_MakeEnvelope({min_x}, {min_y}, {max_x}, {max_y}, 4547))) AS intersection_wkt
        FROM whu_z
        WHERE ST_Intersects(
            geom,
            ST_MakeEnvelope({min_x}, {min_y}, {max_x}, {max_y}, 4547)
        )
    """

    # Execute the query and load results into a GeoDataFrame
    gdf = gpd.read_postgis(bbox_query, con=engine)
    lines = extract_points_from_intersection(gdf['intersection_wkt'])
    return lines

def insert_line_shape(line):
    interval = 0.1

    # Get the total length of the LineString
    line_length = line.length

    # Calculate the number of points to insert
    num_points = int(line_length // interval)

    # Generate points at equal intervals
    points = [line.interpolate(i * interval) for i in range(1, num_points + 1)]
    return points


def extract_points_from_intersection(wkt_strings):
    lines = []
    for wkt_string in wkt_strings:
        geometry = wkt.loads(wkt_string)  # Convert WKT string to a Shapely geometry
        insert_line = insert_line_shape(geometry)
        points = []
        for point in insert_line:  # Iterate over each LineString in the MultiLineString
             points.append(np.array([point.x, point.y, point.z]))
        lines.append(points)

    return lines

def cartesian_to_image(x, y, z):
    if 0:
    #cartesian_to_spherical
        r = np.sqrt(x**2 + y**2 + z**2)  # radius
        lat = np.arcsin(z / r)  # latitude
        lon = np.arctan2(y, x)  # longitude

        #spherical_to_image
        x = (lon + np.pi) / (2 * np.pi) * image_width  # Normalize longitude to [0, image_width]
        y = (np.pi / 2 - lat) / np.pi * image_height  # Normalize latitude to [0, image_height]
    else:
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # radius
        lat = np.arccos(y / r)  # latitude
        lon = np.arctan2(x, z)  # longitude

        # spherical_to_image
        x = (lon + np.pi) / (2 * np.pi) * image_width  # Normalize longitude to [0, image_width]
        y = (np.pi  - lat) / np.pi * image_height  #

    crop_x, crop_y = image_width / 2 - 1024, image_height / 2 - 1024
    x = x - crop_x
    y = y - crop_y

    return int(x), int(y)


def camera_to_pano(ego_line, scale =1.0):
    image_points = []

    for point in ego_line:
        img_x, img_y = cartesian_to_image(point[0], point[1], point[2])
        if img_x > 0 and img_x < 2048 and \
            img_y > 0 and img_y < 2048:
            img_x = img_x * scale
            img_y = img_y * scale
            image_points.append((img_x, img_y))

    return image_points

def pose_6dof_to_matrix(tr):
    t = tr[0:3]
    euler = tr[3:]
    r = R.from_euler('zyx', euler, degrees=False)
    rotation_matrix = r.as_matrix()
    return t, rotation_matrix

def ego_to_camera(ego_line,tr):
    t, rotation_matrix = pose_6dof_to_matrix(tr)
    transformed_points = ego_line @ rotation_matrix.T + t
    return transformed_points
    #return ego_line

def ego_to_pano(ego_line, tr, scale=1.0):
    cam_map_line = ego_to_camera(ego_line, tr)
    image_line = camera_to_pano(cam_map_line, scale)
    return image_line

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

def read_HDI(root_path,log_list):
    # Define the column names based on the structure of the line
    columns = [
        "image_na", "No.","Year", "Month", "Day", "Hour", "Minute", "Second", "Millisecond",
        "X", "Y", "Z","lon","lat", "Heading", "Pitch", "Roll",
        "Angular_Velocity_X", "Angular_Velocity_Y", "Angular_Velocity_Z", "Signal"
    ]

    for log_id in log_list:
        #hdi_files = glob.glob(os.path.join(root_path,log_id, '*.hdi'))
        #hdi_file = hdi_files[0]
        hdi_file = os.path.join(root_path, log_id, 'iScan-Image-1.hdi')
        ins_dict = {}
        # Open the file and read each line
        with open(hdi_file, 'r') as file:
            for line in file:
                # Split the line by tab characters
                values = line.strip().split('\t')

                # Ensure the line has the same number of columns as defined
                if len(values) == len(columns):
                    image_name = values[0] + '.jpg'  # Assuming the first column is image_name

                    # Create a dictionary for the line data
                    line_data = dict(zip(columns, values))

                    # Store the line data in the dictionary with image_name as the key
                    ins_dict[image_name] = line_data

    return ins_dict

def read_GTPose(root_path,log_list):
    # Define the column names based on the structure of the line
    columns = ["image_na", "X", "Y", "Z", "Heading", "Pitch", "Roll"]
    df_list = []
    for log_id in log_list:
        pose_file = os.path.join(root_path,log_id, 'gt_pose.csv')
        df = pd.read_csv(pose_file, names=columns)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df


def make_whu_dataset(root_path,log_list):
    dataset = []
    for log_id in log_list:
        log_folder = os.path.join(root_path, log_id)
        map_folder = os.path.join(log_folder, 'hdmap')
        img_folder = os.path.join(log_folder, 'image')
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


class whuLoader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(whuLoader, self).__init__()
        self.root = root + mode
        #self.root = root
        self.opt = opt
        self.mode = mode

        # store the calibration matrix for each sequence
        self.log_list = []
        if len(opt.log_id) == 0:
            self.log_list = os.listdir(self.root)
        else:
            self.log_list = [opt.log_id]
        self.ins_dict = read_HDI(self.root, self.log_list)
        self.pose_df = read_GTPose(self.root, self.log_list)
        self.dataset = make_whu_dataset(self.root, self.log_list)

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

        self.scale = opt.img_H / 2048
        self.save_label_img = True
        self.view_path = os.path.join(self.root,opt.log_id, 'view')
        self.label_path = os.path.join(self.root,opt.log_id, 'label')
        self.unalignlabel_path = os.path.join(self.root,opt.log_id, 'unalignlabel')
        os.makedirs(self.view_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)
        os.makedirs(self.unalignlabel_path, exist_ok=True)

        self.file = self.opt.decalib_file
        if self.file is not None:
            self.perturb_arr = np.loadtxt(self.file, dtype=np.float32, delimiter=',')

    def augment_pc(self, pc_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        return pc_np

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
        rot = R.from_euler('zyx', angles, degrees=False).as_matrix()
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

        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = self.cal_rotate_matrix(angles)
        P_random = np.identity(4, dtype=float)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def downsample_np(self, pc_np):
        if pc_np.shape[1] >= self.opt.input_pt_num:
            choice_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.opt.input_pt_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]

        return pc_np


    def get_local_map(self, ins, pose_gt):
        # "X", "Y", "Z", "Heading", "Pitch", "Roll",
        bounding_box = get_range_from_pose(float(ins["X"]), float(ins["Y"]), 100)
        heading = float(ins["Heading"])
        #euler_angles = [float(pose["Roll"]) + 90, float(pose["Pitch"]), heading]
        euler_angles = [heading, float(ins["Pitch"]),float(ins["Roll"]) + 90]
        rotation_matrix = R.from_euler('zyx', euler_angles, degrees=True).as_matrix()
        translation = [float(ins["X"]), float(ins["Y"]), float(ins["Z"])]
        translation = np.array(translation)

        global_lines = query_with_geo_range(bounding_box)
        ego_lines = []
        for global_line in global_lines:
            if len(global_line) == 0:
                continue
            relative_points = global_line - translation
            ego_line = relative_points @ rotation_matrix.T
            # Filter points within the local map extent (cube filter)
            mask = (
                    (np.abs(ego_line[:, 0]) <= 20) &
                    (np.abs(ego_line[:, 1]) <= 10) &
                    (ego_line[:, 2] > 0)  & (ego_line[:, 2] <50)
                   )
            ego_line = ego_line[mask]
            ego_lines.append(ego_line)
        ego_points_array = np.vstack(ego_lines)
        return ego_lines, ego_points_array.T

    def get_label_image(self, ego_lines, tr, img, img_na,is_gt):
        # Visualize the points on the panorama
        img_bgr = img.copy()
        img_label = np.zeros((self.opt.img_H, self.opt.img_W), dtype=np.uint8)
        #tr = np.array([-0.1115, 0.4113, 0.2174, -0.0143, -0.0773, -0.0304])
        #img_points = []
        for ego_line in ego_lines:
            image_line = ego_to_pano(ego_line, tr, self.scale)
            if len(image_line) == 0:
                continue
            polyline_line = np.array(image_line, dtype=np.int32)
            #img_points.append(polyline_line)
            cv2.polylines(img_bgr, [polyline_line], isClosed=False,
                          color=(0, 0, 255), thickness=2)
            cv2.polylines(img_label, [polyline_line], isClosed=False,
                          color=(255), thickness=10)
        label_path = os.path.join(self.label_path, img_na)
        unalignlabel_path = os.path.join(self.unalignlabel_path, img_na)

        #img_points_array = np.vstack(img_points)
        if self.save_label_img:
            if is_gt:
                view_path = os.path.join(self.view_path, img_na)
                cv2.imwrite(view_path, img_bgr)  # draw map on orginal image
                cv2.imwrite(label_path, img_label)
            else:
                view_path = os.path.join(self.view_path, img_na[0:-4] + '_unalign.jpg')
                cv2.imwrite(view_path, img_bgr)
                cv2.imwrite(unalignlabel_path, img_label)

        return img_label#,img_points_array.T

    def get_label_mask(self,img_label):
        return get_label_mask(img_label, self.class_values, self.label_colors_list)

    def get_ins(self, img_na):
        return self.ins_dict.get(img_na)

    def get_GTPose(self, img_na):
        pose_record = self.pose_df[self.pose_df['image_na'] == img_na[0:-4]]
        if pose_record.empty:
            pose_gt = np.array([-0.1,0.5,-0.2,0.0189,-0.0698,0.0189])
        else:
            x = pose_record['X'].values[0]
            y = pose_record['Y'].values[0]
            z = pose_record['Z'].values[0]
            yaw = pose_record['Heading'].values[0]
            pitch = pose_record['Pitch'].values[0]
            roll = pose_record['Roll'].values[0]
            pose_gt = np.array([x,y,z,yaw, pitch, roll])
        return pose_gt

    def pose_to_matrix(self, pose):
        angles = pose[3:]
        rot = R.from_euler('zyx', angles, degrees=False).as_matrix()
        P = np.identity(4, dtype=float)
        P[0:3, 0:3] = rot
        P[0:3, 3] = pose[0:3]
        return P

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        log_id, map_folder, img_folder,img_na= self.dataset[index]
        #if index < 250:
        #    return
        ins = self.get_ins(img_na)
        pose_gt = self.get_GTPose(img_na)
        ############img#############
        img_path = os.path.join(img_folder, img_na)
        img = cv2.imread(img_path)
        # scale
        H = self.opt.img_H
        W = self.opt.img_W
        img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)

        ego_lines, pc_np  = self.get_local_map(ins,pose_gt)
        pc_np = self.downsample_np(pc_np)

        img_label = self.get_label_image(ego_lines, pose_gt, img, img_na, is_gt=True)
        img_label = cv2.resize(img_label, (H, W), interpolation=cv2.INTER_LINEAR)
        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            # -------------- augmentation ----------------------
            pc_np = self.augment_pc(pc_np)
            img = self.augment_img(img)

        Pr = []
        if self.file is not None:
            rt = self.perturb_arr[index]
            rot = R.from_euler('xyz', rt[3:6], degrees=True)
            decalib_rot = rot.as_matrix()

            P_random = np.identity(4, dtype=float)
            P_random[0:3, 0:3] = decalib_rot
            P_random[0:3, 3] = rt[0:3]
            Pr = P_random

        # perturb 4*4
        pose_jitter = pose_gt
        pose_jitter = random_jitter_6dof(pose_gt)
        ###for view
        unaligned_label_image = self.get_label_image(ego_lines, pose_jitter, img, img_na, is_gt=False)
        unaligned_label_image = cv2.resize(unaligned_label_image, (H, W), interpolation=cv2.INTER_LINEAR)

        t, r = pose_6dof_to_matrix(pose_jitter)
        # unalign_project_points
        unalign_cam_points = r @ pc_np + t[:, np.newaxis]
        unalign_project_points = camera_to_pano(unalign_cam_points.T, scale=0.25)
        unalign_project_points = np.array(unalign_project_points).T
        unalign_project_points = self.downsample_np(unalign_project_points)
        unalign_project_points = np.vstack([unalign_project_points, unalign_cam_points[2, :]])
        upp = unalign_project_points

        img = self.norm_tfms(image=img)['image']
        img_label = np.expand_dims(img_label, axis=-1)
        img_label = self.get_label_mask(img_label)
        unaligned_label_image = np.expand_dims(unaligned_label_image, axis=-1)
        unaligned_label_image = self.get_label_mask(unaligned_label_image)
        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np.astype(np.float32))  # 3xN

        pose_jitter = torch.from_numpy(pose_jitter.astype(np.float32))  # 6
        pose_gt = torch.from_numpy(pose_gt.astype(np.float32))  # 6

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        img_label = torch.from_numpy(img_label.astype(np.float32)).contiguous()  # HxW

        upp = torch.from_numpy(upp.astype(np.float32))

        unaligned_label_image = torch.from_numpy(unaligned_label_image.astype(np.float32)).contiguous()
        unaligned_label_image = unaligned_label_image.unsqueeze(0)
        return pc, img, img_label,\
               pose_gt, pose_jitter, upp,index,img_path,unaligned_label_image


if __name__ == '__main__':
    opt = options_whu.Options()
    whuloader = whuLoader(opt.input_dir, 'train', opt)
    #argoloader = argoLoader(opt.dataroot, 'val', opt)
    whuloader.save_label_img = True

    for i in range(0, len(whuloader), 1):
        print('--- %d ---' % i)
        data = whuloader[i]
