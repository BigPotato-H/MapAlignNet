# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Lidar egoview visualization."""

import logging
import os
import sys
from pathlib import Path
from typing import Final

#import click
import cv2
import numpy as np


import av2.rendering.color as color_utils
import av2.rendering.rasterize as raster_rendering_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt
from pyarrow import feather
import pandas as pd

from generate_egoview_overlaid_vector_map import generate_egoview_overlaid_map
from generate_egoview_overlaid_lidar import generate_egoview_overlaid_lidar

import csv
from shapely.geometry import LineString
from shapely.geometry import Polygon

from pyproj import Transformer
import av2.geometry.utm as utm

logger = logging.getLogger(__name__)

NUM_RANGE_BINS: Final[int] = 50
RING_CAMERA_FPS: Final[int] = 20


def write_csv(file, header_list, data_list):
    with open(file, encoding="utf-8-sig", mode="w", newline="") as f:
    #with open(file, encoding="utf-8-sig", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerows(data_list)

def save_lane_centerlines(avm):
    lane_centerlines = avm.get_scenario_lane_segments()
    data_list = []
    for lane_id, lane_obj in lane_centerlines.items():
        line = LineString(lane_obj.centerline)
        preds = lane_obj.predecessors
        succs = lane_obj.successors

        indegree = 0 if preds is None else len(preds)
        outdegree = 0 if succs is None else len(succs)

        record = []

        record.append(lane_id)
        record.append(lane_obj.has_traffic_control)
        record.append(lane_obj.is_intersection)
        record.append(lane_obj.turn_direction)
        record.append(lane_obj.l_neighbor_id)
        record.append(lane_obj.r_neighbor_id)
        record.append(lane_obj.successors)
        record.append(lane_obj.predecessors)
        record.append(line.wkt)
        data_list.append(record)

    header_list = ["id", "has_traffic_control", "is_intersection", "turn_direction",
                   "l_neighbor_id", "r_neighbor_id", "successors", "predecessors", "shp"]
    write_csv(output_dir/avm.log_id/"lc.csv", header_list, data_list)


def save_lane_dividers(avm):
    lane_segments = avm.get_scenario_lane_segments()
    data_list = []
    i = 0

    left_neigh_dict={}
    for lane_seg in lane_segments:
        if lane_seg.left_neighbor_id is not None and \
            lane_seg.id < lane_seg.left_neighbor_id:#(用于区分位置重叠方向相反的车道线，只存一次)
            left_neigh_dict[lane_seg.id] = lane_seg.left_neighbor_id

    for lane_seg in lane_segments:
        data = []
        data.append(lane_seg.id)
        data.append(lane_seg.lane_type)
        data.append(lane_seg.is_intersection)
        is_intersection = lane_seg.is_intersection | (lane_seg.right_mark_type == "NONE")
        #is_intersection =  (lane_seg.right_mark_type == "NONE")
        #data.append(is_intersection)
        data.append(avm.log_id)
        data.append(avm.city_name)

        data.append(LineString(lane_seg.right_lane_boundary.xyz))
        data_list.append(data)


        if lane_seg.left_neighbor_id is None:
            data = data[:-1]
            #data[2] = lane_seg.is_intersection | (lane_seg.left_mark_type == "NONE")
            #data[2] =  (lane_seg.left_mark_type == "NONE")
            data.append(LineString(lane_seg.left_lane_boundary.xyz))
            data_list.append(data)

        else:
            left_left_lane_id = left_neigh_dict.get(lane_seg.left_neighbor_id)
            if lane_seg.id == left_left_lane_id:
                data = data[:-1]
                data.append(LineString(lane_seg.left_lane_boundary.xyz))
                data_list.append(data)

    header_list = ["id","lane_type","is_intersection","log_id","city_name","shp"]
    write_csv(output_dir/log_id/"hd_map/ld.csv", header_list, data_list)
    #write_csv(output_dir /"hd_map/ld.csv", header_list, data_list)


def save_lane_dividers_global(avm, coord_type):
    lane_segments = avm.get_scenario_lane_segments()
    data_list = []
    i = 0

    for lane_seg in lane_segments:
        data = []
        data.append(lane_seg.id)
        data.append(lane_seg.lane_type)
        data.append(lane_seg.is_intersection)
        data.append(avm.log_id)
        data.append(avm.city_name)

        if coord_type =='WGS84':
            gg = utm.convert_city_coords_to_wgs84(lane_seg.right_lane_boundary.xyz[:,0:2], avm.city_name)
            gg[:, [0, 1]] = gg[:, [1, 0]]
        elif coord_type == 'UTM':
            gg = utm.convert_city_coords_to_utm(lane_seg.right_lane_boundary.xyz[:, 0:2], avm.city_name)

        zz = lane_seg.right_lane_boundary.xyz[:,2,None]
        ll = np.hstack((gg, zz))
        data.append(LineString(ll))
        data_list.append(data)

        #if lane_seg.left_neighbor_id is None:
        data = data[:-1]

        if coord_type =='WGS84':
            gg = utm.convert_city_coords_to_wgs84(lane_seg.left_lane_boundary.xyz[:,0:2], avm.city_name)
            gg[:, [0, 1]] = gg[:, [1, 0]]
        elif coord_type == 'UTM':
            gg = utm.convert_city_coords_to_utm(lane_seg.left_lane_boundary.xyz[:, 0:2], avm.city_name)

        zz = lane_seg.left_lane_boundary.xyz[:, 2, None]
        ll = np.hstack((gg, zz))

        data.append(LineString(ll))
        data_list.append(data)

    #header_list = ["shp"]
    header_list = ['id','lane_type','is_intersection','log_id','city_name','shp']
    file_na = 'ld_'+coord_type+'.csv'
    #write_csv(output_dir/log_id/'hd_map'/file_na, header_list, data_list)
    write_csv(output_dir / 'hd_map' / file_na, header_list, data_list)


def save_se3(se3_dict, log_id):
    file = output_dir/log_id/"hd_map/city_2_ego.csv"
    with open(file, encoding="utf-8-sig", mode="w") as f:
        list = []
        for tsp, se in se3_dict.items():
            line = se.transform_matrix.tolist()
            line = str(tsp) + ':' + str(line) +  '\n'
            list.append(line)

        f.writelines(list)
        f.close()

def save_pinhole(pinhole_cam, log_id):
    file = output_dir/log_id/"hd_map/ego_2_cam.csv"
    with open(file, encoding="utf-8-sig", mode="w") as f:
        list = pinhole_cam.ego_SE3_cam.transform_matrix.tolist()
        f.writelines(str(list))
        f.close()

    file = output_dir /log_id / "hd_map/intrinsic.csv"
    with open(file, encoding="utf-8-sig", mode="w") as f:
        list = pinhole_cam.intrinsics.K.tolist()
        f.writelines(str(list))
        f.close()

def save_objects(avm):
    ped_crossings = avm.get_scenario_ped_crossings()
    data_list = []
    i = 0
    for ped in ped_crossings:
        data = []
        data.append(ped.id)

        data.append(LineString(ped.polygon))
        data_list.append(data)

    header_list = ["id","shp"]
    write_csv(output_dir/avm.log_id/"hd_map/ped_crossings.csv", header_list, data_list)


def save_data(data_root: Path,
    output_dir: Path,
    log_id: str,
    max_range_m: float,
    use_depth_map_for_occlusion: bool,
    dump_single_frames: bool,
    cam_names):

    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
    save_lane_dividers(avm)
    #save_lane_dividers_global(avm, 'WGS84')
    #return

    #save_lane_dividers_global(avm, 'UTM')
    save_objects(avm)
    #return

    for _, cam_enum in enumerate(cam_names):
        cam_name = cam_enum.value
        pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
        save_pinhole(pinhole_cam,log_id)

        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        se3_dict = dict()
        for i, img_fpath in enumerate(cam_im_fpaths):
            if i % 50 == 0:
                logging.info(f"\tOn file {i}/{num_cam_imgs} of camera {cam_name} of {log_id}")

            cam_timestamp_ns = int(img_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                logger.info("missing LiDAR pose")
                continue
            se3_dict[cam_timestamp_ns] = city_SE3_ego
        save_se3(se3_dict,log_id)



NUM_RANGE_BINS: Final[int] = 50
RING_CAMERA_FPS: Final[int] = 20


def save_pointcloud(out_path, pc):
    f = open(out_path,'w')
    for i in range(pc.shape[0]):
        line = ''
        for j in pc[i]:
            line = line + str(j) + ','
        line = line.rstrip(',')
        line = line + '\n'
        f.write(line)
    f.close()


def save_lidar(
    data_root: Path, output_dir: Path, log_id: str) -> None:
    """Render LiDAR points from a particular camera's viewpoint (color by ground surface, and apply ROI filtering).

    Args:
        data_root: path to directory where the logs live on disk.
        output_dir: path to directory where renderings will be saved.
        log_id: unique ID for AV2 scenario/log.
        render_ground_pts_only: whether to only render LiDAR points located close to the ground surface.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering, in addition to generating
            the mp4 file.

    Raises:
        RuntimeError: If vehicle log data is not present at `data_root` for `log_id`.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    lidar_folder = data_root/log_id/'sensors'/'lidar'
    lidar_list = os.listdir(lidar_folder)
    out_folder = data_root/log_id/'sensors'/'txt'
    os.makedirs(out_folder,exist_ok=True)
    attrib_spec = ['x', 'y', 'z', 'intensity', 'laser_number']
    #attrib_spec ='xyz'
    for lidar_fpath in lidar_list:
        #lidar_points_ego = io_utils.read_lidar_sweep(lidar_folder/lidar_fpath, attrib_spec=attrib_spec)
        sweep_df: pd.DataFrame = feather.read_feather(lidar_folder/lidar_fpath)
        # return only the requested point attributes
        lidar_points_ego: NDArrayFloat = sweep_df[attrib_spec].to_numpy().astype(np.float64)

        lidar_timestamp_ns = int(lidar_fpath.split('.')[0])

        # put into city coords, then prune away ground and non-RoI points
        city_SE3_ego = loader.get_city_SE3_ego(log_id, lidar_timestamp_ns)
        lidar_points_city = city_SE3_ego.transform_point_cloud(lidar_points_ego[:,0:3])
        lidar_points_city = np.concatenate((lidar_points_city, lidar_points_ego[:,3:5]), axis=1)
        #lidar_points_city = avm.remove_non_drivable_area_points(lidar_points_city)
        #is_ground_logicals = avm.get_ground_points_boolean(lidar_points_city)
        #lidar_points_city = lidar_points_city[is_ground_logicals if render_ground_pts_only else ~is_ground_logicals]
        lidar_points_ego = city_SE3_ego.inverse().transform_point_cloud(lidar_points_city[:,0:3])
        out_lidar_path = out_folder/(str(lidar_timestamp_ns) +'.txt')
        save_pointcloud(out_lidar_path, lidar_points_city)



def run_generate_data_hn(
    data_root: "os.PathLike[str]",
    output_dir: "os.PathLike[str]",
    log_id: str,
    max_range_m: float,
    use_depth_map_for_occlusion: bool,
    dump_single_frames: bool,
    cam_names
    #cam_names: List[RingCameras],
) -> None:

    logger.info(
        "data_root: %s, output_dir: %s, log_id: %s, max_range_m: %f, "
        "use_depth_map_for_occlusion: %s, dump_single_frames %s",
        data_root,
        output_dir,
        log_id,
        max_range_m,
        use_depth_map_for_occlusion,
        dump_single_frames,
    )

    #generate_egoview_overlaid_lidar(data_root, output_dir, log_id, render_ground_pts_only, dump_single_frames)
    generate_egoview_overlaid_map(data_root, output_dir, log_id, max_range_m, use_depth_map_for_occlusion, dump_single_frames, cam_names)


def get_all_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


def save_whole_map():
    # 获取该目录下的所有文件夹
    folders = get_all_folders(data_root)
    for log_id in folders:
        print(log_id)
        save_data(data_root, output_dir, log_id, 100, True, True, cam_names)


if __name__ == "__main__":
    #data_root = '/mnt/f/0hn/4data/argoverse/2/tbv/'
    #data_root = '/mnt/f/'
    #data_root = 'F:/0hn/4data/argoverse/2/tbv/'
    data_root = 'G:/0hn/data/argoverse/2/tars/tbv/'
    #data_root =r'G:\0hn\data\argoverse\2\tars\sensor\train\\'
    output_dir = data_root
    #log_id = '01bb304d7bd835f8bbef7086b688e35e__Summer_2019'
    log_id = '05lBLQJs4ilyORCox6j9ndWAKZc31rs9__Autumn_2020'
    #log_id = '1KuQJcTCSK5HQNK5QvqSXaiWOwdORtKk__Spring_2020'
    #log_id ='0c143226-9c39-387c-a935-1391bed6dc75'
    render_ground_pts_only = False
    dump_single_frames = True
    cam_names = [RingCameras("ring_front_center")]

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    save_data(data_root, output_dir, log_id, 100, True, True, cam_names)
    #run_generate_data_hn(data_root, output_dir, log_id, 100, False, True, cam_names)
    #save_lidar(data_root, output_dir, log_id)
#    save_whole_map()

