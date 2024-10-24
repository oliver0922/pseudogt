import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
import types
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan as _dbscan, dbscan_max_cluster as _dbscan_max_cluster, transform_np_points, dbscan_cluster_filter as _dbscan_cluster_filter
from utils.registration_utils import full_registration, fragmentized_full_registration, full_pc_registration
from utils.open3d_utils import set_black_background, set_white_background
from utils.instance_merge_utils import id_merging, merge_instance_ids
from utils.visualizer_utils import visualizer
from utils.object_movement_utils import find_dynamic_objects, dynamic_object_registration
from utils.save_utils import Save_PseudoGT
from utils.bounding_box_utils import BoundingBox

CAM_LOCS = {1:'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}
CAM_NAMES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
#CAM_NAMES = ['FRONT_RIGHT']

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
LIDAR_TO_CAMERA = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def dbscan(pcd):
    dis = np.mean(np.linalg.norm(np.array(pcd.points), axis=1))
    dis = dis * (0.3 * np.pi / 180) * 2.0
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(np.array(pcd.points))
    src.paint_uniform_color([1, 0.706, 0])
    unnoise_idx = _dbscan(src, eps=dis, min_points=5)
    if len(unnoise_idx) == 0:
        return o3d.geometry.PointCloud()
    return src[unnoise_idx]



def dbscan_max_cluster(pcd):
    dis = np.mean(np.linalg.norm(np.array(pcd.points), axis=1))
    dis = dis * (0.3 * np.pi / 180) * 2.0
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(np.array(pcd.points))
    src.paint_uniform_color([1, 0.706, 0])
    unnoise_idx = _dbscan_max_cluster(src, eps=dis, min_points=5)
    if len(unnoise_idx) == 0:
        return o3d.geometry.PointCloud()
    return src[unnoise_idx]



def dbscan_per_frame_instance(instance_frame_pcd):
    dis = np.mean(np.linalg.norm(instance_frame_pcd, axis=1))
    dis = dis * (0.3 * np.pi / 180) * 3.0
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(instance_frame_pcd)
    src.paint_uniform_color([1, 0.706, 0])
    unnoise_idx = _dbscan_cluster_filter(src, eps=dis, min_points=5, max_dist=3.0)
    if len(unnoise_idx) == 0:
        return np.array([])

    return np.array(src.points)[unnoise_idx]



def find_gtbbox(target_src, frame_idx):
    gt_bbox_file = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(frame_idx).zfill(6)}.bin')).reshape(-1, 7)
    center = np.mean(target_src.points, axis=0)
    min_distace = 100000
    min_idx = 0
    for i in range(len(gt_bbox_file)):
        distance = np.linalg.norm(center - gt_bbox_file[i, :3])
        if distance < min_distace:
            min_distace = distance
            min_idx = i
    line_set_gt, _ = BoundingBox.load_gt(gt_bbox_file[min_idx]).get_o3d_instance()
    line_set_gt.paint_uniform_color([0, 0, 1])
    return line_set_gt



def get_valid_transformations(transformation_matrix, target_line_set_lidar, get_rotation=False):
    tr_matrix = transformation_matrix
    tr_matrix = np.linalg.inv(tr_matrix)
    new_tr_mat = np.eye(4)
    rotation = np.arctan2(tr_matrix[1, 0], tr_matrix[0, 0])
    if get_rotation:
        return rotation
    new_tr_mat[:3, :3] = R.from_euler('z', rotation).as_matrix()
    new_tr_mat[:3, 3] = copy.deepcopy(target_line_set_lidar).transform(tr_matrix).get_center() - copy.deepcopy(target_line_set_lidar).transform(new_tr_mat).get_center()
    return new_tr_mat



def pcd_face_detection(pcd):
    angle_diff_threshold = np.pi / 3
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    normals = np.array(pcd.normals)
    normals = normals[abs(normals[:, 1]) < 0.3]
    normals = normals[:, [0, 2]]
    normal_angles = np.arctan2(normals[:, 1], normals[:, 0])


    box_angle = BoundingBox(pcd, 'point_normal').r
    # count angles near box angle, box angle + pi/2, box angle + pi, box angle + 3pi/2
    angle_list = [box_angle, box_angle + np.pi/2, box_angle + np.pi, box_angle + 3*np.pi/2]
    angle_list = np.array(angle_list)

    angle_diff = np.abs(angle_list - normal_angles[:, None])
    
    count = np.sum(angle_diff < angle_diff_threshold, axis=0)
    face_done = np.logical_or(count > 30, count > max(3, len(normals) * 0.2))
    done_count = np.sum(face_done)
    return angle_list, face_done, done_count



def locate_bbox(pcd, bbox_size, prev_direction, give_initial_box=False):
    # Given camera location(0, 0, 0), we can find bbox face visible at camera
    # we are going to extend invisible bbox face to fit the bbox size
    # prev_direction is given to find long side of bbox
    obj = BoundingBox(pcd, 'point_normal')
    # if diff between prev_direction and bbox direction is near 90 degree or 270 degree, we need to add or sub 90 degree
    # check near 90 degree or 270 degree
    if np.abs(np.abs(obj.r - prev_direction) - np.pi/2) < np.pi/6 or np.abs(np.abs(obj.r - prev_direction) - 3*np.pi/2) < np.pi/6:
        if np.abs(obj.r - prev_direction) < np.pi/2:
            obj.r += np.pi/2
            obj.extent = np.array([obj.extent[1], obj.extent[0], obj.extent[2]])
        else:
            obj.r -= np.pi/2
            obj.extent = np.array([obj.extent[1], obj.extent[0], obj.extent[2]])

    obj_cp = copy.deepcopy(obj)
    # degree part is done, now we need to find bbox face visible at camera
    # we can find bbox face visible at camera by checking bbox face normal direction dot camera to face center direction
    
    # find bbox face visible at camera
    face_centers = np.array([[obj.t[0] + obj.extent[0]/2 * np.cos(obj.r), obj.t[1] + obj.extent[0]/2 * np.sin(obj.r)],
                            [obj.t[0] + obj.extent[1]/2 * np.cos(obj.r + np.pi/2), obj.t[1] + obj.extent[1]/2 * np.sin(obj.r + np.pi/2)],
                            [obj.t[0] + obj.extent[0]/2 * np.cos(obj.r + np.pi), obj.t[1] + obj.extent[0]/2 * np.sin(obj.r + np.pi)],
                            [obj.t[0] + obj.extent[1]/2 * np.cos(obj.r + 3*np.pi/2), obj.t[1] + obj.extent[1]/2 * np.sin(obj.r + 3*np.pi/2)]])
    dot_product = np.sum((face_centers - obj.t[:2]) * face_centers, axis=1)

    face_centers_with_z = np.array([[face_centers[0, 0], face_centers[0, 1], obj.t[2]],
                                    [face_centers[1, 0], face_centers[1, 1], obj.t[2]],
                                    [face_centers[2, 0], face_centers[2, 1], obj.t[2]],
                                    [face_centers[3, 0], face_centers[3, 1], obj.t[2]]])
    
    invis = dot_product >= 0
    vis = np.logical_not(invis)
    
    if np.sum(invis) == 3:
        if vis[0] or vis[2]:
            direction = (obj.t[:2] - face_centers[vis])
            normalized_direction = direction / np.linalg.norm(direction)
            l = bbox_size[0] / 2 - obj.extent[0] / 2
            obj.t[:2] += (normalized_direction * l).reshape(-1)
        else:
            direction = (obj.t[:2] - face_centers[vis])
            normalized_direction = direction / np.linalg.norm(direction)
            l = bbox_size[1] / 2 - obj.extent[1] / 2
            obj.t[:2] += (normalized_direction * l).reshape(-1)
    elif np.sum(invis) == 2:
        cor = np.sum(face_centers[vis], axis=0) - obj.t[:2]
        indices = np.where(vis)[0]
        if indices[0] % 2 == 0:
            ind1, ind2 = indices[0], indices[1]
        else:
            ind1, ind2 = indices[1], indices[0]
        dir1 = face_centers[ind1] - cor
        dir2 = face_centers[ind2] - cor
        dir1 = dir1 / np.linalg.norm(dir1) * (bbox_size[1] / 2 - obj.extent[1] / 2)
        dir2 = dir2 / np.linalg.norm(dir2) * (bbox_size[0] / 2 - obj.extent[0] / 2)
        obj.t[:2] = obj.t[:2] + dir1 + dir2
    else:
        if give_initial_box:
            return None, None
        return None

    obj.extent = bbox_size
    if give_initial_box:
        line_set, _ = BoundingBox(pcd, 'point_normal').get_o3d_instance()
        return copy.deepcopy(obj), line_set
    return copy.deepcopy(obj)
    


def main(args):
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)
    save_data = Save_PseudoGT(f'./output/', f'scene-{args.scene_idx}_pseudo_gt.pkl')

    pcd_list = []
    pcd_color_list = []
    pcd_id_list = []

    unique_instance_id_list = []
    for frame_idx in idx_range:
        pcd_with_instance_id = []
        pcd_color = []
        if args.multicam:
            for cam_name in CAM_NAMES:
                try:
                    pcd_with_instance_id.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', cam_name, 'visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4))
                except:
                    print(f"scene-{args.scene_idx} {cam_name} {frame_idx} is not found")
                    continue
                try:
                    pcd_color.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', cam_name, 'visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3])
                except:
                    print(f"scene-{args.scene_idx} {cam_name} {frame_idx} is not found")
                    continue
        else:
            try:
                pcd_with_instance_id.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', 'visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4))
            except:
                print(f"scene-{args.scene_idx} {frame_idx} is not found")
                continue
            try:
                pcd_color.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', 'visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3])
            except:
                print(f"scene-{args.scene_idx} {frame_idx} is not found")
                continue

        pcd_with_instance_id = np.array(pcd_with_instance_id)
        pcd_color = np.array(pcd_color)

        if len(pcd_with_instance_id) == 0:
            pcd_list.append(np.array([]))
            pcd_color_list.append(np.array([]))
            pcd_id_list.append(np.array([]))
            continue

        pcd = pcd_with_instance_id[:, :3]
        pcd_id = pcd_with_instance_id[:, 3]

        pcd_list.append(pcd)
        pcd_color_list.append(pcd_color)
        pcd_id_list.append(pcd_id)

        unique_instance_id_list.append(np.unique(pcd_id))

    unique_instance_id_list = np.unique(np.concatenate(unique_instance_id_list)).astype(int)

    ########################## Full Registration ########################
    try:
        world_transformation_matrices = np.load(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','world_transformation_matrices.npy'), allow_pickle=True)
    except:
        full_pc_list = []
        for i, frame_idx in enumerate(idx_range):
            try:
                full_pc = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
            except:
                idx_range = idx_range[:i]
                args.rgs_end_idx = frame_idx - 1
                break
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(full_pc)
            full_pc_list.append(src)
        pose_graph = full_pc_registration(full_pc_list, args)
        world_transformation_matrices = [np.eye(4) for _ in range(np.max(args.rgs_end_idx) + 1)]
        for i, frame_idx in enumerate(idx_range):
            world_transformation_matrices[frame_idx] = pose_graph[i]
    inv_world_transformation_matrices = [np.linalg.inv(tr) for tr in world_transformation_matrices]

    save_data.add_world_transformation_matrices(world_transformation_matrices)
    #############################################################################

    ########################## Load instance, frame pcds ########################
    instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_frame_pcd_list = [[{} for _ in range(np.max(idx_range) + 1)] for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_pcd_color_list = {}
    for i, instance_id in enumerate(unique_instance_id_list):
        for j, frame_idx in enumerate(idx_range):
            instance_frame_pcd = pcd_list[j][pcd_id_list[j] == instance_id]
            instance_frame_pcd_color = pcd_color_list[j][pcd_id_list[j] == instance_id]

            if len(instance_frame_pcd) == 0:
                continue

            instance_frame_pcd_list[instance_id][frame_idx]["before_dbscan"] = instance_frame_pcd

            instance_pcd_color_list[instance_id] = instance_frame_pcd_color[0]

            if args.perform_db_scan_before_registration:
                instance_frame_pcd = dbscan_per_frame_instance(instance_frame_pcd)

            if len(instance_frame_pcd) == 0:
                continue

            instance_frame_pcd_list[instance_id][frame_idx]["after_dbscan"] = instance_frame_pcd
            instance_frame_pcd_list[instance_id][frame_idx]["color"] = instance_frame_pcd_color[0]

            instance_pcd_list[instance_id][frame_idx] = instance_frame_pcd
    #############################################################################

    ########################## ID merging ########################
    if args.id_merge_with_speed:
        corr, merge_distance_data = id_merging(idx_range, unique_instance_id_list, instance_pcd_list, args.speed_momentum, args.position_diff_threshold)
        instance_pcd_list, instance_pcd_color_list, unique_instance_id_list = merge_instance_ids(instance_pcd_list, instance_pcd_color_list, unique_instance_id_list, corr)
        if args.vis:
            for i in corr.keys():
                print(f"instance {i} is merged with {corr[i]}")

        print(f"Number of instances after id merging: {len(unique_instance_id_list)}")
    #############################################################################

    ########################## Dynamic Object Recognition ########################
    dynamic_instance_id_list, static_instance_id_list = find_dynamic_objects(world_transformation_matrices, instance_pcd_list, unique_instance_id_list, idx_range, args)
    ##############################################################################

    ######################### Sparse Instance ########################
    sparse_instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    new_instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for instance_id in unique_instance_id_list:
        for j, frame_idx in enumerate(idx_range):
            if frame_idx in instance_pcd_list[instance_id].keys():
                instance_frame_pcd_list[instance_id][frame_idx]["after_dbscan_id_merge"] = instance_pcd_list[instance_id][frame_idx]
                instance_frame_pcd_list[instance_id][frame_idx]["color"] = instance_pcd_color_list[instance_id]
                pcd = instance_pcd_list[instance_id][frame_idx]
                if len(pcd) <= 100 or np.mean(np.linalg.norm(pcd, axis=1)) > 50.0:
                    if len(pcd) <= args.sparse_pcd_removal_threshold:
                        continue
                    sparse_instance_pcd_list[instance_id][frame_idx] = pcd
                    continue
                new_instance_pcd_list[instance_id][frame_idx] = pcd

    instance_pcd_list = new_instance_pcd_list
    #############################################################################

    ########################## Dynamic Object ########################
    registration_data_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    sparse_bbox_data_list = [[{} for _ in range(np.max(idx_range) + 1)] for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_bounding_box_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    t_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    sparse_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for dynamic_instance_id in unique_instance_id_list:
        print(f"Dynamic instance id: {dynamic_instance_id}")
        dynamic_instance_src_dict = {}
        for frame_idx in idx_range:
            if frame_idx in instance_pcd_list[dynamic_instance_id].keys():
                src = open3d.geometry.PointCloud()
                src.points = open3d.utility.Vector3dVector(instance_pcd_list[dynamic_instance_id][frame_idx])
                src.paint_uniform_color(instance_pcd_color_list[dynamic_instance_id])
                dynamic_instance_src_dict[frame_idx] = src
        dynamic_transformation_list, center_idx = dynamic_object_registration(dynamic_instance_src_dict, dynamic_instance_id, idx_range, args)
        
        new_dynamic_transformation_list = []
        for tr in dynamic_transformation_list:
            new_dynamic_transformation_list.append(np.linalg.inv(dynamic_transformation_list[center_idx]) @ tr)
        dynamic_transformation_list = new_dynamic_transformation_list
        registration_data_list[dynamic_instance_id]['transformation_matrix'] = dynamic_transformation_list
        save_data.add_registration_matrix(dynamic_instance_id, dynamic_transformation_list)
        
        dynamic_registered_pcd = []
        dynamic_instance_src_list = []
        dynamic_instance_pcd_frame_idx_list = list()
        ptr = 0
        for frame_idx in idx_range:
            if frame_idx in dynamic_instance_src_dict.keys():
                pcd = dynamic_instance_src_dict[frame_idx]
                pcd.transform(dynamic_transformation_list[ptr])
                dynamic_registered_pcd.extend(np.array(pcd.points))
                dynamic_instance_src_list.append(pcd)
                dynamic_instance_pcd_frame_idx_list.append(frame_idx)
                ptr += 1
        dynamic_registered_src = open3d.geometry.PointCloud()
        dynamic_registered_src.points = open3d.utility.Vector3dVector(dynamic_registered_pcd)
        dynamic_registered_src.paint_uniform_color(instance_pcd_color_list[dynamic_instance_id])
        registration_data_list[dynamic_instance_id]['registered_src'] = dynamic_registered_src

        if args.dbscan_each_instance and len(dynamic_registered_src.points) > 500:
            if args.dbscan_max_cluster:
                dynamic_registered_src = dbscan_max_cluster(dynamic_registered_src)
            else:
                dynamic_registered_src = dbscan(dynamic_registered_src)

        line_set_lidar, _ = BoundingBox(dynamic_registered_src, args.bbox_gen_fit_method).get_o3d_instance()
        t_line_set_lidar, _ = BoundingBox(dynamic_registered_src, 'closeness_to_edge').get_o3d_instance()

        if line_set_lidar is None:
            continue

        if dynamic_instance_id in dynamic_instance_id_list:
            line_set_lidar.paint_uniform_color([1, 0, 0])
            t_line_set_lidar.paint_uniform_color([0, 1, 0])
        else:
            line_set_lidar.paint_uniform_color([0.6, 0.3, 0.6])
            t_line_set_lidar.paint_uniform_color([0.3, 0.6, 0.6])

        registration_data_list[dynamic_instance_id]['line_set_lidar'] = line_set_lidar
        registration_data_list[dynamic_instance_id]['t_line_set_lidar'] = t_line_set_lidar

        gt_lines = find_gtbbox(dynamic_registered_src, dynamic_instance_pcd_frame_idx_list[center_idx])
        registration_data_list[dynamic_instance_id]['gt_lines'] = gt_lines

        for i, frame_idx in enumerate(dynamic_instance_pcd_frame_idx_list):
            bbox = copy.deepcopy(line_set_lidar)
            t_bbox = copy.deepcopy(t_line_set_lidar)
            tr_matrix = get_valid_transformations(dynamic_transformation_list[i], line_set_lidar)
            bbox = bbox.transform(tr_matrix)
            t_bbox = t_bbox.transform(tr_matrix)
            instance_bounding_box_list[dynamic_instance_id][frame_idx] = bbox
            t_bbox_list[dynamic_instance_id][frame_idx] = t_bbox

        for frame_idx in sparse_instance_pcd_list[dynamic_instance_id].keys():
            bbox = BoundingBox(dynamic_registered_src, args.bbox_gen_fit_method)
            bbox_size = bbox.s
            ry = bbox.r
            nearest_i = np.argmin(np.abs(np.array(dynamic_instance_pcd_frame_idx_list) - frame_idx))
            prev_direction = get_valid_transformations(dynamic_transformation_list[nearest_i], line_set_lidar, get_rotation=True)
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(sparse_instance_pcd_list[dynamic_instance_id][frame_idx])
            src.paint_uniform_color(instance_pcd_color_list[dynamic_instance_id])
            this_bbox, init_line = locate_bbox(src, bbox_size, prev_direction + ry, give_initial_box=True)
            if this_bbox is None:
                continue
            line_set, _ = this_bbox.get_o3d_instance()

            line_set.paint_uniform_color([1, 0.706, 0])
            init_line.paint_uniform_color([0.706, 0.706, 0])
            sparse_bbox_list[dynamic_instance_id][frame_idx] = line_set
            gt_lines = find_gtbbox(src, frame_idx)

            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['src'] = src
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['line_set'] = line_set
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['init_line'] = init_line
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['gt_lines'] = gt_lines
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['nearest_bbox'] = instance_bounding_box_list[dynamic_instance_id][dynamic_instance_pcd_frame_idx_list[nearest_i]]
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['nearest_registered_idx'] = dynamic_instance_pcd_frame_idx_list[nearest_i]
            sparse_bbox_data_list[dynamic_instance_id][frame_idx]['nearest_i'] = nearest_i

        print(f"Dynamic instance {dynamic_instance_id} is done")
    ##############################################################################

    ############################# Static Object ########################
    static_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for static_instance_id in static_instance_id_list:
        print(f"Static instance id: {static_instance_id}")
        static_registered_pcd = []
        ptr = 0
        for frame_idx in idx_range:
            if frame_idx in instance_pcd_list[static_instance_id].keys():
                pcd = instance_pcd_list[static_instance_id][frame_idx]
                pcd = transform_np_points(pcd, world_transformation_matrices[frame_idx])
                static_registered_pcd.extend(pcd)
                ptr = frame_idx
            if frame_idx in sparse_instance_pcd_list[static_instance_id].keys():
                pcd = sparse_instance_pcd_list[static_instance_id][frame_idx]
                pcd = transform_np_points(pcd, world_transformation_matrices[frame_idx])
                static_registered_pcd.extend(pcd)
                ptr = frame_idx

        if len(static_registered_pcd) == 0:
            continue
        
        static_src = open3d.geometry.PointCloud()
        static_src.points = open3d.utility.Vector3dVector(static_registered_pcd)
        static_src.paint_uniform_color(instance_pcd_color_list[static_instance_id])
        static_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        static_src.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        registration_data_list[static_instance_id]['registered_src'] = static_src

        if args.dbscan_each_instance and len(static_registered_pcd) > 500:
            if args.dbscan_max_cluster:
                static_src = dbscan_max_cluster(static_src)
            else:
                static_src = dbscan(static_src)

        line_set_lidar, _ = BoundingBox(static_src, args.bbox_gen_fit_method).get_o3d_instance()
        t_line_set_lidar, _ = BoundingBox(static_src, 'closeness_to_edge').get_o3d_instance()

        if line_set_lidar is None:
            continue

        line_set_lidar.paint_uniform_color([1, 0, 0.706])
        t_line_set_lidar.paint_uniform_color([0, 1, 0.706])
        registration_data_list[static_instance_id]['line_set_lidar'] = line_set_lidar
        registration_data_list[static_instance_id]['t_line_set_lidar'] = t_line_set_lidar

        gt_lines = find_gtbbox(copy.deepcopy(static_src).transform(world_transformation_matrices[ptr]), ptr)
        registration_data_list[static_instance_id]['gt_lines'] = copy.deepcopy(gt_lines).transform(inv_world_transformation_matrices[ptr])

        for i, frame_idx in enumerate(idx_range):
            bbox = copy.deepcopy(line_set_lidar)
            tr_matrix = get_valid_transformations(world_transformation_matrices[i], line_set_lidar)
            bbox = bbox.transform(tr_matrix)
            bbox.paint_uniform_color([0.666, 0.666, 0.666])
            static_bbox_list[static_instance_id][frame_idx] = bbox

        print(f"Static instance {static_instance_id} is done")
        ##############################################################################

    if args.vis:
        import pickle
        output_dir = f'./output/scene-{args.scene_idx}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'./{output_dir}/world_transformation_matrices.pkl', 'wb') as f:
            pickle.dump(world_transformation_matrices, f)

        visualizer(instance_bounding_box_list, t_bbox_list, sparse_bbox_list, unique_instance_id_list, registration_data_list, sparse_bbox_data_list, instance_frame_pcd_list, merge_distance_data, static_bbox_list, idx_range, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pseudo bounding generation ')
    parser.add_argument('--dataset_path', type=str, default='/workspace/3df_data/waymo_sam2')
    parser.add_argument('--visible_bbox_estimation', type=bool, default=True)
    parser.add_argument('--perform_db_scan_before_registration', type=bool, default=True)
    parser.add_argument('--with_gt_box', type=bool, default=False)
    parser.add_argument('--axis_aligned', type=bool, default=True)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--orient', type=bool, default=True)
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--scene_idx', type=int,default=678)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=0)
    parser.add_argument('--rgs_end_idx',type=int, default=198)
    parser.add_argument('--origin',type=bool, default=False)
    parser.add_argument('--clustering',type=str, default='dbscan')
    parser.add_argument('--dbscan_each_instance', type=bool, default=False)
    parser.add_argument('--bbox_gen_fit_method', type=str, default='point_normal')

    parser.add_argument('--dbscan_max_cluster', type=bool, default=True)
    parser.add_argument('--id_merge_with_speed', type=bool, default=True)
    parser.add_argument('--position_diff_threshold', type=float, default=1.0)
    parser.add_argument('--speed_momentum', type=float, default=0.5)

    parser.add_argument('--registration_with_full_pc', type=bool, default=False)
    parser.add_argument('--z_threshold', type=float, default=0.4)

    parser.add_argument('--fragmentized_registration', type=bool, default=False)
    parser.add_argument('--fragment_size', type=int, default=15)

    parser.add_argument('--multicam', type=bool, default=False)
    parser.add_argument('--dynamic_threshold', type=float, default=0.2)
    parser.add_argument('--dynamic_threshold_single', type=float, default=0.5)

    parser.add_argument('--sparse_pcd_removal_threshold', type=int, default=10)

    args = parser.parse_args()

    if args.origin:
        source = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(args.src_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)

        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(source[:, :3])
        src.paint_uniform_color([1, 0.706, 0])
        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.ones(3)


        vis.add_geometry(AXIS_PCD)
        vis.add_geometry(src)
        vis.run()
    
    main(args)