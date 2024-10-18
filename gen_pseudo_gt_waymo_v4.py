import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
import types
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan as _dbscan, get_obj,translate_boxes_to_open3d_instance, translate_boxes_to_open3d_gtbox, dbscan_max_cluster as _dbscan_max_cluster, translate_boxes_to_lidar_coords, translate_obj_to_open3d_instance
from utils.registration_utils import full_registration
from utils.open3d_utils import set_black_background, set_white_background
from utils.instance_merge_utils import id_merging, merge_instance_ids
from utils.visualizer_utils import visualizer

CAM_LOCS = {1:'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}
CAM_NAMES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
#CAM_NAMES = ['FRONT_RIGHT']

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
LIDAR_TO_CAMERA = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])

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
    dis = dis * (0.3 * np.pi / 180) * 2.5
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(instance_frame_pcd)
    src.paint_uniform_color([1, 0.706, 0])
    unnoise_idx = _dbscan_max_cluster(src, eps=dis, min_points=5)
    if len(unnoise_idx) == 0:
        return np.array([])

    return np.array(src.points)[unnoise_idx]



def gen_bbox(pcd, fit_method, only_angle=False, only_size=False):
    camera_coord_pcd = np.array(pcd.points) @ LIDAR_TO_CAMERA.T
    if len(pcd.points) == 0:
        return None, None
    obj = get_obj(camera_coord_pcd, fit_method)
    if only_angle:
        return np.pi/2 - obj.ry
    if only_size:
        return obj.l, obj.w, obj.h
    _, box3d = translate_boxes_to_open3d_instance(obj)
    return translate_boxes_to_lidar_coords(box3d, obj.ry, LIDAR_TO_CAMERA)



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
    gt_bbox = gt_bbox_file[min_idx]
    line_set_gt, _ = translate_boxes_to_open3d_gtbox(gt_bbox)
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

    box_angle = gen_bbox(pcd, 'point_normal', only_angle=True)
    obj = get_obj(np.array(pcd.points), 'point_normal')
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
    cam_coord_pcd = np.array(pcd.points) @ LIDAR_TO_CAMERA.T
    tmp = get_obj(cam_coord_pcd, 'point_normal')
    obj = types.SimpleNamespace()
    obj.center = LIDAR_TO_CAMERA.T @ tmp.t
    obj.extent = np.array([tmp.l, tmp.w, tmp.h])
    obj.ry = np.pi/2 - tmp.ry
    # if diff between prev_direction and bbox direction is near 90 degree or 270 degree, we need to add or sub 90 degree
    # check near 90 degree or 270 degree
    if np.abs(np.abs(obj.ry - prev_direction) - np.pi/2) < np.pi/6 or np.abs(np.abs(obj.ry - prev_direction) - 3*np.pi/2) < np.pi/6:
        if np.abs(obj.ry - prev_direction) < np.pi/2:
            obj.ry += np.pi/2
        else:
            obj.ry -= np.pi/2

    obj_cp = copy.deepcopy(obj)
    # degree part is done, now we need to find bbox face visible at camera
    # we can find bbox face visible at camera by checking bbox face normal direction dot camera to face center direction
    
    # find bbox face visible at camera
    face_centers = np.array([[obj.center[0] + obj.extent[0]/2 * np.cos(obj.ry), obj.center[1] + obj.extent[0]/2 * np.sin(obj.ry)],
                            [obj.center[0] + obj.extent[1]/2 * np.cos(obj.ry + np.pi/2), obj.center[1] + obj.extent[1]/2 * np.sin(obj.ry + np.pi/2)],
                            [obj.center[0] + obj.extent[0]/2 * np.cos(obj.ry + np.pi), obj.center[1] + obj.extent[0]/2 * np.sin(obj.ry + np.pi)],
                            [obj.center[0] + obj.extent[1]/2 * np.cos(obj.ry + 3*np.pi/2), obj.center[1] + obj.extent[1]/2 * np.sin(obj.ry + 3*np.pi/2)]])
    dot_product = np.sum((face_centers - obj.center[:2]) * face_centers, axis=1)

    face_centers_with_z = np.array([[face_centers[0, 0], face_centers[0, 1], obj.center[2]],
                                    [face_centers[1, 0], face_centers[1, 1], obj.center[2]],
                                    [face_centers[2, 0], face_centers[2, 1], obj.center[2]],
                                    [face_centers[3, 0], face_centers[3, 1], obj.center[2]]])
    
    invis = dot_product >= 0
    vis = np.logical_not(invis)
    
    if np.sum(invis) == 3:
        if vis[0] or vis[2]:
            direction = (obj.center[:2] - face_centers[vis])
            normalized_direction = direction / np.linalg.norm(direction)
            l = bbox_size[0] / 2 - obj.extent[0] / 2
            obj.center[:2] += (normalized_direction * l).reshape(-1)
        else:
            direction = (obj.center[:2] - face_centers[vis])
            normalized_direction = direction / np.linalg.norm(direction)
            l = bbox_size[1] / 2 - obj.extent[1] / 2
            obj.center[:2] += (normalized_direction * l).reshape(-1)
    elif np.sum(invis) == 2:
        cor = np.sum(face_centers[vis], axis=0) - obj.center[:2]
        # odd index is ind1, even index is ind2 in not invis
        indices = np.where(vis)[0]
        if indices[0] % 2 == 0:
            ind1, ind2 = indices[0], indices[1]
        else:
            ind2, ind1 = indices[0], indices[1]
        dir1 = face_centers[ind1] - cor
        dir2 = face_centers[ind2] - cor
        dir1 = dir1 / np.linalg.norm(dir1) * (bbox_size[1] / 2 - obj.extent[1] / 2)
        dir2 = dir2 / np.linalg.norm(dir2) * (bbox_size[0] / 2 - obj.extent[0] / 2)
        obj.center[:2] = obj.center[:2] + dir1 + dir2
    else:
        # visualize
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(face_centers_with_z[vis])
        src.paint_uniform_color([1, 0, 0])
        line_set, _ = translate_obj_to_open3d_instance(obj_cp)
        o3d.visualization.draw_geometries([src, line_set])
        raise ValueError("bbox face visible at camera is not 1 or 2")

    obj.extent = bbox_size
    if give_initial_box:
        line_set, _ = gen_bbox(pcd, 'point_normal')
        return copy.deepcopy(obj), line_set
    return copy.deepcopy(obj)
    


def main(args):


    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)

    pcd_list = []
    pcd_color_list = []
    pcd_id_list = []

    unique_instance_id_list = []
    for frame_idx in idx_range:
        pcd_with_instance_id = []
        pcd_color = []
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

    ########################## Load instance, frame pcds ########################
    sparse_instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
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
            instance_frame_pcd_list[instance_id][frame_idx]["color"] = instance_frame_pcd_color[0]

            instance_pcd_color_list[instance_id] = instance_frame_pcd_color[0]

            if args.perform_db_scan_before_registration:
                instance_frame_pcd = dbscan_per_frame_instance(instance_frame_pcd)

            if len(instance_frame_pcd) != 0:
                instance_frame_pcd_list[instance_id][frame_idx]["after_dbscan"] = instance_frame_pcd

            if len(instance_frame_pcd) <= 70 or np.mean(np.linalg.norm(instance_frame_pcd, axis=1)) > 40.0:
                if len(instance_frame_pcd) == 0:
                    continue
                sparse_instance_pcd_list[instance_id][frame_idx] = instance_frame_pcd
                continue

            instance_pcd_list[instance_id][frame_idx] = instance_frame_pcd
    #############################################################################

    ########################## ID merging ########################
    if args.id_merge_with_speed:
        corr = id_merging(idx_range, instance_pcd_list, args.speed_momentum, args.position_diff_threshold)
        instance_pcd_list, instance_pcd_color_list, unique_instance_id_list = merge_instance_ids(instance_pcd_list, instance_pcd_color_list, unique_instance_id_list, corr)
        if args.vis:
            for i in corr.keys():
                print(f"instance {i} is merged with {corr[i]}")
    #############################################################################

    ########################## Registration ########################
    registration_data_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    sparse_bbox_data_list = [[{} for _ in range(np.max(idx_range) + 1)] for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_bounding_box_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    t_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    sparse_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for instance_id in unique_instance_id_list:
        max_ptr_idx, max_ptr = 0, 0
        ptr_cnt = 0
        single_instance_pcd_list = []
        single_instance_pcd_frame_idx_list = []
        for frame_idx in idx_range:
            if frame_idx in instance_pcd_list[instance_id].keys():
                single_instance_pcd_list.append(instance_pcd_list[instance_id][frame_idx])
                single_instance_pcd_frame_idx_list.append(frame_idx)
                if len(instance_pcd_list[instance_id][frame_idx]) > max_ptr:
                    max_ptr = len(instance_pcd_list[instance_id][frame_idx])
                    max_ptr_idx = ptr_cnt
                ptr_cnt += 1

        single_instance_src_list = []
        for pcd in single_instance_pcd_list:
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(pcd)
            src.paint_uniform_color(instance_pcd_color_list[instance_id])
            src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            src.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            single_instance_src_list.append(src)

        pose_graph, mean_dis = full_registration(single_instance_src_list)
        #############################################################################

        ########################## Optimization ########################
        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=mean_dis,
            edge_prune_threshold=0.9,
            reference_node=max_ptr_idx)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        #############################################################################
        
        transformation_matrices = []
        for i in range(len(single_instance_src_list)):
            transformation_matrices.append(np.linalg.inv(pose_graph.nodes[max_ptr_idx].pose) @ pose_graph.nodes[i].pose)
        registration_data_list[instance_id]['transformation_matrix'] = copy.deepcopy(transformation_matrices)

        registered_pcd = []
        for i, pcd in enumerate(single_instance_src_list):
            pcd.transform(transformation_matrices[i])
            registered_pcd.extend(np.array(pcd.points))
        registered_src = open3d.geometry.PointCloud()
        registered_src.points = open3d.utility.Vector3dVector(registered_pcd)
        registered_src.paint_uniform_color(instance_pcd_color_list[instance_id])
        registration_data_list[instance_id]['registered_src'] = registered_src
        if args.vis:
            print(f"instance {instance_id} is registered")
            #o3d.visualization.draw_geometries_with_key_callbacks([registered_src], {ord("B"): set_black_background, ord("W"): set_white_background })

        if args.dbscan_each_instance and len(registered_pcd.points) > 500:
            if args.dbscan_max_cluster:
                registered_src = dbscan_max_cluster(registered_src)
            else:
                registered_src = dbscan(registered_src)

        line_set_lidar, _ = gen_bbox(registered_src, args.bbox_gen_fit_method)
        t_line_set_lidar, _ = gen_bbox(registered_src, 'closeness_to_edge')

        if line_set_lidar is None:
            continue

        line_set_lidar.paint_uniform_color([1, 0, 0])
        t_line_set_lidar.paint_uniform_color([0, 1, 0])
        registration_data_list[instance_id]['line_set_lidar'] = line_set_lidar
        registration_data_list[instance_id]['t_line_set_lidar'] = t_line_set_lidar

        if args.vis:
            gt_lines = find_gtbbox(registered_src, single_instance_pcd_frame_idx_list[max_ptr_idx])
            registration_data_list[instance_id]['gt_lines'] = gt_lines
            print(f"instance {instance_id} is generated")
            #o3d.visualization.draw_geometries_with_key_callbacks([line_set_lidar, registered_src, gt_lines, t_line_set_lidar], {ord("B"): set_black_background, ord("W"): set_white_background })
        #############################################################################

        ########################## Transform bbox to each frame ########################
        for i, frame_idx in enumerate(single_instance_pcd_frame_idx_list):
            bbox = copy.deepcopy(line_set_lidar)
            t_bbox = copy.deepcopy(t_line_set_lidar)
            tr_mat = get_valid_transformations(transformation_matrices[i], line_set_lidar)
            bbox.transform(tr_mat)
            t_bbox.transform(tr_mat)
            instance_bounding_box_list[instance_id][frame_idx] = bbox
            t_bbox_list[instance_id][frame_idx] = t_bbox


        for frame_idx in sparse_instance_pcd_list[instance_id].keys():
            bbox_size = np.array(gen_bbox(registered_src, args.bbox_gen_fit_method, only_size=True))
            # Get prev_direction from nearest frame
            ry = gen_bbox(registered_src, args.bbox_gen_fit_method, only_angle=True)
            # find nearest frame in single_instance_pcd_frame_idx_list
            # single_instance_pcd_frame_idx_list is sorted
            nearest_i = np.argmin(np.abs(np.array(single_instance_pcd_frame_idx_list) - frame_idx))
            prev_direction = get_valid_transformations(transformation_matrices[nearest_i], line_set_lidar, get_rotation=True) + ry
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(sparse_instance_pcd_list[instance_id][frame_idx])
            src.paint_uniform_color(instance_pcd_color_list[instance_id])
            this_bbox, init_line = locate_bbox(src, bbox_size, prev_direction, give_initial_box=True)
            line_set, _ = translate_obj_to_open3d_instance(this_bbox)
            # paint orange
            line_set.paint_uniform_color([1, 0.706, 0])

            init_line.paint_uniform_color([0.706, 0.706, 0])
            sparse_bbox_list[instance_id][frame_idx] = line_set
            gt_lines = find_gtbbox(src, frame_idx)
            #print(f"pseudo bbox for instance {instance_id} in frame {frame_idx} is generated, bbox angle from {single_instance_pcd_frame_idx_list[nearest_i]}")
            sparse_bbox_data_list[instance_id][frame_idx]['src'] = src
            sparse_bbox_data_list[instance_id][frame_idx]['line_set'] = line_set
            sparse_bbox_data_list[instance_id][frame_idx]['init_line'] = init_line
            sparse_bbox_data_list[instance_id][frame_idx]['gt_lines'] = gt_lines
            sparse_bbox_data_list[instance_id][frame_idx]['nearest_bbox'] = instance_bounding_box_list[instance_id][single_instance_pcd_frame_idx_list[nearest_i]]
            sparse_bbox_data_list[instance_id][frame_idx]['nearest_registered_idx'] = single_instance_pcd_frame_idx_list[nearest_i]
            sparse_bbox_data_list[instance_id][frame_idx]['nearest_i'] = nearest_i
            #o3d.visualization.draw_geometries([src, line_set, AXIS_PCD, instance_bounding_box_list[instance_id][single_instance_pcd_frame_idx_list[nearest_i]], gt_lines, init_line])
        #############################################################################

    if args.vis:
        visualizer(instance_bounding_box_list, t_bbox_list, sparse_bbox_list, unique_instance_id_list, registration_data_list, sparse_bbox_data_list, instance_frame_pcd_list, idx_range, args)

# tr_mash : open3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh.create_text("Hello", depth=0.1).to_legacy()
# tr_mash.paint_uniform_color([1, 0, 0])
# location = np.array([2, 1, 5])
# tr_mash.transform([[0.1, 0, 0, location[0]], [0, 0.1, 0, location[1]], [0, 0, 0.1, location[2]], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([tr_mash])


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
    parser.add_argument('--scene_idx', type=int,default=3)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=0)
    parser.add_argument('--rgs_end_idx',type=int, default=5)
    parser.add_argument('--origin',type=bool, default=False)
    parser.add_argument('--clustering',type=str, default='dbscan')
    parser.add_argument('--dbscan_each_instance', type=bool, default=False)
    parser.add_argument('--bbox_gen_fit_method', type=str, default='point_normal')

    parser.add_argument('--dbscan_max_cluster', type=bool, default=True)
    parser.add_argument('--id_merge_with_speed', type=bool, default=True)
    parser.add_argument('--position_diff_threshold', type=float, default=1.5)
    parser.add_argument('--speed_momentum', type=float, default=0.5)

    parser.add_argument('--registration_with_full_pc', type=bool, default=False)
    parser.add_argument('--z_threshold', type=float, default=0.3)

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