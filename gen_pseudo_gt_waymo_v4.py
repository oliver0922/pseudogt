import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan as _dbscan, get_obj,translate_boxes_to_open3d_instance, translate_boxes_to_open3d_gtbox, dbscan_max_cluster as _dbscan_max_cluster, translate_boxes_to_lidar_coords
from utils.registration_utils import full_registration
from utils.open3d_utils import set_black_background, set_white_background
from utils.instance_merge_utils import id_merging, merge_instance_ids

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])


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



def gen_bbox(pcd, fit_method, only_angle=False):
    lidar_to_camera = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
    camera_coord_pcd = np.array(pcd.points) @ lidar_to_camera.T
    if len(pcd.points) == 0:
        return None, None
    obj = get_obj(camera_coord_pcd, fit_method)
    if only_angle:
        return obj.ry
    _, box3d = translate_boxes_to_open3d_instance(obj)
    return translate_boxes_to_lidar_coords(box3d, obj.ry, lidar_to_camera)



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



def get_valid_transformations(transformation_matrix, target_line_set_lidar):
    tr_matrix = transformation_matrix
    tr_matrix = np.linalg.inv(tr_matrix)
    new_tr_mat = np.eye(4)
    rotation = np.arctan2(tr_matrix[1, 0], tr_matrix[0, 0])
    new_tr_mat[:3, :3] = R.from_euler('z', rotation).as_matrix()
    new_tr_mat[:3, 3] = copy.deepcopy(target_line_set_lidar).transform(tr_matrix).get_center() - copy.deepcopy(target_line_set_lidar).transform(new_tr_mat).get_center()
    return new_tr_mat



def visualize_whole_frame(frame_idx, bbox_set, z_threshold):
    full_pc = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
    full_pc = full_pc[full_pc[:, 2] > z_threshold]
    gt_bbox = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(frame_idx).zfill(6)}.bin')).reshape(-1, 7)
    gt_list = []
    for i in range(len(gt_bbox)):
        line_gt, _ = translate_boxes_to_open3d_gtbox(gt_bbox[i])
        line_gt.paint_uniform_color([0, 0, 1])
        gt_list.append(line_gt)
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(full_pc)
    src.paint_uniform_color([0.706, 0.706, 0.706])
    print(f"frame {frame_idx} is visualized")
    o3d.visualization.draw_geometries_with_key_callbacks([src, AXIS_PCD] + bbox_set + gt_list, {ord("B"): set_black_background, ord("W"): set_white_background })



def extend_bbox(pcd, bbox_size):
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
    threshold = np.pi / 3
    count = np.sum(angle_diff < threshold, axis=0)
    face_done = np.logical_or(count > 30, count > max(3, len(normals) * 0.2))
    done_count = np.sum(face_done)
    if done_count == 0:
        pass
    if done_count == 1:
        face = np.argmax(face_done)
        angle = angle_list[face]
        pass
    if done_count == 2:
        if (face_done[0] and face_done[2]) or (face_done[1] and face_done[3]):
            print("error")
            pass

    if done_count == 3:
        face = np.argmin(face_done)
        angle = angle_list[face]
        pass
    pass




def main(args):
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)

    pcd_list = []
    pcd_color_list = []
    pcd_id_list = []

    unique_instance_id_list = []
    for frame_idx in idx_range:
        pcd_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)
        pcd_color = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3]
        pcd = pcd_with_instance_id[:, :3]
        pcd_id = pcd_with_instance_id[:, 3]

        pcd_list.append(pcd)
        pcd_color_list.append(pcd_color)
        pcd_id_list.append(pcd_id)

        unique_instance_id_list.append(np.unique(pcd_id))

    unique_instance_id_list = np.unique(np.concatenate(unique_instance_id_list)).astype(int)

    ########################## Load instance, frame pcds ########################
    instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_pcd_color_list = {}
    for i, instance_id in enumerate(unique_instance_id_list):
        for j, frame_idx in enumerate(idx_range):
            instance_frame_pcd = pcd_list[j][pcd_id_list[j] == instance_id]
            instance_frame_pcd_color = pcd_color_list[j][pcd_id_list[j] == instance_id]

            if len(instance_frame_pcd) == 0:
                continue

            instance_pcd_color_list[instance_id] = instance_frame_pcd_color[0]

            if args.perform_db_scan_before_registration:
                instance_frame_pcd = dbscan_per_frame_instance(instance_frame_pcd)

            if len(instance_frame_pcd) <= 70 or np.mean(np.linalg.norm(instance_frame_pcd, axis=1)) > 40.0:
                if len(instance_frame_pcd) == 0:
                    continue
                src = open3d.geometry.PointCloud()
                src.points = open3d.utility.Vector3dVector(instance_frame_pcd)
                src.paint_uniform_color(instance_frame_pcd_color[0])
                direction = extend_bbox(src, [2.0, 3.0, 1.8])
                continue

            instance_pcd_list[instance_id][frame_idx] = instance_frame_pcd
    #############################################################################

    if args.id_merge_with_speed:
        corr = id_merging(idx_range, instance_pcd_list, args.speed_momentum, args.position_diff_threshold)
        instance_pcd_list, instance_pcd_color_list, unique_instance_id_list = merge_instance_ids(instance_pcd_list, instance_pcd_color_list, unique_instance_id_list, corr)
        if args.vis:
            for i in corr.keys():
                print(f"instance {i} is merged with {corr[i]}")

    ########################## Registration and generate bbox ########################
    instance_bounding_box_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    t_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for instance_id in unique_instance_id_list:
        if instance_id != 22:
            continue
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
        
        transformation_matrices = []
        for i in range(len(single_instance_src_list)):
            transformation_matrices.append(np.linalg.inv(pose_graph.nodes[max_ptr_idx].pose) @ pose_graph.nodes[i].pose)
        
        registered_pcd = []
        for i, pcd in enumerate(single_instance_src_list):
            pcd.transform(transformation_matrices[i])
            registered_pcd.extend(np.array(pcd.points))
        registered_src = open3d.geometry.PointCloud()
        registered_src.points = open3d.utility.Vector3dVector(registered_pcd)
        registered_src.paint_uniform_color(instance_pcd_color_list[instance_id])
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

        if args.vis:
            gt_lines = find_gtbbox(registered_src, single_instance_pcd_frame_idx_list[max_ptr_idx])
            print(f"instance {instance_id} is generated")
            o3d.visualization.draw_geometries_with_key_callbacks([line_set_lidar, registered_src, gt_lines, t_line_set_lidar], {ord("B"): set_black_background, ord("W"): set_white_background })
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
        #############################################################################

    if args.vis:
        for frame_idx in idx_range:
            if frame_idx % 10 != 0:
                continue
            line_list = []
            for instance_id in unique_instance_id_list:
                if frame_idx in instance_bounding_box_list[instance_id].keys():
                    line_list.append(instance_bounding_box_list[instance_id][frame_idx])
                if frame_idx in t_bbox_list[instance_id].keys():
                    line_list.append(t_bbox_list[instance_id][frame_idx])
            visualize_whole_frame(frame_idx, line_list, args.z_threshold)



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
    parser.add_argument('--scene_idx', type=int,default=311)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=0)
    parser.add_argument('--rgs_end_idx',type=int, default=156)
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