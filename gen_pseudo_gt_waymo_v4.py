import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan, get_obj,translate_boxes_to_open3d_instance, translate_boxes_to_open3d_gtbox, dbscan_max_cluster, translate_boxes_to_lidar_coords
from utils.registration_utils import full_registration
from utils.open3d_utils import set_black_background, set_white_background

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
CAR_CLASS_SIZE = [4.5, 1.9, 2.0] #l, w, h





def main(args):
    
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15.
    max_correspondence_distance_fine = voxel_size * 1.5
        
    source_full_pc = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(args.rgs_start_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
    src_list = list()
    pcd_id_list = list()
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)
    src_gt_bbox  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.src_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 

    speed_list = dict()
    estimated_position_list = dict()
    previous_position = dict()
    same_id_dict = dict()
    appeared_id = list()
   
    for frame_idx in idx_range:
        pcd_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)
        pcd_color = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3]
        pcd = pcd_with_instance_id[:, :3]
        pcd_id = pcd_with_instance_id[:, 3]
        
        ####################### id merge with position estimation ############################
        if args.id_merge_with_speed:
            for i in range(len(pcd_id)):
                while pcd_id[i] in same_id_dict.keys():
                    pcd_id[i] = same_id_dict[pcd_id[i]]
        ####################### id merge with position estimation ############################



        ####################### dbscan before registration ############################
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(pcd)
        src.colors = open3d.utility.Vector3dVector(pcd_color)
        
        if args.vis:
            print(f"frame{frame_idx}'s point_cloud before dbscan")
            # o3d.visualization.draw_geometries_with_key_callbacks([src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })
        
        if args.perform_db_scan_before_registration:        
            un_noise_idx = dbscan(src, pcd_id, eps=0.3, min_points= 5)
            pcd_id = pcd_id[un_noise_idx]
            masked_pcd_color = pcd_color[un_noise_idx]
            masked_pcd = pcd[un_noise_idx]
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(masked_pcd)
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)

            pcd = masked_pcd
            pcd_color = masked_pcd_color
            if args.vis:
                print(f"frame{frame_idx}'s point_cloud after dbscan")
                # o3d.visualization.draw_geometries_with_key_callbacks([src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })
        ####################### dbscan before registration ############################



        ####################### id merge with position estimation ############################
        if args.id_merge_with_speed:

            # estimate position based on speed and previous position
            new_estimated_position = dict()
            for i in speed_list.keys():
                new_estimated_position[i] = estimated_position_list[i] + speed_list[i]
            estimated_position_list = new_estimated_position

            new_previous_position = dict()
            new_speed_list = dict()
            new_estimated_position_list = dict()

            for i in range(np.unique(pcd_id).shape[0]):
                instance_id = np.unique(pcd_id)[i]
                # get center position of the instance point cloud
                mask = np.where(pcd_id == instance_id)
                position = np.mean(pcd[mask], axis=0)
                # if any estimated position is close enough to the current position, merge the id
                if appeared_id.count(instance_id) == 0:
                    for j in estimated_position_list.keys():
                        print(f"distance between {instance_id} and {j} is {np.linalg.norm(position - estimated_position_list[j])}")
                        if not j in pcd_id and np.linalg.norm(position - estimated_position_list[j]) < args.position_diff_threshold:
                            same_id_dict[instance_id] = j
                            pcd_id[mask] = j
                            instance_id = j
                            break
                # update lists
                if previous_position.get(instance_id) is not None:
                    prev_frame, prev_pos = previous_position[instance_id]
                    if speed_list.get(instance_id) is not None:
                        new_speed_list[instance_id] = (position - prev_pos) / (frame_idx - prev_frame) * args.speed_momentum + (1 - args.speed_momentum) * speed_list[instance_id]
                    else:
                        new_speed_list[instance_id] = (position - prev_pos) / (frame_idx - prev_frame)
                new_previous_position[instance_id] = (frame_idx, position)
                new_estimated_position_list[instance_id] = position
                if not instance_id in appeared_id:
                    appeared_id.append(instance_id)

            tmp = new_estimated_position_list
            for i in estimated_position_list.keys():
                if tmp.get(i) is None:
                    tmp[i] = estimated_position_list[i]
            estimated_position_list = tmp

            tmp = new_speed_list
            for i in speed_list.keys():
                if tmp.get(i) is None:
                    tmp[i] = speed_list[i]
            speed_list = tmp
            
            tmp = new_previous_position
            for i in previous_position.keys():
                if tmp.get(i) is None:
                    tmp[i] = previous_position[i]
            previous_position = tmp

        for i in range(len(pcd_id)):
            while pcd_id[i] in same_id_dict.keys():
                pcd_id[i] = same_id_dict[pcd_id[i]]
        ####################### id merge with position estimation ############################

        pcd_id_list.append(pcd_id)
        src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        src_list.append(src)

    if args.id_merge_with_speed and args.vis:
        for i in same_id_dict.keys():
            print(f"instance {i} is merged with {same_id_dict[i]}")

    ############### bounding box generation ################
    np_aggre_pcd_id = np.hstack(pcd_id_list).astype(np.int16)
    instance_idx_list = np.unique(np_aggre_pcd_id)
    bounding_boxes = dict()
    
    for idx_instance in instance_idx_list:
        if idx_instance in [1, 3]:
            pass
            #continue
        instance_src_list = list()
        instance_frame_indices = list()
        center_list = list()
        for i, frame_idx in enumerate(idx_range):
            if (pcd_id_list[i] == idx_instance).sum() == 0:
                continue
            instance_frame_indices.append(frame_idx)
            single_frame_instace_pcd = np.array(src_list[i].points)[pcd_id_list[i] == idx_instance]
            center_list.append(np.mean(single_frame_instace_pcd, axis=0))

            single_frame_instace_pcd_color = np.array(src_list[i].colors)[pcd_id_list[i] == idx_instance]
            single_frame_instance_src = open3d.geometry.PointCloud()
            single_frame_instance_src.points = open3d.utility.Vector3dVector(single_frame_instace_pcd)
            single_frame_instance_src.colors = open3d.utility.Vector3dVector(single_frame_instace_pcd_color)
            single_frame_instance_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            # make normals to be oriented to the camera
            single_frame_instance_src.orient_normals_to_align_with_direction()
            # visualize estimated normals
            if args.vis:
                print(f"instance_id:{idx_instance} before registration, frame:{frame_idx}")
                # o3d.visualization.draw_geometries([single_frame_instance_src], point_show_normal=True)
            instance_src_list.append(single_frame_instance_src)

        # gen initial pose based on the center of the instance
        center_list = np.array(center_list)
        initial_transformation_list = list()
        initial_transformation_list.append(np.eye(4))
        for i in range(len(center_list) - 1):
            translation = center_list[i+1] - center_list[i]
            initial_transformation = np.eye(4)
            initial_transformation[:3, 3] = translation
            initial_transformation_list.append(copy.deepcopy(initial_transformation))

        transformation_matrix_list = list()
        pose_graph = full_registration(instance_src_list,
                            max_correspondence_distance_coarse,
                            max_correspondence_distance_fine,
                            initial_transformation_list)

        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        transformed_src_list = copy.deepcopy(instance_src_list)
        print("Transform points and display")
        for i in range(len(instance_src_list)):
            transformation_matrix_list.append(pose_graph.nodes[i].pose)
            print(pose_graph.nodes[i].pose)
            transformed_src_list[i].transform(pose_graph.nodes[i].pose)

            src1 = copy.deepcopy(instance_src_list[i]).transform(pose_graph.nodes[i].pose)
            src2 = copy.deepcopy(instance_src_list[i-1]).transform(pose_graph.nodes[i-1].pose)
            #o3d.visualization.draw_geometries_with_key_callbacks([src1, src2], {ord("B"): set_black_background, ord("W"): set_white_background })

        global_xyz_transformed_src_list = list()
        for point_id in range(len(instance_src_list)):
            transformed_pcd = copy.deepcopy(instance_src_list[point_id]).transform(transformation_matrix_list[point_id])
            np_transformed_pcd = np.array(transformed_pcd.points)
            np_transformed_pcd_color = np.array(transformed_pcd.colors)
            global_xyz_transformed_src_list.append(transformed_pcd)

        if args.vis:
            print(f"instance_id:{idx_instance} after registration")
            #o3d.visualization.draw_geometries_with_key_callbacks(global_xyz_transformed_src_list, {ord("B"): set_black_background, ord("W"): set_white_background })

        merged_global_xyz_transformed_src = open3d.geometry.PointCloud()
        merged_global_xyz_transformed_src.points = open3d.utility.Vector3dVector(np.vstack([np.array(src.points) for src in global_xyz_transformed_src_list]))
        merged_global_xyz_transformed_src.colors = open3d.utility.Vector3dVector(np.vstack([np.array(src.colors) for src in global_xyz_transformed_src_list]))

        ############ noise 제거하기 위해서 instance단위로 dbscan 각각 ##################
        if args.dbscan_each_instance:
            if len(merged_global_xyz_transformed_src.points) < 300:
                continue
            if args.dbscan_max_cluster:
                un_noise_idx = dbscan_max_cluster(merged_global_xyz_transformed_src, None, eps = 0.5, min_points=100)
            else:
                un_noise_idx = dbscan(merged_global_xyz_transformed_src, None, eps = 0.5, min_points=100)

            instance_points = np.array(merged_global_xyz_transformed_src.points)[un_noise_idx]
            instance_colors = np.array(merged_global_xyz_transformed_src.colors)[un_noise_idx]

        else:
            instance_points = np.array(merged_global_xyz_transformed_src.points)
            instance_colors = np.array(merged_global_xyz_transformed_src.colors)

        
        lidar_to_camera = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
        camera_coord_pcd_all = source_full_pc @ lidar_to_camera.T
        camera_coord_pcd_instance = instance_points @ lidar_to_camera.T
        
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(camera_coord_pcd_instance)
        src.colors = open3d.utility.Vector3dVector(instance_colors)
        
        if len(src.points) != 0: #dbscan 이후 point가 남아있을 경우만 bounding box 얻기
            obj = get_obj(np.array(src.points),camera_coord_pcd_all, args.bbox_gen_fit_method)
        else:
            continue
        ############ camera coords to lidar #########
        line_set, box3d = translate_boxes_to_open3d_instance(obj)
        line_from_lineset = np.array(line_set.lines)
        pt_from_line_set = np.array(line_set.points)
        origin_line_set_lidar, original_box3d_lidar = translate_boxes_to_lidar_coords(box3d, obj.ry, lidar_to_camera)
        
        if args.vis:
            print(f"bounding box for instance :{idx_instance}")
            src_lidar = open3d.geometry.PointCloud()
            src_lidar.points = open3d.utility.Vector3dVector(instance_points)
            src_lidar.colors = open3d.utility.Vector3dVector(instance_colors)
            o3d.visualization.draw_geometries_with_key_callbacks([src_lidar, original_box3d_lidar], {ord("B"): set_black_background, ord("W"): set_white_background })
        
        i = 0
        for frame_idx in idx_range:
            if frame_idx in instance_frame_indices:
                tr_matrix = transformation_matrix_list[i]
                line_set_lidar = copy.deepcopy(origin_line_set_lidar).transform(np.linalg.inv(tr_matrix))
                bounding_boxes[(idx_instance, frame_idx)] = copy.deepcopy(line_set_lidar)
                i += 1
        ############### bounding box generation ################

    if args.vis:
        for frame_idx in idx_range:
            load_list = list()
            for idx_instance in instance_idx_list:
                if (idx_instance, frame_idx) in bounding_boxes.keys():
                    line_set_lidar = bounding_boxes[(idx_instance, frame_idx)]
                    line_set_lidar.paint_uniform_color([1, 0, 0])
                    load_list.append(line_set_lidar)
            full_pc_xyz = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
            full_pc_xyz = full_pc_xyz[full_pc_xyz[:, 2] > args.z_threshold]
            color = np.ones_like(full_pc_xyz) * [0.5, 0.5, 0.5]
            full_pc = open3d.geometry.PointCloud()
            full_pc.points = open3d.utility.Vector3dVector(full_pc_xyz)
            full_pc.colors = open3d.utility.Vector3dVector(color)
            load_list.append(full_pc)
            load_list.append(AXIS_PCD)
            # load gt box
            gt_box = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(frame_idx).zfill(6)}.bin')).reshape(-1, 7)
            for i in range(gt_box.shape[0]):
                line_set_gt, box3d_gt = translate_boxes_to_open3d_gtbox(gt_box[i])
                line_set_gt.paint_uniform_color([0, 0, 1])
                load_list.append(line_set_gt)
            print(f"bounding box for frame :{frame_idx}")
            o3d.visualization.draw_geometries_with_key_callbacks(load_list, {ord("B"): set_black_background, ord("W"): set_white_background })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pseudo bounding generation ')
    parser.add_argument('--dataset_path', type=str, default='../3df_data/waymo_sam2')
    parser.add_argument('--visible_bbox_estimation', type=bool, default=True)
    parser.add_argument('--perform_db_scan_before_registration', type=bool, default=True)
    parser.add_argument('--with_gt_box', type=bool, default=False)
    parser.add_argument('--axis_aligned', type=bool, default=True)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--orient', type=bool, default=True)
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--scene_idx', type=int,default=17)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=80)
    parser.add_argument('--rgs_end_idx',type=int, default=180)
    parser.add_argument('--origin',type=bool, default=False)
    parser.add_argument('--clustering',type=str, default='dbscan')
    parser.add_argument('--dbscan_each_instance', type=bool, default=True)
    parser.add_argument('--bbox_gen_fit_method', type=str, default='closeness_to_edge')

    parser.add_argument('--dbscan_max_cluster', type=bool, default=False)
    parser.add_argument('--id_merge_with_speed', type=bool, default=True)
    parser.add_argument('--position_diff_threshold', type=float, default=1.0)
    parser.add_argument('--speed_momentum', type=float, default=0.5)
    
    parser.add_argument('--registration_with_full_pc', type=bool, default=True)
    parser.add_argument('--z_threshold', type=float, default=1.0)
    parser.add_argument('--registration_remove_instance', type=bool, default=True)

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
