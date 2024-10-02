import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan, get_obj,translate_boxes_to_open3d_instance, translate_boxes_to_open3d_gtbox, dbscan_max_cluster
from utils.registration_utils import full_registration
from utils.open3d_utils import set_black_background, set_white_background

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
CAR_CLASS_SIZE = [4.5, 1.9, 2.0] #l, w, h





def main(args):
    
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
        
    
    

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

        ####################### id merge ############################
        if args.id_merge_with_speed:
            for i in range(len(pcd_id)):
                while pcd_id[i] in same_id_dict.keys():
                    pcd_id[i] = same_id_dict[pcd_id[i]]
        ####################### id merge ############################
        
        ####################### dbscan before registration ############################
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(pcd)
        src.colors = open3d.utility.Vector3dVector(pcd_color)
        
        if args.vis:
            print(f"frame{frame_idx}'s point_cloud before dbscan")
            # o3d.visualization.draw_geometries_with_key_callbacks([src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })
        
        if args.perform_db_scan_before_registration:        
            un_noise_idx = dbscan(src, pcd_id, eps=0.8, min_points= 50)
            pcd_id = pcd_id[un_noise_idx]
            pcd_id_list.append(pcd_id)
            masked_pcd_color = pcd_color[un_noise_idx]
            masked_pcd = pcd[un_noise_idx]
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(masked_pcd)
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
            if args.vis:
                print(f"frame{frame_idx}'s point_cloud after dbscan")
                # o3d.visualization.draw_geometries_with_key_callbacks([src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })
            src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            src_list.append(src)
        
        else:    
            pcd_id_list.append(pcd_id)
            src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # src_down = src.voxel_down_sample(voxel_size)
            src_list.append(src)
        ####################### dbscan before registration ############################

        ####################### position estimation ############################
        if args.id_merge_with_speed:

            # estimate position based on speed and previous position
            new_estimated_position = dict()
            for i in estimated_position_list.keys():
                if new_speed_list.get(i) is not None:
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
                        if not j in pcd_id and np.linalg.norm(position - estimated_position_list[j]) < args.position_diff_threshold:
                            same_id_dict[instance_id] = j
                            pcd_id[mask] = j
                            instance_id = j
                            break
                # update lists
                if previous_position.get(instance_id) is not None:
                    if speed_list.get(instance_id) is not None:
                        new_speed_list[instance_id] = (position - previous_position[instance_id]) * args.speed_momentum + (1 - args.speed_momentum) * speed_list[instance_id]
                    else:
                        new_speed_list[instance_id] = position - previous_position[instance_id]
                new_previous_position[instance_id] = position
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
            previous_position = new_previous_position

    
    #print estimated corresponding position
    if args.id_merge_with_speed and args.vis:
        for i in same_id_dict.keys():
            print(f"instance {i} is merged with {same_id_dict[i]}")
        ####################### position estimation ############################
    
    
    
    
    ####################### registration ############################
    
    aggre_pcd_list = list()
    aggre_pcd_color_list = list()
    
    pose_graph = full_registration(src_list,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)
    
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
        
    print("Transform points and display")    
    
    for point_id in range(len(src_list)):
        print(pose_graph.nodes[point_id].pose)
        transformed_pcd = src_list[point_id].transform(pose_graph.nodes[point_id].pose)
        np_transformed_pcd = np.array(transformed_pcd.points)
        np_transformed_pcd_color = np.array(transformed_pcd.colors)
        aggre_pcd_list.append(np_transformed_pcd)
        aggre_pcd_color_list.append(np_transformed_pcd_color)
    
    src_list.append(AXIS_PCD)
    if args.vis:
        print("After Registration!")
        o3d.visualization.draw_geometries_with_key_callbacks(src_list, {ord("B"): set_black_background, ord("W"): set_white_background }) ########## registration 결과

    np_aggre_pcd_id = np.hstack(pcd_id_list).astype(np.int16)
    np_aggre_pcd = np.vstack(aggre_pcd_list)
    np_aggre_pcd_color = np.vstack(aggre_pcd_color_list)
    
    ####################### registration ############################
    
    
    
    
    
    ############ noise 제거하기 위해서 instance단위로 dbscan 각각 ##################
    
    
    if args.dbscan_each_instance:
        
        instance_idx_list = np.unique(np_aggre_pcd_id)
        pcd_list_after_dbscan = list()
        pcd_id_list_after_dbscan = list()
        pcd_color_list_after_dbscan = list()
        refiend_bbox_list = list()
        
        for idx_instance in instance_idx_list:
            instance_mask = np.where(np_aggre_pcd_id == (idx_instance))
            single_instance_pcd = np_aggre_pcd[instance_mask]
            
            
            if len(single_instance_pcd) < 300:
                continue
            single_instance_pcd_color = np_aggre_pcd_color[instance_mask]
            single_instance_pcd_id = np_aggre_pcd_id[instance_mask]
            
            single_instance_src = open3d.geometry.PointCloud()
            single_instance_src.points = open3d.utility.Vector3dVector(single_instance_pcd)
            single_instance_src.colors = open3d.utility.Vector3dVector(single_instance_pcd_color)
            
            
            if args.vis:
                print(f"instnace_id:{idx_instance} before dbscan")
                o3d.visualization.draw_geometries_with_key_callbacks([single_instance_src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })

            if args.dbscan_max_cluster:
                un_noise_idx = dbscan_max_cluster(single_instance_src, None, eps = 0.5, min_points=100)
            else:
                un_noise_idx = dbscan(single_instance_src, None, eps = 0.5, min_points=100)

            masked_pcd_id = single_instance_pcd_id[un_noise_idx]
            masked_pcd_color = single_instance_pcd_color[un_noise_idx]
            masked_pcd = single_instance_pcd[un_noise_idx]
            
            if args.vis:
                single_instance_after_db_scan_src = open3d.geometry.PointCloud()
                single_instance_after_db_scan_src.points = open3d.utility.Vector3dVector(masked_pcd)
                single_instance_after_db_scan_src.colors = open3d.utility.Vector3dVector(masked_pcd_color)                
                print(f"instnace_id:{idx_instance} after dbscan")
                o3d.visualization.draw_geometries_with_key_callbacks([single_instance_after_db_scan_src, AXIS_PCD],{ord("B"): set_black_background, ord("W"): set_white_background })
                print(f"masked_pcd_size:{len(masked_pcd)}")
            
            
            
            
            
            #############################  TODO boudning box generation lidar coordinate로 변환
            lidar_to_camera = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
            camera_coord_pcd_all = np_aggre_pcd @ lidar_to_camera.T
            camera_coord_pcd_instance = masked_pcd @ lidar_to_camera.T
            
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(camera_coord_pcd_instance[:, :3])
            # src.paint_uniform_color([1, 0.706, 0])
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
            
            
            if len(src.points) != 0: #dbscan 이후 point가 남아있을 경우만 bounding box 얻기
                obj = get_obj(np.array(src.points),camera_coord_pcd_all, args.bbox_gen_fit_method)
            else:
                continue
            ############ camera coords to lidar #########
            line_set, box3d = translate_boxes_to_open3d_instance(obj)
            line_from_lineset = np.array(line_set.lines)
            pt_from_line_set = np.array(line_set.points)
            
            
            src_lidar = open3d.geometry.PointCloud()
            src_lidar.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
            src_lidar.colors = open3d.utility.Vector3dVector(masked_pcd_color)
            pcd_list_after_dbscan.append(src_lidar)
            
            bbox_pt_lidar_coords = pt_from_line_set @ lidar_to_camera            
            lidar_lineset = o3d.geometry.LineSet()
            
            ############pseudo G를 통계값으로 얻은 bounindg box 크기로 키우기 #########################
            # initial_box_corner_points = o3d.utility.Vector3dVector(bbox_pt_lidar_coords)
            # initial_box = o3d.geometry.OrientedBoundingBox.create_from_points(initial_box_corner_points)
            # initial_box.color = [1,0,0]
            # r = initial_box.R
            # R_inv = np.linalg.inv(r)
            # extent = initial_box.extent # extent 순서 l, w, h     
            # scale_factor = np.array([CAR_CLASS_SIZE[0]/extent[0],CAR_CLASS_SIZE[1]/extent[1], CAR_CLASS_SIZE[2]/extent[2]])
            # # o3d.visualization.draw_geometries_with_key_callbacks([ initial_box, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### lidar coords

            # initial_box_copy = copy.deepcopy(initial_box)
            # obb_aligned = initial_box_copy.rotate(R_inv, center=initial_box.center)
            # obb_aligned.color = [0,1,0]
            # # o3d.visualization.draw_geometries_with_key_callbacks([ initial_box, obb_aligned, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### lidar coords
    
            
            # obb_aligned_corners = np.asarray(obb_aligned.get_box_points())
            # fixed_x = np.min(obb_aligned_corners[:, 0])
            # new_extent = extent * scale_factor
            
            # refined_obb_center = np.array(obb_aligned.center)
            # refined_obb_center[0] = fixed_x + new_extent[0]/ 2
            
            
            # box_scaled = o3d.geometry.OrientedBoundingBox(refined_obb_center, np.eye(3), new_extent)
            # box_scaled.color = [0, 0, 1]
            # # o3d.visualization.draw_geometries_with_key_callbacks([ initial_box, obb_aligned, box_scaled, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### lidar coords

            # box_scaled.rotate(r, box_scaled.center )
            # # # refined_bbox.rotate(r,center=refined_bbox.center )
            # # box_scaled.color = [1, 0, 0]
            # # o3d.visualization.draw_geometries_with_key_callbacks([ initial_box, obb_aligned, box_scaled, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### lidar coords

            ###############################################################################
            
            
            
            lidar_lineset.points = o3d.utility.Vector3dVector(bbox_pt_lidar_coords)
            lidar_lineset.lines = o3d.utility.Vector2iVector(line_from_lineset)
            pcd_list_after_dbscan.append(lidar_lineset)
            
            
            if args.vis:
                # o3d.visualization.draw_geometries_with_key_callbacks([src, line_set, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### camera coords
                print(f"bounding box for instance :{idx_instance}")
                # o3d.visualization.draw_geometries_with_key_callbacks([ src_lidar, lidar_lineset,box_scaled, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background }) ### lidar coords
            ############################################################

        if src_gt_bbox is not None:
            gt_box_line_set = list()
            for i in range(src_gt_bbox.shape[0]):
                line_set, box3d = translate_boxes_to_open3d_gtbox(src_gt_bbox[i])
                line_set.paint_uniform_color((0, 0, 1))
                gt_box_line_set.append(line_set)
            pcd_list_after_dbscan.extend(gt_box_line_set)
        
        print(f"Whole bounding box")
        pcd_list_after_dbscan.append(AXIS_PCD)        
        o3d.visualization.draw_geometries_with_key_callbacks(pcd_list_after_dbscan, {ord("B"): set_black_background, ord("W"): set_white_background })

 
 
 
 
 
        
    #     np_aggre_pcd = np.vstack(pcd_list_after_dbscan)
    #     np_aggre_pcd_color = np.vstack(pcd_color_list_after_dbscan)
    #     np_aggre_pcd_id = np.concatenate(pcd_id_list_after_dbscan)
        
        
    #     src = open3d.geometry.PointCloud()
    #     src.points = open3d.utility.Vector3dVector(np_aggre_pcd)
    #     src.colors = open3d.utility.Vector3dVector(np_aggre_pcd_color)
    


    #     orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, np_aggre_pcd_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
    #     draw_point_and_3Dpred_bbox_not_l_shaped(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
    

    # aggre_src = open3d.geometry.PointCloud()
    # aggre_src.points = open3d.utility.Vector3dVector(np_aggre_pcd)
    # aggre_src.colors = open3d.utility.Vector3dVector(np_aggre_pcd_color)
    
    # o3d.visualization.draw_geometries([aggre_src])
    
    
    # ###### dbscan ######
    
    # if args.clustering =='dbscan':
    
    #     un_noise_idx = dbscan(aggre_src, np_aggre_pcd_id, eps=0.2, min_points=150)
    #     masked_pcd_id = np_aggre_pcd_id[un_noise_idx]
    #     masked_pcd_color = np_aggre_pcd_color[un_noise_idx]
    #     masked_pcd = np_aggre_pcd[un_noise_idx]
    #     src = open3d.geometry.PointCloud()
    #     src.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
    #     # src.paint_uniform_color([1, 0.706, 0])
    #     src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
    
    # elif args.clustering =='hdbscan':
 
    #     un_noise_idx = hdbscan_idx(aggre_src, min_cluster_size=30,culster_selection_epsilion=0.4)
    #     masked_pcd_id = np_aggre_pcd_id[un_noise_idx]
    #     masked_pcd_color = np_aggre_pcd_color[un_noise_idx]
    #     masked_pcd = np_aggre_pcd[un_noise_idx]
    #     src = open3d.geometry.PointCloud()
    #     src.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
    #     # src.paint_uniform_color([1, 0.706, 0])
    #     src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
    
    # o3d.visualization.draw_geometries([src])
    
    # ######################
    
    

    
    # orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, masked_pcd_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
    # draw_point_and_3Dpred_bbox_not_l_shaped(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
    
    



if __name__ == "__main__":
    

    
    
    
    
    parser = argparse.ArgumentParser(description='pseudo bounding generation ')
    parser.add_argument('--dataset_path', type=str, default='/Users/injae/Desktop/code/OpenPCDet/3df_data/waymo_sam2')
    parser.add_argument('--visible_bbox_estimation', type=bool, default=True)
    parser.add_argument('--perform_db_scan_before_registration', type=bool, default=False)
    parser.add_argument('--with_gt_box', type=bool, default=False)
    parser.add_argument('--axis_aligned', type=bool, default=True)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--orient', type=bool, default=True)
    parser.add_argument('--vis', type=bool, default=False)
    parser.add_argument('--scene_idx', type=int,default=3)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=0)
    parser.add_argument('--rgs_end_idx',type=int, default=24)
    parser.add_argument('--origin',type=bool, default=False)
    parser.add_argument('--clustering',type=str, default='dbscan')
    parser.add_argument('--dbscan_each_instance', type=bool, default=True)
    parser.add_argument('--bbox_gen_fit_method', type=str, default='closeness_to_edge')

    parser.add_argument('--dbscan_max_cluster', type=bool, default=False)
    parser.add_argument('--id_merge_with_speed', type=bool, default=False)
    parser.add_argument('--position_diff_threshold', type=float, default=0.5)
    parser.add_argument('--speed_momentum', type=float, default=0.8)
    

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




    



