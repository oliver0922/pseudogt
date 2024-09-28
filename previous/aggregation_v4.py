import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan, draw_point_and_3Dgt_bbox, draw_point_and_3Dpred_bbox, build_3d_pseudo_box, draw_registration_result, hdbscan_idx
from utils.utils import get_obj,translate_boxes_to_open3d_instance,draw_point_and_3Dpred_bbox_not_l_shaped

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp



def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph



def set_black_background(vis):
    opt = vis.get_render_option()
    opt.background_color = [0, 0, 0]  # 배경을 검은색으로 설정 (RGB값)
    return False

def set_white_background(vis):
    opt = vis.get_render_option()
    opt.background_color = [1, 1, 1]  # 배경을 검은색으로 설정 (RGB값)
    return False


def main(args):
    
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
        
    
    
    aggre_pcd_list = list()
    aggre_pcd_color_list = list()
    src_list = list()
    pcd_id_list = list()
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)
    src_gt_bbox  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.src_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 
   
    for frame_idx in idx_range:
        pcd_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)
        pcd_color = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3]
        pcd = pcd_with_instance_id[:, :3]
        pcd_id = pcd_with_instance_id[:, 3]
        pcd_id_list.append(pcd_id)
        
        
        
        ###### dbscan #####
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(pcd)
        src.colors = open3d.utility.Vector3dVector(pcd_color)        
        un_noise_idx = dbscan(src, pcd_id)
        pcd_id = pcd_id[un_noise_idx]
        masked_pcd_color = pcd_color[un_noise_idx]
        masked_pcd = pcd[un_noise_idx]
        
        src = open3d.geometry.PointCloud()
        if args.perform_db_scan:
            src.points = open3d.utility.Vector3dVector(masked_pcd)
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
        else:
            src.points = open3d.utility.Vector3dVector(pcd)
            src.colors = open3d.utility.Vector3dVector(pcd_color)
        
        src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # src_down = src.voxel_down_sample(voxel_size)
        
        
        ###### 각 sample visaulization#######
        if args.vis:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries_with_key_callbacks([src, axis_pcd],{ord("B"): set_black_background, ord("W"): set_white_background })
        src_list.append(src)
    
    
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
    
    o3d.visualization.draw_geometries(src_list) ########## registration 결과

    np_aggre_pcd_id = np.hstack(pcd_id_list).astype(np.int16)
    np_aggre_pcd = np.vstack(aggre_pcd_list)
    np_aggre_pcd_color = np.vstack(aggre_pcd_color_list)
    
    
    
    ############ noise 제거하기 위해서 instance단위로 dbscan 각각 ##################
    
    
    if args.dbscan_each:
        num_instance = np.max(np_aggre_pcd_id).astype(np.int16)
        pcd_list_after_dbscan = list()
        pcd_id_list_after_dbscan = list()
        pcd_color_list_after_dbscan = list()
        
        for idx_instance in range(num_instance):
            instance_mask = np.where(np_aggre_pcd_id == (idx_instance+1))
            single_instance_pcd = np_aggre_pcd[instance_mask]
            
            
            if len(single_instance_pcd) < 300:
                continue
            single_instance_pcd_color = np_aggre_pcd_color[instance_mask]
            single_instance_pcd_id = np_aggre_pcd_id[instance_mask]
            
            single_instance_src = open3d.geometry.PointCloud()
            single_instance_src.points = open3d.utility.Vector3dVector(single_instance_pcd)
            single_instance_src.colors = open3d.utility.Vector3dVector(single_instance_pcd_color)
            
            
            if args.vis:
                o3d.visualization.draw_geometries([single_instance_src])

            
            un_noise_idx = dbscan(single_instance_src, None, eps=0.4, min_points=150)
            masked_pcd_id = single_instance_pcd_id[un_noise_idx]
            masked_pcd_color = single_instance_pcd_color[un_noise_idx]
            masked_pcd = single_instance_pcd[un_noise_idx]
            
            
            #############################  TODO boudning box lidar coordinate로 변환
            lidar_to_camera = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])
            camera_coord_pcd_all = np_aggre_pcd @ lidar_to_camera.T
            camera_coord_pcd_instance = masked_pcd @ lidar_to_camera.T
            
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(camera_coord_pcd_instance[:, :3])
            # src.paint_uniform_color([1, 0.706, 0])
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
            
            obj = get_obj(np.array(src.points),camera_coord_pcd_all, args.bbox_gen_fit_method)
            
            line_set, box3d = translate_boxes_to_open3d_instance(obj)
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            if args.vis:
                o3d.visualization.draw_geometries([src,line_set,axis_pcd])
            ############################################################
            
            
            
            
            
            
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
            # src.paint_uniform_color([1, 0.706, 0])
            src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
            
            # obj = get_obj(np.array(src.points),np_aggre_pcd, args.bbox_gen_fit_method)
            
            # line_set, box3d = translate_boxes_to_open3d_instance(obj)
            # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
            # if args.vis:
            #     o3d.visualization.draw_geometries([src,line_set,axis_pcd])
            
            
            
            np_src = np.array(src.points)
            np_src_color = np.array(src.colors)
            pcd_list_after_dbscan.append(np_src)
            pcd_id_list_after_dbscan.append(masked_pcd_id)
            pcd_color_list_after_dbscan.append(np_src_color)

        o3d.visualization.draw_geometries([src])
    ###########################################################
    
    
        np_aggre_pcd = np.vstack(pcd_list_after_dbscan)
        np_aggre_pcd_color = np.vstack(pcd_color_list_after_dbscan)
        np_aggre_pcd_id = np.concatenate(pcd_id_list_after_dbscan)
        
        
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(np_aggre_pcd)
        src.colors = open3d.utility.Vector3dVector(np_aggre_pcd_color)
    


        orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, np_aggre_pcd_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
        draw_point_and_3Dpred_bbox_not_l_shaped(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
    

    aggre_src = open3d.geometry.PointCloud()
    aggre_src.points = open3d.utility.Vector3dVector(np_aggre_pcd)
    aggre_src.colors = open3d.utility.Vector3dVector(np_aggre_pcd_color)
    
    o3d.visualization.draw_geometries([aggre_src])
    
    
    ###### dbscan ######
    
    if args.clustering =='dbscan':
    
        un_noise_idx = dbscan(aggre_src, np_aggre_pcd_id, eps=0.2, min_points=150)
        masked_pcd_id = np_aggre_pcd_id[un_noise_idx]
        masked_pcd_color = np_aggre_pcd_color[un_noise_idx]
        masked_pcd = np_aggre_pcd[un_noise_idx]
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
        # src.paint_uniform_color([1, 0.706, 0])
        src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
    
    elif args.clustering =='hdbscan':
 
        un_noise_idx = hdbscan_idx(aggre_src, min_cluster_size=30,culster_selection_epsilion=0.4)
        masked_pcd_id = np_aggre_pcd_id[un_noise_idx]
        masked_pcd_color = np_aggre_pcd_color[un_noise_idx]
        masked_pcd = np_aggre_pcd[un_noise_idx]
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(masked_pcd[:, :3])
        # src.paint_uniform_color([1, 0.706, 0])
        src.colors = open3d.utility.Vector3dVector(masked_pcd_color)
    
    o3d.visualization.draw_geometries([src])
    
    ######################
    
    
    
    
    
    
    
    
    
    orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, masked_pcd_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
    draw_point_and_3Dpred_bbox_not_l_shaped(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
    
    
    # for src in src_list:
        
        
    
    # o3d.visualization.draw_geometries(src_list)
    
    # pcd_combined = o3d.geometry.PointCloud()
    # for point_id in range(len(src_list)):
    #     src_list[point_id].transform(pose_graph.nodes[point_id].pose)
    #     pcd_combined += src_list[point_id]
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    # o3d.visualization.draw_geometries([pcd_combined_down])
    # print("1")
    
    
    
    # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    # src_list.append(axis_pcd)
    # o3d.visualization.draw_geometries(src_list)
        
        
        
        
        ######## instance 하나만 우선 실험 ############# 
        # num_ins_pcd = np.max(pcd_id).astype(np.int16)
        # pcd_instance_list = list()
        # pcd_instance_color_list = list()
        
        # for pcd_instance_idx in range(num_ins_pcd):
        #     pcd_instance_mask = np.where(pcd_id == (pcd_instance_idx+1))
        #     pcd_instance = pcd_with_instance_id[pcd_instance_mask]
        #     pcd_instance_color = pcd_color[pcd_instance_mask]
        #     pcd_instance_list.append(pcd_instance)
        #     pcd_instance_color_list.append(pcd_instance_color)
        
        
        
        # for pcd_instance, pcd_instance_color in zip(pcd_instance_list, pcd_instance_color_list):
                
                
        #     pcd = open3d.geometry.PointCloud()
        #     pcd.points = open3d.utility.Vector3dVector(pcd_instance[:,:3])
        
        
        # o3d.visualization.draw_geometries(vis_list)
        # pcd_list.append(pcd)
    
    
        
    
    
    
    
    
    ########## front image에서 sam 영역안에 들어가는 Point cloud만 ##########    
#     source_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_sam',f'{str(args.src_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)
#     target_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_sam',f'{str(args.tgt_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)

#     source = source_with_instance_id[:, :3]
#     target = target_with_instance_id[:, :3]

#     source_id = source_with_instance_id[:, 3]
#     target_id = target_with_instance_id[:, 3]

#     num_ins_src = np.max(source_with_instance_id[:, 3]).astype(np.int16)
#     num_ins_tgt = np.max(target_with_instance_id[:, 3]).astype(np.int16)

#     src_color  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_sam',f'{str(args.src_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3] 
#     tgt_color = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_sam',f'{str(args.tgt_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3] 

#     src_gt_bbox  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.src_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 
#     tgt_gt_bbox = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.tgt_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 


#     voxel_size = 0.1
#     threshold = voxel_size * 20



#     src = open3d.geometry.PointCloud()
#     src.points = open3d.utility.Vector3dVector(source[:, :3])
#     # src.paint_uniform_color([1, 0.706, 0])
#     src.colors = open3d.utility.Vector3dVector(src_color)
#     src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#     if args.perform_db_scan:
#         un_noise_idx = dbscan(src, source_id)
#         source_id = source_id[un_noise_idx]
#         masked_src_color = src_color[un_noise_idx]
#         masked_source = source[un_noise_idx]
#         src = open3d.geometry.PointCloud()
#         src.points = open3d.utility.Vector3dVector(masked_source[:, :3])
#         # src.paint_uniform_color([1, 0.706, 0])
#         src.colors = open3d.utility.Vector3dVector(masked_src_color)

#     if args.with_gt_box:
#         draw_point_and_3Dgt_bbox(src, src_gt_bbox)

#     if args.visible_bbox_estimation:
#         orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, source_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
#         # axis_extended_bbox = build_extended_bbox(axis_bboxs)
#         draw_point_and_3Dpred_bbox(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
#     else:
#         draw_point_and_3Dpred_bbox(src, vis=args.vis)


# ########################################### registration ########################################### 

#     # TODO registration each 
    
#     voxel_size = 0.1
#     threshold = voxel_size * 20
    
#     src_instance_list = list()
#     src_instance_color_list = list()
    
#     tgt_instance_list = list()
#     tgt_instance_color_list = list()

#     for src_instance_idx in range(num_ins_src):
#         src_instance_mask = np.where(source_with_instance_id[:,3] == (src_instance_idx+1))
#         src_instance = source_with_instance_id[src_instance_mask]
#         src_instance_color = src_color[src_instance_mask]
#         src_instance_list.append(src_instance)
#         src_instance_color_list.append(src_instance_color)
        

#     for tgt_instance_idx in range(num_ins_tgt):
#         tgt_instance_mask = np.where(target_with_instance_id[:,3] == (tgt_instance_idx+1))
#         tgt_instance = target_with_instance_id[tgt_instance_mask]
#         tgt_instance_color = tgt_color[tgt_instance_mask]
#         tgt_instance_list.append(tgt_instance)    
#         tgt_instance_color_list.append(tgt_instance_color)
    
    
#     for src_instance, src_instance_color, tgt_instance, tgt_instance_color \
#         in zip(src_instance_list, src_instance_color_list, tgt_instance_list, tgt_instance_color_list):
#         src = open3d.geometry.PointCloud()
#         src.points = open3d.utility.Vector3dVector(src_instance[:,:3])
#         src.paint_uniform_color([1, 0.706, 0])
#         # src.colors = open3d.utility.Vector3dVector(src_instance_color)
#         src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
#         tgt = open3d.geometry.PointCloud()
#         tgt.points = open3d.utility.Vector3dVector(tgt_instance[:,:3])
#         tgt.colors = open3d.utility.Vector3dVector(tgt_instance_color)
#         tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                
#         # reg_p2p = o3d.pipelines.registration.registration_icp(
#         # src, tgt, threshold,
#         # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
#         # )
        
#         transform_matrix, _ = pairwise_registration(src,tgt,threshold)
        
#         src_down = src.voxel_down_sample(voxel_size)
#         tgt_down = tgt.voxel_down_sample(voxel_size)
        
#         transformed_src = src_down.transform(transform_matrix)
#         transfromed_points = np.array(transformed_src.points)
        
        
#         draw_registration_result(src_down, tgt_down, transformation=transform_matrix, src_bbox=src_gt_bbox, tgt_bbox=tgt_gt_bbox)
        
    # TODO    


    # tgt = open3d.geometry.PointCloud()
    # tgt.points = open3d.utility.Vector3dVector(target[:, :3])
    # tgt.paint_uniform_color([0, 0.651, 0.929])
    # tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # tgt.colors = open3d.utility.Vector3dVector(tgt_color)


    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     src, tgt, threshold,trans_init,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    #     )

    # transform, _ = pairwise_registration(src,tgt)


    # src_down = src.voxel_down_sample(voxel_size)
    # tgt_down = tgt.voxel_down_sample(voxel_size)



    # draw_registration_result(src, tgt, transformation=reg_p2p.transformation, src_bbox=src_bbox, tgt_bbox=tgt_bbox)
    # draw_registration_result(src_down, tgt_down, transformation=transform, src_bbox=src_gt_bbox, tgt_bbox=tgt_gt_bbox)


    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target, threshold, trans_init)
    # print(evaluation)


########################################################################################################################



if __name__ == "__main__":
    

    
    
    
    
    parser = argparse.ArgumentParser(description='pseudo bounding generation ')
    parser.add_argument('--dataset_path', type=str, default='/Users/injae/Desktop/code/OpenPCDet/3df_data/waymo_sam2')
    parser.add_argument('--visible_bbox_estimation', type=bool, default=True)
    parser.add_argument('--perform_db_scan', type=bool, default=False)
    parser.add_argument('--with_gt_box', type=bool, default=False)
    parser.add_argument('--axis_aligned', type=bool, default=True)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--orient', type=bool, default=True)
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--scene_idx', type=int,default=3)
    parser.add_argument('--src_frame_idx', type=int, default=11)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--rgs_start_idx',type=int, default=11)
    parser.add_argument('--rgs_end_idx',type=int, default=24)
    parser.add_argument('--origin',type=bool, default=False)
    parser.add_argument('--clustering',type=str, default='dbscan')
    parser.add_argument('--dbscan_each', type=bool, default=True)
    parser.add_argument('--bbox_gen_fit_method', type=str, default='closeness_to_edge')
    

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


        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
        vis.add_geometry(src)
        vis.run()    
    
    main(args)        




    



