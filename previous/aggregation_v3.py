import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
from utils.utils import dbscan, draw_point_and_3Dgt_bbox, draw_point_and_3Dpred_bbox, build_3d_pseudo_box, pairwise_registration, draw_registration_result



def main(args):
    
    
    
    ########## front image에서 sam 영역안에 들어가는 Point cloud만 ##########    
    source_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_sam',f'{str(args.src_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)
    target_with_instance_id = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_sam',f'{str(args.tgt_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4)

    source = source_with_instance_id[:, :3]
    target = target_with_instance_id[:, :3]

    source_id = source_with_instance_id[:, 3]
    target_id = target_with_instance_id[:, 3]

    num_ins_src = np.max(source_with_instance_id[:, 3]).astype(np.int16)
    num_ins_tgt = np.max(target_with_instance_id[:, 3]).astype(np.int16)

    src_color  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_sam',f'{str(args.src_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3] 
    tgt_color = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','visualization/uppc_color_sam',f'{str(args.tgt_frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3] 

    src_gt_bbox  = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.src_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 
    tgt_gt_bbox = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(args.tgt_frame_idx).zfill(6)}.bin')).reshape(-1, 7) 


    voxel_size = 0.1
    threshold = voxel_size * 20



    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(source[:, :3])
    # src.paint_uniform_color([1, 0.706, 0])
    src.colors = open3d.utility.Vector3dVector(src_color)
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if args.perform_db_scan:
        un_noise_idx = dbscan(src, source_id)
        source_id = source_id[un_noise_idx]
        masked_src_color = src_color[un_noise_idx]
        masked_source = source[un_noise_idx]
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(masked_source[:, :3])
        # src.paint_uniform_color([1, 0.706, 0])
        src.colors = open3d.utility.Vector3dVector(masked_src_color)

    if args.with_gt_box:
        draw_point_and_3Dgt_bbox(src, src_gt_bbox)

    if args.visible_bbox_estimation:
        orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, source_id, axis=args.axis_aligned, orient=args.orient, pca=args.pca, vis=args.vis)
        # axis_extended_bbox = build_extended_bbox(axis_bboxs)
        draw_point_and_3Dpred_bbox(src, orient_3dbbox, axis_bbox, src_gt_bbox, vis=args.vis)
    else:
        draw_point_and_3Dpred_bbox(src, vis=args.vis)


########################################### registration ########################################### 

    # TODO registration each 
    
    # voxel_size = 0.1
    # threshold = voxel_size * 20
    
    # src_instance_list = list()
    # src_instance_color_list = list()
    
    # tgt_instance_list = list()
    # tgt_instance_color_list = list()

    # for src_instance_idx in range(num_ins_src):
    #     src_instance_mask = np.where(source_with_instance_id[:,3] == (src_instance_idx+1))
    #     src_instance = source_with_instance_id[src_instance_mask]
    #     src_instance_color = src_color[src_instance_mask]
    #     src_instance_list.append(src_instance)
    #     src_instance_color_list.append(src_instance_color)
        

    # for tgt_instance_idx in range(num_ins_tgt):
    #     tgt_instance_mask = np.where(target_with_instance_id[:,3] == (tgt_instance_idx+1))
    #     tgt_instance = target_with_instance_id[tgt_instance_mask]
    #     tgt_instance_color = tgt_color[tgt_instance_mask]
    #     tgt_instance_list.append(tgt_instance)    
    #     tgt_instance_color_list.append(tgt_instance_color)
    
    
    # for src_instance, src_instance_color, tgt_instance, tgt_instance_color \
    #     in zip(src_instance_list, src_instance_color_list, tgt_instance_list, tgt_instance_color_list):
    #     src = open3d.geometry.PointCloud()
    #     src.points = open3d.utility.Vector3dVector(src_instance[:,:3])
    #     src.paint_uniform_color([1, 0.706, 0])
    #     # src.colors = open3d.utility.Vector3dVector(src_instance_color)
    #     src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
    #     tgt = open3d.geometry.PointCloud()
    #     tgt.points = open3d.utility.Vector3dVector(tgt_instance[:,:3])
    #     tgt.colors = open3d.utility.Vector3dVector(tgt_instance_color)
    #     tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                
    #     # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     # src, tgt, threshold,
    #     # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    #     # )
        
    #     transform_matrix, _ = pairwise_registration(src,tgt,threshold)
        
    #     src_down = src.voxel_down_sample(voxel_size)
    #     tgt_down = tgt.voxel_down_sample(voxel_size)
        
    #     transformed_src = src_down.transform(transform_matrix)
    #     transfromed_points = np.array(transformed_src.points)
        
        
    #     draw_registration_result(src_down, tgt_down, transformation=transform_matrix, src_bbox=src_gt_bbox, tgt_bbox=tgt_gt_bbox)
        
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
    parser.add_argument('--perform_db_scan', type=bool, default=True)
    parser.add_argument('--with_gt_box', type=bool, default=False)
    parser.add_argument('--axis_aligned', type=bool, default=True)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--orient', type=bool, default=True)
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--scene_idx', type=int,default=3)
    parser.add_argument('--src_frame_idx', type=int, default=0)
    parser.add_argument('--tgt_frame_idx', type=int, default=0)
    parser.add_argument('--reg_start_idx',type=int, default=0)
    parser.add_argument('--reg_end_idx',type=int, default=17)
    

    args = parser.parse_args()    
    main(args)        


    ############## original #####################
    # source = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/data_for_vis/rawpoint/000163.bin', dtype=np.float32).reshape(-1, 3)[:, :3] #original
    # src_gt_bbox  = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/annotations/'+f'{str(src_frame_idx).zfill(6)}.bin').reshape(-1, 7) 

    # src = open3d.geometry.PointCloud()
    # src.points = open3d.utility.Vector3dVector(source[:, :3])
    # src.paint_uniform_color([1, 0.706, 0])
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()

    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.zeros(3)


    # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    # vis.add_geometry(axis_pcd)
    # vis = draw_box(vis, src_gt_bbox, (1, 1, 1))
    # vis.add_geometry(src)
    # vis.run()
    # target = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/data_for_vis/rawpoint/000001.bin', dtype=np.float32).reshape(-1, 3)[:, :3] #original
    ####################################################



    



