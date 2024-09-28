import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd

box_colormap = [
    [1, 1, 1], # 
    [0, 1, 0], #car Green  
    [0, 1, 1], #truck Cyan
    [1, 1, 0.88], #construction vehicle Navy_blue
    [1, 0.5, 0.5], #bus Light Red
    [1, 0, 0], # trailer Red
    [1, 0, 1], #barrier Magenta
    [0.5, 0.5, 1], #motorcycle Light Blue````
    [0, 0.5, 0], #bicycle Dark Green
    [0.5, 0, 0], #pedestrian Dark Red
    [0, 0, 0.5], #traffic_cone Dark Blue
    [0.5, 0.5, 0.5], #objectness Gray
    ###############################
    [0.5, 0, 0.5], # openset car Purple
    [1, 1, 0], # openset truck Light Yellow 
    [0, 1, 0], #openset construction_vehicle Green  
    [0, 1, 1], #openset bus Cyan
    [1, 1, 0.88], #openset trailer Navy_blue
    [1, 0.5, 0.5], #openset barrier Light Red
    [1, 0, 0], # openset motorcycle Red
    [1, 0, 1], #openset bicycle Magenta
    [0.5, 0.5, 1], #openset pedestrian Light Blue
    [0, 0.5, 0], #openset traffic_cone  Dark Green    
    
]

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_point_and_3Dgt_bbox(source, bbox):
    
    
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3)
    

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    
    vis.add_geometry(source)
    
    vis = draw_box(vis, bbox, (0, 0, 1))
    
    vis.run()
    vis.destroy_window()
    


def draw_point_and_3Dpred_bbox(pcd, orient_bbox=None, axis_bbox= None, gt_box=None):
    
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    pcd_with_box = [pcd]
    
    if orient_bbox is not None:
        pcd_with_box.extend(orient_bbox)
        
    if axis_bbox is not None:
        pcd_with_box.extend(axis_bbox)
    
    if gt_box is not None:
        gt_box_line_set = list()
        for i in range(gt_box.shape[0]):
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box[i])
            line_set.paint_uniform_color((0, 0, 1))
            gt_box_line_set.append(line_set)
        pcd_with_box.extend(gt_box_line_set)
       
    pcd_with_box.append(axis_pcd)
    
    o3d.visualization.draw_geometries(pcd_with_box)


def dbscan(pcd, instance_id, eps=0.5, min_points=10, print_progress=False, debug=False ):

    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    un_noise_idx = np.where(labels != -1)[0]
    
    
    
    return un_noise_idx


def build_3d_pseudo_box(pcd, instance_id, axis, orient):
    
    orient_bbx = []
    axis_bbx = []
    indices = pd.Series(range(len(instance_id))).groupby(instance_id, sort=False).apply(list).tolist()
    
    if orient:
        for i in range(len(indices)):
            sub_cloud = pcd.select_by_index(indices[i])
            if len(sub_cloud.points)<4:
                continue
            

            
            
            
            
            ####### original #######
            obb = sub_cloud.get_oriented_bounding_box()
            # # OBB의 rotation matrix를 가져옵니다
            # R = obb.R

            # # Yaw rotation만 남기고 pitch와 roll을 제거하기 위해 Yaw 축을 추출합니다
            # # Yaw를 Z축에 대해 회전시키는 방식으로 처리
            # yaw_only_R = np.eye(3)
            # yaw_only_R[0, 0] = R[0, 0]
            # yaw_only_R[0, 1] = R[0, 1]
            # yaw_only_R[1, 0] = R[1, 0]
            # yaw_only_R[1, 1] = R[1, 1]

            # # Pitch와 Roll을 제거한 Rotation matrix로 OBB를 업데이트합니다
            # obb.R = yaw_only_R
            obb.color = (1, 0, 0)   
            orient_bbx.append(obb)
            ########################
            
    if axis:
        for i in range(len(indices)):
            sub_cloud = pcd.select_by_index(indices[i])
            if len(sub_cloud.points)<4:
                continue
            obb = sub_cloud.get_axis_aligned_bounding_box()
            obb.color = (0, 1, 0)
            axis_bbx.append(obb)        
        
    
    
        
    print(f"Number of Oriented_Bounding Boxes calculated {len(orient_bbx)}")
    print(f"Number of Axis_Aligned_Bounding Boxes calculated {len(axis_bbx)}")
    
    return orient_bbx, axis_bbx

def build_extended_bbox(axis_bboxes):
    
    axis_bbox = axis_bboxes[0]
    eight_points = axis_bbox.get_box_points()
    length = eight_points[1] - eight_points[0]
    eight_points.__setitem__(0, np.array([48.21191406+4.8,  2.88964844,  0.20977783]))
    eight_points.__setitem__(2, np.array([48.21191406+4.8,  4.65820312,  0.20977783]))
    eight_points.__setitem__(3, np.array([48.21191406+4.8,  2.88964844,  1.69036865]))
    eight_points.__setitem__(5, np.array([48.21191406+4.8,  4.65820312,  1.69036865]))
    
    eight_points = np.array([[48.21191406+4.8,  2.88964844,  0.20977783],
                             [48.21191406+4.8,  4.65820312,  0.20977783],
                             [48.21191406+4.8,  2.88964844,  1.69036865],
                             [48.21191406+4.8,  4.65820312,  1.69036865],
                             [48.21191406  ,  2.88964844,  0.20977783],
                             [48.21191406  ,  4.65820312,  1.69036865],
                             [48.21191406  ,  2.88964844,  1.69036865],
                             [48.21191406  ,  4.65820312,  0.20977783]
                             ])
    
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(eight_points[:, :3])
    new_bbox = src.get_axis_aligned_bounding_box()
    new_bbox.color = (0, 0, 0)
    axis_bboxes.append(new_bbox)
    
    # if length < 5:
        
        
    
    
    return axis_bboxes


def draw_registration_result(source, target, transformation=None, src_bbox=None, tgt_bbox=None):
    
    if transformation is not None:
        source.transform(transformation)
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    
    vis.add_geometry(source)
    vis.add_geometry(target)
    
    if src_bbox is not None:
        vis = draw_box(vis, src_bbox, (0, 0, 1))
        vis = draw_box(vis, tgt_bbox, (0, 1, 0))
    
    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold,
        icp_fine.transformation)
    return transformation_icp, information_icp




########################################################################################################################

visible_bbox_estimation = True 
perform_db_scan = True
with_gt_box = False
AXIS_ALIGNED = True
ORIENT = True
src_frame_idx = 0
tgt_frame_idx = 1

trans_init = np.asarray([[1,0,0,1],
                        [0,1,0,0],
                        [0,0,1,0], [0.0, 0.0, 0.0, 1.0]])



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


########## front image에서 sam 영역안에 들어가는 Point cloud만 ##########
source_with_instance_id = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/visualization/uppc_sam/'+f'{str(src_frame_idx).zfill(6)}.bin', dtype=np.float32).reshape(-1, 4)
target_with_instance_id = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/visualization/uppc_sam/'+f'{str(tgt_frame_idx).zfill(6)}.bin', dtype=np.float32).reshape(-1, 4)

source = source_with_instance_id[:, :3]
target = target_with_instance_id[:, :3]

source_id = source_with_instance_id[:, 3]
target_id = target_with_instance_id[:, 3]

num_ins_src = np.max(source_with_instance_id[:, 3]).astype(np.int16)
num_ins_tgt = np.max(target_with_instance_id[:, 3]).astype(np.int16)

src_color  = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/visualization/uppc_color_sam/'+f'{str(src_frame_idx).zfill(6)}.bin', dtype=np.float32).reshape(-1, 3)[:, :3] 
tgt_color = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/visualization/uppc_color_sam/'+f'{str(tgt_frame_idx).zfill(6)}.bin', dtype=np.float32).reshape(-1, 3)[:, :3] 

src_gt_bbox  = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/annotations/'+f'{str(src_frame_idx).zfill(6)}.bin').reshape(-1, 7) 
tgt_gt_bbox = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/visualization/scene-3/annotations/'+f'{str(tgt_frame_idx).zfill(6)}.bin').reshape(-1, 7)



# TODO registration each 
# src_instance_list = list()
# tgt_instance_list = list()

# for src_instance_idx in num_ins_src:
#     src_instance_mask = np.where(source_with_instance_id[:,3] == (src_instance_idx+1))
#     src_instance = source_with_instance_id[src_instance_mask]
#     src_instance_list.append(src_instance)

# for tgt_instance_idx in num_ins_tgt:
#     tgt_instance_mask = np.where(target_with_instance_id[:,3] == (tgt_instance_idx+1))
#     tgt_instance = target_with_instance_id[tgt_instance_mask]
#     tgt_instance_list.append(tgt_instance)    
# TODO    



voxel_size = 0.1
threshold = voxel_size * 20



src = open3d.geometry.PointCloud()
src.points = open3d.utility.Vector3dVector(source[:, :3])
# src.paint_uniform_color([1, 0.706, 0])
src.colors = open3d.utility.Vector3dVector(src_color)
src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

if perform_db_scan:
    un_noise_idx = dbscan(src, source_id)
    source_id = source_id[un_noise_idx]
    masked_src_color = src_color[un_noise_idx]
    masked_source = source[un_noise_idx]
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(masked_source[:, :3])
    # src.paint_uniform_color([1, 0.706, 0])
    src.colors = open3d.utility.Vector3dVector(masked_src_color)
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

if with_gt_box:
    draw_point_and_3Dgt_bbox(src, src_gt_bbox)

if visible_bbox_estimation:
    orient_3dbbox, axis_bbox = build_3d_pseudo_box(src, source_id, axis=AXIS_ALIGNED, orient=ORIENT)
    axis_extended_bbox = build_extended_bbox(axis_bbox)
    draw_point_and_3Dpred_bbox(src, orient_bbox=None, axis_bbox=axis_bbox, gt_box=src_gt_bbox )
else:
    draw_point_and_3Dpred_bbox(src)



############################# registration ##############################################


tgt = open3d.geometry.PointCloud()
tgt.points = open3d.utility.Vector3dVector(target[:, :3])
tgt.paint_uniform_color([0, 0.651, 0.929])
tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# tgt.colors = open3d.utility.Vector3dVector(tgt_color)


# reg_p2p = o3d.pipelines.registration.registration_icp(
#     src, tgt, threshold,trans_init,
#     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
#     )

transform, _ = pairwise_registration(src,tgt)


src_down = src.voxel_down_sample(voxel_size)
tgt_down = tgt.voxel_down_sample(voxel_size)



# draw_registration_result(src, tgt, transformation=reg_p2p.transformation, src_bbox=src_bbox, tgt_bbox=tgt_bbox)
# draw_registration_result(src_down, tgt_down, transformation=transform, src_bbox=src_gt_bbox, tgt_bbox=tgt_gt_bbox)


# print("Initial alignment")
# evaluation = o3d.pipelines.registration.evaluate_registration(
#     source, target, threshold, trans_init)
# print(evaluation)

#######################################################################


