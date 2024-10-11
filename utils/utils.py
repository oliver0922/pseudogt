import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
import hdbscan
import types
import math
 
 
 
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
    
 
 
def draw_point_and_3Dpred_bbox(pcd, orient_bbox=None, axis_bbox= None, gt_box=None, vis=False):
    
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
    
    # if vis:
    o3d.visualization.draw_geometries(pcd_with_box)
 
 
def dbscan(pcd, eps=0.2, min_points=10, print_progress=False, debug=False ):
 
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
 
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")
 
    un_noise_idx = np.where(labels != -1)[0]
  
    return un_noise_idx

def dbscan_max_cluster(pcd, eps=0.2, min_points=10, print_progress=False, debug=False ):
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
 
    # same with dbscan, but only returns label with maximum points
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")

    labels_new = labels[np.where(labels != -1)]
    if len(labels_new) == 0:
        return np.array([])
    max_cluster_idx = np.argmax(np.bincount(labels_new))
    un_noise_idx = np.where(labels == max_cluster_idx)[0]

    return un_noise_idx
 
 
def hdbscan_idx(pcd, min_cluster_size, culster_selection_epsilion):
    
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon= culster_selection_epsilion, gen_min_span_tree=True)
    cluster.fit(np.array(pcd.points))
    labels = cluster.labels_
    
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
 
    un_noise_idx = np.where(labels != -1)[0]
    
    
    
    return un_noise_idx
    
 
 
 
def build_3d_pseudo_box(pcd, instance_id, axis, orient, pca, vis):
    
    orient_bbx = []
    axis_bbx = []
    indices = pd.Series(range(len(instance_id))).groupby(instance_id, sort=False).apply(list).tolist()
    
    if orient:
        for i in range(len(indices)):
            sub_cloud = pcd.select_by_index(indices[i])
            if len(sub_cloud.points)<4:
                continue
            
            if pca:
                #### debug ######
                if vis:
                    colors = np.zeros((len(sub_cloud.points),3))
                    colors[:,0] = 1
                    sub_cloud.colors = open3d.utility.Vector3dVector(colors)
                    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                    vis_list = [sub_cloud, axis_pcd]
                    # o3d.visualization.draw_geometries(vis_list)
                ################
                
                
                points = np.asarray(sub_cloud.points)
                centroid = np.mean(points, axis=0)
                points_centered = points - centroid
 
                cov_matrix = np.cov(points_centered[:,:2].T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
 
                yaw_only_rotation = np.eye(3)
                yaw_only_rotation[:2, :2] = eigenvectors[:, :2]  
 
                rotated_points = points_centered @ yaw_only_rotation.T
                
                
                #visualiation eigen vector
                if vis:
                    arrows = []
                    xy_eigenvectors = eigenvectors
                    xy_eigenvalues = eigenvalues[:2]
                    
                    for i in range(len(xy_eigenvalues)):
                        eigenvector = xy_eigenvectors[:, i] * xy_eigenvalues[i]
                        arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.02,  # 화살표 몸통의 반지름
                            cone_radius=0.04,      # 화살표 끝부분의 반지름
                            cylinder_height=np.linalg.norm(eigenvector),  # 벡터 길이
                            cone_height=0.1,       # 화살표 끝부분 길이
                            resolution=20,         # 화살표 해상도
                            cylinder_split=4,      # 화살표의 원통 세분화
                            cone_split=1           # 화살표의 끝부분 세분화
                        )
                        
                        arrow.translate(centroid)
                        vec = np.array([eigenvector[0], eigenvector[1], 0])
                        vec = vec / np.linalg.norm(vec)
                        z_axis = np.array([0,0,1])
                        rotation_matrix = R.align_vectors([vec], [z_axis])[0].as_matrix()
                        arrow.rotate(rotation_matrix, centroid)
                        arrow.paint_uniform_color(np.array([0,0,0]))
                        
                        arrows.append(arrow)
                    vis_list.extend(arrows)
                    o3d.visualization.draw_geometries(vis_list)
                    
                    
                
                if vis:
                    rotated_src = open3d.geometry.PointCloud()
                    rotated_src.points = open3d.utility.Vector3dVector(rotated_points)
                    colors = np.zeros((len(sub_cloud.points),3))
                    colors[:,1] = 1
                    rotated_src.colors = open3d.utility.Vector3dVector(colors)
                    vis_list.append(rotated_src)
                    o3d.visualization.draw_geometries(vis_list)
                ################
                
 
                # Bounding Box의 최솟값과 최댓값을 계산합니다
                min_bound = np.min(rotated_points, axis=0)
                max_bound = np.max(rotated_points, axis=0)
 
                # 최종 바운딩 박스 좌표를 계산한 뒤 다시 원래 좌표계로 복귀
                obb_corners = np.array([
                    [min_bound[0], min_bound[1], min_bound[2]],
                    [max_bound[0], min_bound[1], min_bound[2]],
                    [max_bound[0], max_bound[1], min_bound[2]],
                    [min_bound[0], max_bound[1], min_bound[2]],
                    [min_bound[0], min_bound[1], max_bound[2]],
                    [max_bound[0], min_bound[1], max_bound[2]],
                    [max_bound[0], max_bound[1], max_bound[2]],
                    [min_bound[0], max_bound[1], max_bound[2]],
                ])
 
                # 바운딩 박스 좌표를 원래 좌표계로 변환
                obb_corners_world = (obb_corners @ yaw_only_rotation) + centroid
 
                # Open3D OrientedBoundingBox 생성
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obb_corners_world))
 
                # Bounding Box의 색상을 설정합니다 (선택 사항)
                obb.color = (1, 0, 0)  # 빨간색
                orient_bbx.append(obb)
 
            else:
                obb = sub_cloud.get_oriented_bounding_box()
                obb.color = (1, 0, 0)
                orient_bbx.append(obb)
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
 
 
def draw_registration_result(source, target, transformation=None, src_bbox=None, tgt_bbox=None):
    
    if transformation is not None:
        source.transform(transformation)
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
 
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3)
    
 
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    
    vis.add_geometry(source)
    vis.add_geometry(target)
    
    if src_bbox is not None:
        vis = draw_box(vis, src_bbox, (0, 0, 1))
        vis = draw_box(vis, tgt_bbox, (0, 1, 0))
    
    vis.run()
    vis.destroy_window()
 
 
def translate_boxes_to_open3d_instance2(gt_boxes):
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
 
 
def pairwise_registration(source, target, threshold):
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
 
 
def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:,0].min(), projection[:,0].max()
        min_y, max_y = projection[:,1].min(), projection[:,1].max()
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        beta = np.vstack((Dx, Dy)).min(axis=0)
        beta = np.maximum(beta, d0)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
 
    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
 
    area = (max_x - min_x) * (max_y - min_y)
 
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area
 
def get_lowest_point_rect(ptc, xz_center, l, w, ry):
    ptc_xz = ptc[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    ys = ptc[mask, 1]
    return ys.max()

def point_normal_rectangle(src):
    normals = np.array(src.normals)
    # remove normals with strong y
    new_src = o3d.geometry.PointCloud()
    new_src.points = o3d.utility.Vector3dVector(np.array(src.points)[np.where(np.abs(normals[:, 1]) < 0.5)])
    new_src.normals = o3d.utility.Vector3dVector(normals[np.where(np.abs(normals[:, 1]) < 0.5)])
    #o3d.visualization.draw_geometries([new_src], point_show_normal=True)
    normals = normals[np.abs(normals[:, 1]) < 0.5]
    # only 0, 2 axis
    normals = normals[:, [0, 2]]
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    # find the most frequent angle
    bins = np.arange(-np.pi, np.pi, np.pi/72)
    hist, _ = np.histogram(angles, bins=bins)
    angle = bins[np.argmax(hist)]
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = np.array(src.points)[:, [0, 2]] @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    if (max_x - min_x) < (max_y - min_y):
        angle += np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = np.array(src.points)[:, [0, 2]] @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def given_angle_bbox_fit(src, angle):
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = np.array(src.points)[:, [0, 2]] @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    if (max_x - min_x) < (max_y - min_y):
        angle += np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = np.array(src.points)[:, [0, 2]] @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area

 
def get_obj(ptc, fit_method, angle=None):
    if fit_method == 'min_zx_area_fit':
        corners, ry, area = minimum_bounding_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'PCA':
        corners, ry, area = PCA_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'variance_to_edge':
        corners, ry, area = variance_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'closeness_to_edge':
        corners, ry, area = closeness_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'point_normal':
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(ptc)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=50))
        src.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        corners, ry, area = point_normal_rectangle(src)
    elif fit_method == 'given_angle':
        if angle is None:
            raise ValueError('angle is None')
        corners, ry, area = given_angle_bbox_fit(ptc, angle)
    else:
        raise NotImplementedError(fit_method)
    ry *= -1
    l = np.linalg.norm(corners[0] - corners[1])
    w = np.linalg.norm(corners[0] - corners[-1])
    c = (corners[0] + corners[2]) / 2
    bottom = ptc[:, 1].max()
    # bottom = get_lowest_point_rect(full_ptc, c, l, w, ry)
    h = bottom - ptc[:, 1].min()
    obj = types.SimpleNamespace()
    obj.t = np.array([c[0], bottom, c[1]])
    obj.l = l
    obj.w = w
    obj.h = h
    obj.ry = ry
    obj.volume = area * h
    return obj
 
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
    center = copy.deepcopy(gt_boxes.t)
    center[1] = center[1]-gt_boxes.h/2
    lwh = [gt_boxes.l, gt_boxes.h, gt_boxes.w]
    axis_angles = np.array([0,  gt_boxes.ry + 1e-10 , 0])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
 
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
 
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
 
    line_set.lines = open3d.utility.Vector2iVector(lines)
 
    return line_set, box3d
 

def translate_boxes_to_lidar_coords(gt_boxes, angle, lidar_to_camera):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes.center
    center_lidar = lidar_to_camera.T @ center
    lwh = [gt_boxes.extent[0], gt_boxes.extent[2], gt_boxes.extent[1]]
    axis_angles = np.array([0, 0, np.pi/2 - angle + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center_lidar, rot, lwh)
 
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
 
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
 
    line_set.lines = open3d.utility.Vector2iVector(lines)
 
    return line_set, box3d


def translate_obj_to_open3d_instance(obj):
    center = obj.center
    center[2] = center[2]+obj.extent[2]/2
    lwh = obj.extent
    axis_angles = np.array([0, 0, obj.ry + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d
 
 
def translate_boxes_to_open3d_gtbox(gt_boxes):
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



def draw_point_and_3Dpred_bbox_not_l_shaped(pcd, orient_bbox=None, axis_bbox= None, gt_box=None, vis=False):
    
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    pcd_with_box = [pcd]
    
    if orient_bbox is not None:
        pcd_with_box.extend(orient_bbox)
        
    if axis_bbox is not None:
        pcd_with_box.extend(axis_bbox)
    
    if gt_box is not None:
        gt_box_line_set = list()
        for i in range(gt_box.shape[0]):
            line_set, box3d = translate_boxes_to_open3d_instance_not_lshaped(gt_box[i])
            line_set.paint_uniform_color((0, 0, 1))
            gt_box_line_set.append(line_set)
        pcd_with_box.extend(gt_box_line_set)
       
    pcd_with_box.append(axis_pcd)
    
    # if vis:
    o3d.visualization.draw_geometries(pcd_with_box)