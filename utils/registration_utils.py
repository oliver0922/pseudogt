import numpy as np
import open3d as o3d

from utils.utils import translate_boxes_to_open3d_instance2 as translate_boxes_to_open3d_instance
import copy

def pairwise_registration(source, target, initial, near):
    #source_down, source_fpfh = preprocess_point_cloud(source, max_correspondence_distance_coarse)
    #target_down, target_fpfh = preprocess_point_cloud(target, max_correspondence_distance_coarse)
    #result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, max_correspondence_distance_coarse*5)
    
    dis = np.mean(np.linalg.norm(np.array(source.points), axis=1)) + np.mean(np.linalg.norm(np.array(target.points), axis=1))
    dis /= 2
    dis = dis * (0.2 * np.pi / 180) * 6
    if np.abs(np.sum(np.diag(initial)) - 4) < 1e-8:
        init_transformation = np.eye(4) 
        translate = np.mean(np.array(target.points), axis=0) - np.mean(np.array(source.points), axis=0)
        init_transformation[:3, 3] = translate
    else:
        init_transformation = initial
        #init_src = copy.deepcopy(source)
        #init_src.transform(init_transformation)
        #o3d.visualization.draw_geometries([init_src, target])

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, dis, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, dis / 6, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    roll = np.arctan2(icp_fine.transformation[2, 1], icp_fine.transformation[2, 2])
    pitch = np.arctan2(-icp_fine.transformation[2, 0], np.sqrt(icp_fine.transformation[2, 1] ** 2 + icp_fine.transformation[2, 2] ** 2))
    yaw = np.arctan2(icp_fine.transformation[1, 0], icp_fine.transformation[0, 0])
    error = o3d.pipelines.registration.evaluate_registration(source, target, dis/6, icp_fine.transformation)
    #print(roll, pitch)
    if np.abs(roll) > np.pi/12 or np.abs(pitch) > np.pi/12 or np.abs(yaw) > np.pi/6 or error.fitness < 0.3:
        #print("Repeating with point to point")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, dis, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, dis/6, icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # make a bounding box size of max_correspondence_distance_fine, max_correspondence_distance_coarse
    if np.linalg.norm(icp_fine.transformation - init_transformation) < 1e-5 and near:
        print("Failed")
    if near:
        gt_box = np.array(np.concatenate((np.mean(np.array(source.points), axis=0), np.array([dis, dis, dis, 0]))))
        gt_box2 = np.array(np.concatenate((np.mean(np.array(target.points), axis=0), np.array([dis/6, dis/6, dis/6, 0]))))
        line_set1, box1 = translate_boxes_to_open3d_instance(gt_box)
        line_set2, box2 = translate_boxes_to_open3d_instance(gt_box2)
        src1 = copy.deepcopy(source)
        src2 = copy.deepcopy(target)
        src1.paint_uniform_color([1, 0, 0])
        src2.paint_uniform_color([0, 1, 0])
        src1.transform(icp_fine.transformation)
        print(icp_fine.transformation)
        print(o3d.pipelines.registration.evaluate_registration(src1, target, dis/6))
        o3d.visualization.draw_geometries([src1, src2, line_set1, line_set2, box1, box2])
        src1 = copy.deepcopy(source)
        src1.paint_uniform_color([1, 0, 0])
        src1.transform(init_transformation)
        o3d.visualization.draw_geometries([src1, target, line_set1, line_set2, box1, box2])


    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, dis/6,
        icp_fine.transformation)
    return transformation_icp, information_icp, dis/6



def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    mean_dis = 0
    inv_odo_list = []
    inv_odo_list.append(np.linalg.inv(odometry))
    odo_list = []
    odo_list.append(odometry)
    for target_id in range(n_pcds):
        if n_pcds > 2:
            print(f"Registration: {target_id + 1}/{n_pcds}", end="\r")
        for source_id in range(target_id - 1, -1, -1):
            if target_id == source_id + 1:
                initial_transformation = np.identity(4)
            else:
                initial_transformation = np.dot(odo_list[target_id], inv_odo_list[source_id])
            transformation_icp, information_icp, dis = pairwise_registration(
                pcds[source_id], pcds[target_id], initial_transformation, False and source_id + 1 == target_id)
            mean_dis += dis
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                inv_odo_list.append(np.linalg.inv(odometry))
                odo_list.append(odometry)
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
    if n_pcds > 1:
        mean_dis /= n_pcds * (n_pcds - 1) / 2
    else :
        mean_dis = mean_dis if mean_dis > 0 else 0.1
    if n_pcds > 2:
        print()
    return pose_graph, mean_dis

def fragmetized_full_registration(pcds, fragment_size, max_ptr_idx=0):
    transformation_matrices = []
    transformation_matrices.append(np.eye(4))
    for i in range(len(pcds) - 1):
        pose, _ = full_registration(pcds[i:i+2])
        transformation_matrices.append(transformation_matrices[-1] @ pose.nodes[1].pose)
    return transformation_matrices
