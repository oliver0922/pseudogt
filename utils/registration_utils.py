import numpy as np
import open3d as o3d

from utils.utils import translate_boxes_to_open3d_instance2 as translate_boxes_to_open3d_instance
import copy

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    #source_down, source_fpfh = preprocess_point_cloud(source, max_correspondence_distance_coarse)
    #target_down, target_fpfh = preprocess_point_cloud(target, max_correspondence_distance_coarse)
    #result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, max_correspondence_distance_coarse*5)
    
    init_transformation = np.eye(4) 
    translate = np.mean(np.array(target.points), axis=0) - np.mean(np.array(source.points), axis=0)
    init_transformation[:3, 3] = translate

    icp_very_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse*5, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, icp_very_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #icp_fine = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_fine,icp_coarse.transformation,o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = icp_coarse
    # make a bounding box size of max_correspondence_distance_fine, max_correspondence_distance_coarse
    gt_box = np.array(np.concatenate((np.mean(np.array(source.points), axis=0), np.array([max_correspondence_distance_fine, max_correspondence_distance_fine, max_correspondence_distance_fine, 0]))))
    gt_box2 = np.array(np.concatenate((np.mean(np.array(target.points), axis=0), np.array([max_correspondence_distance_coarse, max_correspondence_distance_coarse, max_correspondence_distance_coarse, 0]))))
    line_set1, box1 = translate_boxes_to_open3d_instance(gt_box)
    line_set2, box2 = translate_boxes_to_open3d_instance(gt_box2)
    src1 = copy.deepcopy(source)
    src2 = copy.deepcopy(target)
    src1.paint_uniform_color([1, 0, 0])
    src2.paint_uniform_color([0, 1, 0])
    src1.transform(icp_fine.transformation)
    #o3d.visualization.draw_geometries([src1, src2, line_set1, line_set2, box1, box2])
    src1 = copy.deepcopy(source)
    src1.paint_uniform_color([1, 0, 0])
    src1.transform(icp_very_coarse.transformation)
    #o3d.visualization.draw_geometries([src1, target, line_set1, line_set2, box1, box2])
    src1 = copy.deepcopy(source)
    src1.paint_uniform_color([1, 0, 0])
    src1.transform(init_transformation)
    #o3d.visualization.draw_geometries([src1, target, line_set1, line_set2, box1, box2])


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
        print("                                                                            ", end="\r")
        print(f"Registration: {source_id}/{n_pcds}", end="\r")
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
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