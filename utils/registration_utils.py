import numpy as np
import open3d as o3d

from utils.utils import translate_boxes_to_open3d_instance2 as translate_boxes_to_open3d_instance
import copy
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, init_transformation):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance_coarse, icp_coarse.transformation))
    icp_fine = icp_coarse#o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance_fine,icp_coarse.transformation,o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    print(o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance_fine, icp_fine.transformation))
    
    if np.sum(init_transformation == np.eye(4)) != 16:
        print(o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance_coarse, np.eye(4)))
        print(o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance_coarse, init_transformation))
        print(o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance_fine, init_transformation))
        source = copy.deepcopy(source)
        target = copy.deepcopy(target)
        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])
        
        # make a bounding box size of max_correspondence_distance_fine, max_correspondence_distance_coarse
        gt_box = np.array(np.concatenate((np.mean(np.array(source.points), axis=0), np.array([max_correspondence_distance_fine, max_correspondence_distance_fine, max_correspondence_distance_fine, 0]))))
        gt_box2 = np.array(np.concatenate((np.mean(np.array(target.points), axis=0), np.array([max_correspondence_distance_coarse, max_correspondence_distance_coarse, max_correspondence_distance_coarse, 0]))))
        line_set1, box1 = translate_boxes_to_open3d_instance(gt_box)
        line_set2, box2 = translate_boxes_to_open3d_instance(gt_box2)
        #o3d.visualization.draw_geometries([copy.deepcopy(source).transform(icp_fine.transformation), target, line_set1, line_set2]) 
        #o3d.visualization.draw_geometries([copy.deepcopy(source).transform(icp_coarse.transformation), target, line_set1, line_set2])
        #o3d.visualization.draw_geometries([copy.deepcopy(source).transform(init_transformation), target, line_set1, line_set2])
        #o3d.visualization.draw_geometries([source, target], point_show_normal=True)
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp



def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, init_transformation_list):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            init_transformation = np.eye(4)
            if target_id == source_id + 1:
                init_transformation = init_transformation_list[target_id]
                print(f"registration between {source_id} and {target_id} with init_transformation")
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine, init_transformation)
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