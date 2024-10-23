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

def fragmentized_full_registration(pcds, fragment_size, max_ptr_idx=0):
    transformation_matrices = []
    transformation_matrices.append(np.eye(4))
    for i in range(len(pcds) - 1):
        pose, _ = full_registration(pcds[i:i+2])
        transformation_matrices.append(transformation_matrices[-1] @ pose.nodes[1].pose)
        print(f"Fragmentized Registration: {i + 1}/{len(pcds) - 1}", end="\r")
    print()
    return transformation_matrices



def frame_pairwise_registration(source, target):
    dis = np.mean(np.linalg.norm(np.array(source.points), axis=1)) + np.mean(np.linalg.norm(np.array(target.points), axis=1))
    dis *= 2
    trans = np.mean(np.array(target.points), axis=0) - np.mean(np.array(source.points), axis=0)
    init_transformation = np.eye(4)
    init_transformation[:3, 3] = trans
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, dis, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, dis / 4, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp_fine.transformation



def full_pc_registration(full_pc_list):
    voxel_down_sampled_full_pc_list = []
    for full_pc in full_pc_list:
        voxel_down_sampled_full_pc = full_pc.voxel_down_sample(voxel_size=0.3)
        voxel_down_sampled_full_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        voxel_down_sampled_full_pc.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        voxel_down_sampled_full_pc_list.append(voxel_down_sampled_full_pc)
        print(f"Downsampled {len(voxel_down_sampled_full_pc.points)} points from {len(full_pc.points)}, frame {len(voxel_down_sampled_full_pc_list)}")
    tr_matrices = fragmentized_full_registration(voxel_down_sampled_full_pc_list, 1)
    # pose_graph = o3d.pipelines.registration.PoseGraph()
    # for i in range(len(tr_matrices)):
    #     pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(tr_matrices[i]))

    # prefix_matmul = []
    # prefix_matmul.append(tr_matrices[0])
    # for i in range(1, len(tr_matrices)):
    #     prefix_matmul.append(prefix_matmul[-1] @ tr_matrices[i])

    # for i in range(len(tr_matrices) - 1):
    #     print("                                                                        ", end="\r")
    #     for j in range(i + 1, len(tr_matrices)):
    #         print(f"Pairwise Registration: {i + 1}/{len(tr_matrices) - 1}, {j + 1}/{len(tr_matrices)}", end="\r")
    #         current_tr = np.dot(np.linalg.inv(prefix_matmul[i]), prefix_matmul[j])
    #         dis = np.mean(np.linalg.norm(np.array(voxel_down_sampled_full_pc_list[i].points), axis=1)) + np.mean(np.linalg.norm(np.array(voxel_down_sampled_full_pc_list[j].points), axis=1))
    #         dis /= 2
    #         dis = dis * (0.2 * np.pi / 180) * 6
    #         information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    #             voxel_down_sampled_full_pc_list[i], voxel_down_sampled_full_pc_list[j], dis,
    #             current_tr)
    #         if j == i + 1:
    #             pose_graph.edges.append(
    #                 o3d.pipelines.registration.PoseGraphEdge(i,
    #                                                          j,
    #                                                          current_tr,
    #                                                          information_icp,
    #                                                          uncertain=False))
    #         else:
    #             pose_graph.edges.append(
    #                 o3d.pipelines.registration.PoseGraphEdge(i,
    #                                                          j,
    #                                                          current_tr,
    #                                                          information_icp,
    #                                                          uncertain=True))
    # print()
    # option = o3d.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=dis,
    #     edge_prune_threshold=0.9,
    #     reference_node=0)
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     o3d.pipelines.registration.global_optimization(
    #         pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #         o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
    # tr_matrices = []
    # for node in pose_graph.nodes:
    #     tr_matrices.append(node.pose)
    return tr_matrices

# def full_pc_registration(full_pc_list):
#     voxel_down_sampled_full_pc_list = []
#     for full_pc in full_pc_list:
#         voxel_down_sampled_full_pc = full_pc.voxel_down_sample(voxel_size=0.3)
#         voxel_down_sampled_full_pc.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
#         voxel_down_sampled_full_pc.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
#         voxel_down_sampled_full_pc_list.append(voxel_down_sampled_full_pc)
#         print(f"Downsampled {len(voxel_down_sampled_full_pc.points)} points from {len(full_pc.points)}, frame {len(voxel_down_sampled_full_pc_list)}")
#     pose_graph, dis = full_registration(voxel_down_sampled_full_pc_list)
#     option = o3d.pipelines.registration.GlobalOptimizationOption(
#         max_correspondence_distance=dis,
#         edge_prune_threshold=0.9,
#         reference_node=0)
#     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#         o3d.pipelines.registration.global_optimization(
#             pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#             o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
#     tr_matrices = []
#     for node in pose_graph.nodes:
#         tr_matrices.append(node.pose)
#     return tr_matrices