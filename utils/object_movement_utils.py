import numpy as np
import open3d as o3d

from utils.registration_utils import fragmentized_full_registration, full_registration


def find_dynamic_objects(instance_pcd_list, unique_instance_id_list, idx_range, args):
    dynamic_instance_id_list = []
    static_instance_id_list = []
    for instance_id in unique_instance_id_list:
        center_list = []
        for frame_idx in idx_range:
            if frame_idx in instance_pcd_list[instance_id].keys():
                center_list.append(np.mean(instance_pcd_list[instance_id][frame_idx], axis=0))
        if len(center_list) < 2:
            static_instance_id_list.append(instance_id)
            continue
        diff_sum = 0
        max_diff = 0
        for i in range(1, len(center_list)):
            diff_sum += np.linalg.norm(center_list[i] - center_list[i - 1])
            if np.linalg.norm(center_list[i] - center_list[i - 1]) > max_diff:
                max_diff = np.linalg.norm(center_list[i] - center_list[i - 1])
        if diff_sum / len(center_list) > args.dynamic_threshold or max_diff > args.dynamic_threshold_single:
            dynamic_instance_id_list.append(instance_id)
        else:
            static_instance_id_list.append(instance_id)
    return dynamic_instance_id_list, static_instance_id_list

def dynamic_object_registration(instance_pcd_list, instance_id, idx_range, args):
    max_ptr_idx, max_ptr = 0, 0
    ptr_cnt = 0
    single_instance_pcd_list = []
    single_instance_pcd_frame_idx_list = []
    for frame_idx in idx_range:
        if frame_idx in instance_pcd_list.keys():
            single_instance_pcd_list.append(instance_pcd_list[frame_idx])
            single_instance_pcd_frame_idx_list.append(frame_idx)
            single_instance_pcd_list[-1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            single_instance_pcd_list[-1].orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            if len(instance_pcd_list[frame_idx].points) > max_ptr:
                max_ptr = len(instance_pcd_list[frame_idx].points)
                max_ptr_idx = ptr_cnt
            ptr_cnt += 1

    transformation_list = []
    if args.fragmentized_registration:
        transformation_list = fragmentized_full_registration(single_instance_pcd_list, args.fragment_size, max_ptr_idx)
    else:
        pose_graph, dis = full_registration(single_instance_pcd_list)
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=dis,
            edge_prune_threshold=0.9,
            reference_node=max_ptr_idx)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        transformation_list = [pose_graph.nodes[i].pose for i in range(len(pose_graph.nodes))]
    return transformation_list, max_ptr_idx