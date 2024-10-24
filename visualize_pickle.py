import os
import copy
import pickle
import argparse
import numpy as np
import open3d as o3d
from utils.save_utils import Save_PseudoGT
from utils.bounding_box_utils import BoundingBox
from utils.instance_merge_utils import merge_instance_ids
from utils.pickle_visualizer_utils import visualizer

CAM_LOCS = {1:'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}
CAM_NAMES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

def main(args):
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)

    pickle_path = os.path.join(args.data_path, f'scene-{args.scene_idx}/')
    data = Save_PseudoGT(pickle_path, 'pseudo_gt.pkl', save=False).load()
    id_merge_data = data['id_merge_data']
    world_transformation_matrices = data['world_transformation_matrices']
    registration_matrix = data['registration_matrix']
    bbox_obj_list = data['bbox_obj_list']
    static_bbox_obj_list = data['static_bbox_obj_list']

    pcd_list = []
    pcd_color_list = []
    pcd_id_list = []

    unique_instance_id_list = []
    for frame_idx in idx_range:
        pcd_with_instance_id = []
        pcd_color = []
        if args.multicam:
            for cam_name in CAM_NAMES:
                try:
                    pcd_with_instance_id.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', cam_name, 'visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4))
                except:
                    print(f"scene-{args.scene_idx} {cam_name} {frame_idx} is not found")
                    continue
                try:
                    pcd_color.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', cam_name, 'visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3])
                except:
                    print(f"scene-{args.scene_idx} {cam_name} {frame_idx} is not found")
                    continue
        else:
            try:
                pcd_with_instance_id.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', 'visualization/uppc_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4))
            except:
                print(f"scene-{args.scene_idx} {frame_idx} is not found")
                continue
            try:
                pcd_color.extend(np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}', 'visualization/uppc_color_continuous_sam',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)[:, :3])
            except:
                print(f"scene-{args.scene_idx} {frame_idx} is not found")
                continue

        pcd_with_instance_id = np.array(pcd_with_instance_id)
        pcd_color = np.array(pcd_color)

        if len(pcd_with_instance_id) == 0:
            pcd_list.append(np.array([]))
            pcd_color_list.append(np.array([]))
            pcd_id_list.append(np.array([]))
            continue

        pcd = pcd_with_instance_id[:, :3]
        pcd_id = pcd_with_instance_id[:, 3]

        pcd_list.append(pcd)
        pcd_color_list.append(pcd_color)
        pcd_id_list.append(pcd_id)

        unique_instance_id_list.append(np.unique(pcd_id))

    unique_instance_id_list = np.unique(np.concatenate(unique_instance_id_list)).astype(int)
    inv_world_transformation_matrices = [np.linalg.inv(tr) for tr in world_transformation_matrices]

    ########################## Load instance, frame pcds ########################
    instance_pcd_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_frame_pcd_list = [[{} for _ in range(np.max(idx_range) + 1)] for _ in range(np.max(unique_instance_id_list) + 1)]
    instance_pcd_color_list = {}
    for i, instance_id in enumerate(unique_instance_id_list):
        for j, frame_idx in enumerate(idx_range):
            instance_frame_pcd = pcd_list[j][pcd_id_list[j] == instance_id]
            instance_frame_pcd_color = pcd_color_list[j][pcd_id_list[j] == instance_id]

            if len(instance_frame_pcd) == 0:
                continue

            instance_frame_pcd_list[instance_id][frame_idx]["pcd"] = instance_frame_pcd
            instance_frame_pcd_list[instance_id][frame_idx]["color"] = instance_frame_pcd_color[0]

            instance_pcd_color_list[instance_id] = instance_frame_pcd_color[0]
            instance_pcd_list[instance_id][frame_idx] = instance_frame_pcd
    
    instance_pcd_list, instance_pcd_color_list, unique_instance_id_list = merge_instance_ids(instance_pcd_list, instance_pcd_color_list, unique_instance_id_list, id_merge_data)

    ########################## Load Bounding Box ########################
    instance_bounding_box_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    static_bbox_list = [{} for _ in range(np.max(unique_instance_id_list) + 1)]
    for instance_id in unique_instance_id_list:
        for frame_idx in idx_range:
            if frame_idx in bbox_obj_list[instance_id]:
                instance_bounding_box_list[instance_id][frame_idx] = BoundingBox().load_gt(bbox_obj_list[instance_id][frame_idx])
        if instance_id in static_bbox_obj_list:
            static_obj = BoundingBox().load_gt(static_bbox_obj_list[instance_id])
            for frame_idx in idx_range:
                static_bbox_list[instance_id][frame_idx] = copy.deepcopy(static_obj).transform(world_transformation_matrices[frame_idx])
    
    ########################## Visualize ########################
    visualizer(instance_pcd_list, instance_pcd_color_list, instance_frame_pcd_list, instance_bounding_box_list, static_bbox_list, unique_instance_id_list, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PseudoGT')
    parser.add_argument('--scene_idx', type=int, default=3)
    parser.add_argument('--rgs_start_idx',type=int, default=0)
    parser.add_argument('--rgs_end_idx',type=int, default=10)
    parser.add_argument('--data_path', type=str, default='/workspace/psuedogt/output')
    parser.add_argument('--dataset_path', type=str, default='/workspace/3df_data/waymo_sam2')
    parser.add_argument('--multicam', type=bool, default=False)
    parser.add_argument('--z_threshold', type=float, default=0.4)

    args = parser.parse_args()
    main(args)