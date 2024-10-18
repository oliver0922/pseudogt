import open3d as o3d
import open3d
import numpy as np
import os
import argparse
from utils.utils import translate_boxes_to_open3d_gtbox
from utils.open3d_utils import set_black_background, set_white_background


AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

def visualizer(instance_bounding_box_list, t_bbox_list, sparse_bbox_list, unique_instance_id_list, registration_data_list, sparse_bbox_data_list, instance_frame_pcd_list, idx_range, args):
    # show frame 0 full point cloud as example
    full_pc = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud','000000.bin'), dtype=np.float32).reshape(-1, 3)
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(full_pc)
    src.paint_uniform_color([0.706, 0.706, 0.706])
    o3d.visualization.draw_geometries([src, AXIS_PCD])
    
    while True:
        menu_option = input("1. Visualize specific frame\n2. Visualize registered instance\n3. Visualize sparse instance by frame\n4. Visualize instance by frame\n5. Exit\n")
        if menu_option == "1":
            frame_idx = int(input("Enter frame index: "))
            if not frame_idx in idx_range:
                print("Invalid frame index")
                continue
            line_list = []
            for instance_id in unique_instance_id_list:
                if frame_idx in instance_bounding_box_list[instance_id].keys():
                    line_list.append(instance_bounding_box_list[instance_id][frame_idx])
                if frame_idx in t_bbox_list[instance_id].keys():
                    line_list.append(t_bbox_list[instance_id][frame_idx])
                if frame_idx in sparse_bbox_list[instance_id].keys():
                    line_list.append(sparse_bbox_list[instance_id][frame_idx])
            visualize_whole_frame(instance_frame_pcd_list, unique_instance_id_list, frame_idx, line_list, args)

        elif menu_option == "2":
            instance_id = int(input("Enter instance id: "))
            if not instance_id in unique_instance_id_list:
                print("Invalid instance id")
                continue
            src = registration_data_list[instance_id]['registered_src']
            line_set_lidar = registration_data_list[instance_id]['line_set_lidar']
            t_line_set_lidar = registration_data_list[instance_id]['t_line_set_lidar']
            gt_lines = registration_data_list[instance_id]['gt_lines']
            print(f"Visualizing instance {instance_id}")
            o3d.visualization.draw_geometries_with_key_callbacks([src, line_set_lidar, t_line_set_lidar, gt_lines], {ord("B"): set_black_background, ord("W"): set_white_background })

        elif menu_option == "3":
            instance_id = int(input("Enter instance id: "))
            if not instance_id in unique_instance_id_list:
                print("Invalid instance id")
                continue

            valid_frame_idx = []
            for frame_idx in idx_range:
                if 'src' in sparse_bbox_data_list[instance_id][frame_idx].keys():
                    valid_frame_idx.append(frame_idx)
            print(f"Valid frame indices: {valid_frame_idx}")

            frame_idx = int(input("Enter frame index: "))
            if not frame_idx in idx_range:
                print("Invalid frame index")
                continue
            if not 'src' in sparse_bbox_data_list[instance_id][frame_idx].keys():
                print("Invalid frame index")
                continue

            src = sparse_bbox_data_list[instance_id][frame_idx]['src']
            line_set = sparse_bbox_data_list[instance_id][frame_idx]['line_set']
            init_line = sparse_bbox_data_list[instance_id][frame_idx]['init_line']
            gt_lines = sparse_bbox_data_list[instance_id][frame_idx]['gt_lines']
            nearest_bbox = sparse_bbox_data_list[instance_id][frame_idx]['nearest_bbox']
            nearest_registered_idx = sparse_bbox_data_list[instance_id][frame_idx]['nearest_registered_idx']
            print(f"Visualizing instance {instance_id} at frame {frame_idx}, with nearest registered bbox index {nearest_registered_idx}")
            o3d.visualization.draw_geometries_with_key_callbacks([src, line_set, init_line, gt_lines, nearest_bbox, AXIS_PCD], {ord("B"): set_black_background, ord("W"): set_white_background })

        elif menu_option == "4":
            idx = int(input("Enter instance id: "))
            frame_idx = int(input("Enter frame index: "))
            if not frame_idx in idx_range:
                print("Invalid frame index")
                continue

            if not "before_dbscan" in instance_frame_pcd_list[idx][frame_idx].keys():
                print("Invalid instance id or frame index")
                continue

            src1 = o3d.geometry.PointCloud()
            src1.points = o3d.utility.Vector3dVector(instance_frame_pcd_list[idx][frame_idx]["before_dbscan"])
            src1.paint_uniform_color(instance_frame_pcd_list[idx][frame_idx]["color"])
            print(f"Visualizing instance {idx} at frame {frame_idx}, before dbscan")
            o3d.visualization.draw_geometries([src1, AXIS_PCD])

            if not "after_dbscan" in instance_frame_pcd_list[idx][frame_idx].keys():
                continue

            src2 = o3d.geometry.PointCloud()
            src2.points = o3d.utility.Vector3dVector(instance_frame_pcd_list[idx][frame_idx]["after_dbscan"])
            src2.paint_uniform_color(instance_frame_pcd_list[idx][frame_idx]["color"])
            print(f"Visualizing instance {idx} at frame {frame_idx}, after dbscan")
            o3d.visualization.draw_geometries([src2, AXIS_PCD])

        elif menu_option == "5":
            break

        else:
            print("Invalid option")

def visualize_whole_frame(instance_frame_pcd_list, unique_instance_id_list, frame_idx, bbox_set, args):
    full_pc = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','pointcloud',f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
    full_pc = full_pc[full_pc[:, 2] > args.z_threshold]
    gt_bbox = np.fromfile(os.path.join(args.dataset_path,f'scene-{args.scene_idx}','annotations',f'{str(frame_idx).zfill(6)}.bin')).reshape(-1, 7)
    gt_list = []
    for i in range(len(gt_bbox)):
        line_gt, _ = translate_boxes_to_open3d_gtbox(gt_bbox[i])
        line_gt.paint_uniform_color([0, 0, 1])
        gt_list.append(line_gt)
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(full_pc)
    src.paint_uniform_color([0.706, 0.706, 0.706])

    src_list = []
    id_list = []
    for i in range(np.max(unique_instance_id_list) + 1):
        if "after_dbscan" in instance_frame_pcd_list[i][frame_idx].keys():
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(instance_frame_pcd_list[i][frame_idx]["after_dbscan"])
            src.paint_uniform_color(instance_frame_pcd_list[i][frame_idx]["color"])
            src_list.append(src)
            id_text : open3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh.create_text(str(i), depth= 0.1).to_legacy()
            id_text.paint_uniform_color(instance_frame_pcd_list[i][frame_idx]["color"])
            location = np.mean(instance_frame_pcd_list[i][frame_idx]["after_dbscan"], axis=0) + np.array([0, 0, 1.0])
            id_text.transform([[0.1, 0, 0, location[0]], [0, 0.1, 0, location[1]], [0, 0, 0.1, location[2]], [0, 0, 0, 1]])
            id_list.append(id_text)
            print(i)
    print(f"frame {frame_idx} is visualized")
    o3d.visualization.draw_geometries_with_key_callbacks([AXIS_PCD] + bbox_set + gt_list + src_list + id_list, {ord("B"): set_black_background, ord("W"): set_white_background })

