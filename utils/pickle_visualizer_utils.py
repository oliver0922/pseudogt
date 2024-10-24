import open3d as o3d
import open3d
import numpy as np
import os
import argparse
import copy
from utils.open3d_utils import set_black_background, set_white_background
from utils.bounding_box_utils import BoundingBox


AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

class Frame_Visualizer:
    def __init__(self, frame_idx, instance_frame_pcd_list, unique_instance_id_list, \
                 instance_bounding_box_list, \
                    static_bbox_list, args):
        self.frame_idx = frame_idx
        self.instance_frame_pcd_list = instance_frame_pcd_list
        self.unique_instance_id_list = unique_instance_id_list
        self.instance_bounding_box_list = instance_bounding_box_list
        self.static_bbox_list = static_bbox_list
        self.args = args

    def load_data(self):
        load_list = []
        for instance_id in self.unique_instance_id_list:
            if self.frame_idx in self.instance_bounding_box_list[instance_id].keys():
                dynamic_bbox = self.instance_bounding_box_list[instance_id][self.frame_idx].get_o3d_instance()[0]
                dynamic_bbox.paint_uniform_color(np.array([1, 0, 0]))
                load_list.append(dynamic_bbox)
            if self.frame_idx in self.static_bbox_list[instance_id].keys():
                static_bbox = self.static_bbox_list[instance_id][self.frame_idx].get_o3d_instance()[0]
                static_bbox.paint_uniform_color(np.array([0, 1, 0]))
                load_list.append(static_bbox)
        
        full_pc = np.fromfile(os.path.join(self.args.dataset_path,f'scene-{self.args.scene_idx}','pointcloud',f'{str(self.frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)

        full_pc = full_pc[full_pc[:, 2] > self.args.z_threshold]
        gt_bbox = np.fromfile(os.path.join(self.args.dataset_path,f'scene-{self.args.scene_idx}','annotations',f'{str(self.frame_idx).zfill(6)}.bin')).reshape(-1, 7)
        gt_list = []
        for i in range(len(gt_bbox)):
            line_gt, _ = BoundingBox().load_gt(gt_bbox[i]).get_o3d_instance()
            line_gt.paint_uniform_color([0, 0, 1])
            gt_list.append(line_gt)
        src = open3d.geometry.PointCloud()
        src.points = open3d.utility.Vector3dVector(full_pc)
        src.paint_uniform_color([0.706, 0.706, 0.706])

        src_list = []
        id_list = []
        
        for i in range(np.max(self.unique_instance_id_list) + 1):
            if 'pcd' in self.instance_frame_pcd_list[i][self.frame_idx].keys():
                src = open3d.geometry.PointCloud()
                src.points = open3d.utility.Vector3dVector(self.instance_frame_pcd_list[i][self.frame_idx]["pcd"])
                src.paint_uniform_color(self.instance_frame_pcd_list[i][self.frame_idx]["color"])
                src_list.append(src)
                id_text : open3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh.create_text(str(i), depth=3.0).to_legacy()
                id_text.paint_uniform_color(self.instance_frame_pcd_list[i][self.frame_idx]["color"])
                location = np.mean(self.instance_frame_pcd_list[i][self.frame_idx]["pcd"], axis=0) + np.array([0, 0, 1.0])
                id_text.transform([[0.1, 0, 0, location[0]], [0, 0.1, 0, location[1]], [0, 0, 0.1, location[2]], [0, 0, 0, 1]])
                id_list.append(id_text)
        
        return [AXIS_PCD] + load_list + gt_list + src_list + id_list
    
    def visualize(self):
        print(f"Visualizing frame {self.frame_idx}")
        o3d.visualization.draw_geometries_with_key_callbacks(self.load_data(), {ord("B"): set_black_background, ord("W"): set_white_background,
            ord("A"): self.vis_prev_frame(), ord("D"): self.vis_next_frame()})
        
    def vis_next_frame(self):
        def vis_next_frame(vis):
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            camera_position = copy.deepcopy(params.extrinsic)

            self.frame_idx = min(self.frame_idx + 1, self.args.rgs_end_idx)
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)

            vis.update_renderer()
            intrinsics = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.intrinsic = intrinsics
            cam_params.extrinsic = camera_position
            ctr.convert_from_pinhole_camera_parameters(cam_params, True)
            print(f"frame_idx: {self.frame_idx}")
        return vis_next_frame
    
    def vis_prev_frame(self):
        def vis_prev_frame(vis):
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            camera_position = copy.deepcopy(params.extrinsic)

            self.frame_idx = max(self.frame_idx - 1, self.args.rgs_start_idx)
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)

            vis.update_renderer()
            intrinsics = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.intrinsic = intrinsics
            cam_params.extrinsic = camera_position
            ctr.convert_from_pinhole_camera_parameters(cam_params, True)
            print(f"frame_idx: {self.frame_idx}")
        return vis_prev_frame
    
class Instance_Visualizer:
    def __init__(self, instance_frame_pcd_list, instance_id, frame_idx, valid_frame_idx):
        self.instance_id = instance_id
        self.frame_idx = frame_idx
        self.instance_frame_pcd_list = instance_frame_pcd_list
        self.valid_frame_idx = valid_frame_idx
        self.show_before_dbscan = True
    
    def load_data(self):
        src = open3d.geometry.PointCloud()
        if self.show_before_dbscan:
            if "before_dbscan" in self.instance_frame_pcd_list[self.instance_id][self.frame_idx].keys():
                src.points = open3d.utility.Vector3dVector(self.instance_frame_pcd_list[self.instance_id][self.frame_idx]["before_dbscan"])
                src.paint_uniform_color(self.instance_frame_pcd_list[self.instance_id][self.frame_idx]["color"])
        else:
            if "after_dbscan" in self.instance_frame_pcd_list[self.instance_id][self.frame_idx].keys():
                src.points = open3d.utility.Vector3dVector(self.instance_frame_pcd_list[self.instance_id][self.frame_idx]["after_dbscan"])
                src.paint_uniform_color(self.instance_frame_pcd_list[self.instance_id][self.frame_idx]["color"])
        return [AXIS_PCD, src]
    
    def visualize(self):
        print(f"Visualizing instance {self.instance_id}, frame {self.frame_idx}")
        o3d.visualization.draw_geometries_with_key_callbacks(self.load_data(), 
            {ord("B"): set_black_background, ord("W"): set_white_background,
             ord("A"): self.vis_prev_frame(), ord("D"): self.vis_next_frame(),
             ord("S"): self.toggle_show_before_dbscan()})
    
    def vis_next_frame(self):
        def vis_next_frame(vis):
            # find next frame idx
            next_frame_idx_idx = np.argmin(np.abs(np.array(self.valid_frame_idx) - self.frame_idx)) + 1
            if next_frame_idx_idx != len(self.valid_frame_idx):
                self.frame_idx = self.valid_frame_idx[next_frame_idx_idx]
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)
            vis.update_renderer()
            print(f"frame_idx: {self.frame_idx}, {len(loads[1].points)} points")
        return vis_next_frame
    
    def vis_prev_frame(self):
        def vis_prev_frame(vis):
            # find previous frame idx
            prev_frame_idx_idx = np.argmin(np.abs(np.array(self.valid_frame_idx) - self.frame_idx)) - 1
            if prev_frame_idx_idx != -1:
                self.frame_idx = self.valid_frame_idx[prev_frame_idx_idx]
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)
            vis.update_renderer()
            print(f"frame_idx: {self.frame_idx}, {len(loads[1].points)} points")
        return vis_prev_frame
    
    def toggle_show_before_dbscan(self):
        def toggle_show_before_dbscan(vis):
            if not "after_dbscan" in self.instance_frame_pcd_list[self.instance_id][self.frame_idx]:
                return

            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            camera_position = copy.deepcopy(params.extrinsic)

            self.show_before_dbscan = not self.show_before_dbscan
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)
            vis.update_renderer()
            intrinsics = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.intrinsic = intrinsics
            cam_params.extrinsic = camera_position

            ctr.convert_from_pinhole_camera_parameters(cam_params, True)
            print(f"show_before_dbscan: {self.show_before_dbscan}, {len(loads[1].points)} points")
        return toggle_show_before_dbscan

def visualizer(instance_pcd_list, instance_pcd_color_list, instance_frame_pcd_list, instance_bounding_box_list, static_bbox_list, unique_instance_id_list, args):
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)
    while True:
        menu_option = input("1. Visualize specific frame\n2. Visualize registered instance\n3. Visualize sparse instance by frame\n4. Visualize instance by frame\n5. Show instance ID list\n6. Print merge distance data\n7. Exit\n")
        if menu_option == "1":
            frame_idx = input("Enter frame index: ")
            if not frame_idx.isdigit() or not int(frame_idx) in idx_range:
                print("Invalid frame index")
                continue
            frame_idx = int(frame_idx)
            frame_vis = Frame_Visualizer(frame_idx, instance_frame_pcd_list, unique_instance_id_list, instance_bounding_box_list, static_bbox_list, args)
            frame_vis.visualize()

        elif menu_option == "4":
            instance_id = input("Enter instance id: ")
            if not instance_id.isdigit() or not int(instance_id) in unique_instance_id_list:
                print("Invalid instance id")
                continue
            instance_id = int(instance_id)
            valid_frame_idx = []
            for frame_idx in idx_range:
                if 'before_dbscan' in instance_frame_pcd_list[instance_id][frame_idx].keys():
                    valid_frame_idx.append(frame_idx)
            print(f"Valid frame indices: {valid_frame_idx}")
            frame_idx = input("Enter frame index: ")
            if not frame_idx.isdigit() or not int(frame_idx) in valid_frame_idx:
                print("Invalid frame index")
                continue
            frame_idx = int(frame_idx)
            instance_vis = Instance_Visualizer(instance_frame_pcd_list, instance_id, frame_idx, valid_frame_idx)
            instance_vis.visualize()

        elif menu_option == "5":
            print(unique_instance_id_list)

        elif menu_option == "7":
            break
        
        else:
            print("Invalid option")