import open3d as o3d
import open3d
import numpy as np
import os
import argparse
import copy
from utils.utils import translate_boxes_to_open3d_gtbox
from utils.open3d_utils import set_black_background, set_white_background


AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

class Frame_Visualizer:
    def __init__(self, frame_idx, instance_frame_pcd_list, unique_instance_id_list, instance_bounding_box_list, t_bbox_list, sparse_bbox_list, sparse_bbox_data_list, args):
        self.frame_idx = frame_idx
        self.instance_frame_pcd_list = instance_frame_pcd_list
        self.unique_instance_id_list = unique_instance_id_list
        self.instance_bounding_box_list = instance_bounding_box_list
        self.t_bbox_list = t_bbox_list
        self.sparse_bbox_list = sparse_bbox_list
        self.sparse_bbox_data_list = sparse_bbox_data_list
        self.args = args

    def load_data(self):
        load_list = []
        for instance_id in self.unique_instance_id_list:
            if self.frame_idx in self.instance_bounding_box_list[instance_id].keys():
                load_list.append(self.instance_bounding_box_list[instance_id][self.frame_idx])
            if self.frame_idx in self.t_bbox_list[instance_id].keys():
                load_list.append(self.t_bbox_list[instance_id][self.frame_idx])
            if self.frame_idx in self.sparse_bbox_list[instance_id].keys():
                load_list.append(self.sparse_bbox_list[instance_id][self.frame_idx])
                load_list.append(self.sparse_bbox_data_list[instance_id][self.frame_idx]["init_line"])
        try:
            full_pc = np.fromfile(os.path.join(self.args.dataset_path,f'scene-{self.args.scene_idx}','pointcloud',f'{str(self.frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 3)
        except:
            self.frame_idx -= 1
            self.args.rgs_end_idx -= 1
            return self.load_data()

        full_pc = full_pc[full_pc[:, 2] > self.args.z_threshold]
        gt_bbox = np.fromfile(os.path.join(self.args.dataset_path,f'scene-{self.args.scene_idx}','annotations',f'{str(self.frame_idx).zfill(6)}.bin')).reshape(-1, 7)
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
        for i in range(np.max(self.unique_instance_id_list) + 1):
            if "after_dbscan" in self.instance_frame_pcd_list[i][self.frame_idx].keys():
                src = open3d.geometry.PointCloud()
                src.points = open3d.utility.Vector3dVector(self.instance_frame_pcd_list[i][self.frame_idx]["after_dbscan"])
                src.paint_uniform_color(self.instance_frame_pcd_list[i][self.frame_idx]["color"])
                src_list.append(src)
                id_text : open3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh.create_text(str(i), depth=3.0).to_legacy()
                id_text.paint_uniform_color(self.instance_frame_pcd_list[i][self.frame_idx]["color"])
                location = np.mean(self.instance_frame_pcd_list[i][self.frame_idx]["after_dbscan"], axis=0) + np.array([0, 0, 1.0])
                id_text.transform([[0.1, 0, 0, location[0]], [0, 0.1, 0, location[1]], [0, 0, 0.1, location[2]], [0, 0, 0, 1]])
                id_list.append(id_text)
        
        return [AXIS_PCD] + load_list + gt_list + src_list + id_list
    
    def visualize(self):
        print(f"Visualizing frame {self.frame_idx}")
        o3d.visualization.draw_geometries_with_key_callbacks(self.load_data(), {ord("B"): set_black_background, ord("W"): set_white_background,
            ord("A"): self.vis_prev_frame(),
            ord("D"): self.vis_next_frame()})
        
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
    
class Registered_Instance_Visualizer:
    def __init__(self, instance_id, registration_data_list, unique_instance_id_list):
        self.instance_id = instance_id
        self.registration_data_list = registration_data_list
        self.unique_instance_id_list = unique_instance_id_list

    def load_data(self):
        load_list = []
        load_list.append(self.registration_data_list[self.instance_id]["registered_src"])
        load_list.append(self.registration_data_list[self.instance_id]["line_set_lidar"])
        load_list.append(self.registration_data_list[self.instance_id]["t_line_set_lidar"])
        load_list.append(self.registration_data_list[self.instance_id]["gt_lines"])
        return [AXIS_PCD] + load_list
    
    def visualize(self):
        print(f"Visualizing registered instance {self.instance_id}")
        o3d.visualization.draw_geometries_with_key_callbacks(self.load_data(), {ord("B"): set_black_background, ord("W"): set_white_background, ord("A"): self.vis_prev_instance(), ord("D"): self.vis_next_instance()})

    def vis_next_instance(self):
        def vis_next_instance(vis):
            # find next instance id
            next_instance_id_idx = self.instance_id + 1
            while next_instance_id_idx <= len(self.unique_instance_id_list) - 1 and \
                not (next_instance_id_idx in self.unique_instance_id_list and "line_set_lidar" in self.registration_data_list[next_instance_id_idx]):
                next_instance_id_idx += 1
            if next_instance_id_idx != len(self.unique_instance_id_list):
                self.instance_id = next_instance_id_idx

            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)
            vis.update_renderer()
            print(f"instance_id: {self.instance_id}")
        return vis_next_instance
    
    def vis_prev_instance(self):
        def vis_prev_instance(vis):
            # find previous instance id
            prev_instance_id_idx = self.instance_id - 1
            while prev_instance_id_idx >= 0 and \
                not (prev_instance_id_idx in self.unique_instance_id_list and "line_set_lidar" in self.registration_data_list[prev_instance_id_idx]):
                prev_instance_id_idx -= 1
            if prev_instance_id_idx != -1:
                self.instance_id = prev_instance_id_idx
            
            vis.clear_geometries()
            loads = self.load_data()
            for load in loads:
                vis.add_geometry(load)
            vis.update_renderer()
            print(f"instance_id: {self.instance_id}")
        return vis_prev_instance
    
class Sparse_Instance_Visualizer:
    def __init__(self, instance_id, frame_idx, valid_frame_idx, sparse_bbox_data_list):
        self.frame_idx = frame_idx
        self.instance_id = instance_id
        self.valid_frame_idx = valid_frame_idx
        self.sparse_bbox_data_list = sparse_bbox_data_list

    def load_data(self):
        load_list = []
        if "src" in self.sparse_bbox_data_list[self.instance_id][self.frame_idx].keys():
            load_list.append(self.sparse_bbox_data_list[self.instance_id][self.frame_idx]["src"])
        if "line_set" in self.sparse_bbox_data_list[self.instance_id][self.frame_idx].keys():
            load_list.append(self.sparse_bbox_data_list[self.instance_id][self.frame_idx]["line_set"])
        if "init_line" in self.sparse_bbox_data_list[self.instance_id][self.frame_idx].keys():
            load_list.append(self.sparse_bbox_data_list[self.instance_id][self.frame_idx]["init_line"])
        if "gt_lines" in self.sparse_bbox_data_list[self.instance_id][self.frame_idx].keys():
            load_list.append(self.sparse_bbox_data_list[self.instance_id][self.frame_idx]["gt_lines"])
        if "nearest_bbox" in self.sparse_bbox_data_list[self.instance_id][self.frame_idx].keys():
            load_list.append(self.sparse_bbox_data_list[self.instance_id][self.frame_idx]["nearest_bbox"])
        return [AXIS_PCD] + load_list
    
    def visualize(self):
        print(f"Visualizing sparse instance {self.instance_id}, frame {self.frame_idx}")
        o3d.visualization.draw_geometries_with_key_callbacks(self.load_data(), {ord("B"): set_black_background, ord("W"): set_white_background, ord("A"): self.vis_prev_frame(), ord("D"): self.vis_next_frame()})

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
            print(f"frame_idx: {self.frame_idx}, {len(load[1].points)} points")
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
            print(f"frame_idx: {self.frame_idx}, {len(load[1].points)} points")
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
            print(f"frame_idx: {self.frame_idx}, {len(load[1].points)} points")
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
            print(f"frame_idx: {self.frame_idx}, {len(load[1].points)} points")
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

def visualizer(instance_bounding_box_list, t_bbox_list, sparse_bbox_list, unique_instance_id_list, registration_data_list, sparse_bbox_data_list, instance_frame_pcd_list, merge_distance_data, idx_range, args):
    while True:
        menu_option = input("1. Visualize specific frame\n2. Visualize registered instance\n3. Visualize sparse instance by frame\n4. Visualize instance by frame\n5. Show instance ID list\n6. Print merge distance data\n7. Exit\n")
        if menu_option == "1":
            frame_idx = input("Enter frame index: ")
            if not frame_idx.isdigit() or not int(frame_idx) in idx_range:
                print("Invalid frame index")
                continue
            frame_idx = int(frame_idx)
            frame_vis = Frame_Visualizer(frame_idx, instance_frame_pcd_list, unique_instance_id_list, instance_bounding_box_list, t_bbox_list, sparse_bbox_list, sparse_bbox_data_list, args)
            frame_vis.visualize()

        elif menu_option == "2":
            instance_id = input("Enter instance id: ")
            if not instance_id.isdigit() or not int(instance_id) in unique_instance_id_list:
                print("Invalid instance id")
                continue
            instance_id = int(instance_id)
            registered_instance_vis = Registered_Instance_Visualizer(instance_id, registration_data_list, unique_instance_id_list)
            registered_instance_vis.visualize()

        elif menu_option == "3":
            instance_id = input("Enter instance id: ")
            if not instance_id.isdigit() or not int(instance_id) in unique_instance_id_list:
                print("Invalid instance id")
                continue
            instance_id = int(instance_id)
            valid_frame_idx = []
            for frame_idx in idx_range:
                if 'src' in sparse_bbox_data_list[instance_id][frame_idx].keys():
                    valid_frame_idx.append(frame_idx)
            print(f"Valid frame indices: {valid_frame_idx}")
            frame_idx = input("Enter frame index: ")
            if not frame_idx.isdigit() or not int(frame_idx) in valid_frame_idx:
                print("Invalid frame index")
                continue
            frame_idx = int(frame_idx)
            sparse_instance_vis = Sparse_Instance_Visualizer(instance_id, frame_idx, valid_frame_idx, sparse_bbox_data_list)
            sparse_instance_vis.visualize()

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

        elif menu_option == "6":
            ind1 = input("Enter instance id 1: ")
            ind2 = input("Enter instance id 2: ")
            if not ind1.isdigit() or not ind2.isdigit():
                print("Invalid instance id")
                continue
            ind1 = int(ind1)
            ind2 = int(ind2)
            if (ind1, ind2) in merge_distance_data.keys():
                print(f"distance between {ind1} and {ind2} is {merge_distance_data[(ind1, ind2)]}")
            else:
                print(f"distance between {ind1} and {ind2} is not found")
        
        else:
            print("Invalid option")