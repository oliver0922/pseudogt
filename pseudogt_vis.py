import open3d
import numpy as np
import pickle

from utils.visualizer_utils import visualizer

CAM_LOCS = {1:'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}
CAM_NAMES = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
CAM_NAMES = ['FRONT']

AXIS_PCD = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
LIDAR_TO_CAMERA = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])


if __name__ == "__main__":
    # load files
    with open('results/insatnce_bounding_box_list.pkl', 'rb') as f:
        instance_bounding_box_list = pickle.load(f)
    with open('results/t_bbox_list.pkl', 'rb') as f:
        t_bbox_list = pickle.load(f)
    with open('results/sparse_bbox_list.pkl', 'rb') as f:
        sparse_bbox_list = pickle.load(f)
    with open('results/unique_instance_id_list.pkl', 'rb') as f:
        unique_instance_id_list = pickle.load(f)
    with open('results/registration_data_list.pkl', 'rb') as f:
        registration_data_list = pickle.load(f)
    with open('results/sparse_bbox_data_list.pkl', 'rb') as f:
        sparse_bbox_data_list = pickle.load(f)
    with open('results/instance_frame_pcd_list.pkl', 'rb') as f:
        instance_frame_pcd_list = pickle.load(f)
    with open('results/args.pkl', 'rb') as f:
        args = pickle.load(f)
    idx_range = range(args.rgs_start_idx, args.rgs_end_idx+1)
    visualizer(instance_bounding_box_list, t_bbox_list, sparse_bbox_list, unique_instance_id_list, registration_data_list, sparse_bbox_data_list, instance_frame_pcd_list, idx_range, args)
