import open3d
import numpy as np
import pickle


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes.t
    lwh = [gt_boxes.l, gt_boxes.w, gt_boxes.h]
    axis_angles = np.array([0, 0, gt_boxes.ry + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d



pcd = np.fromfile('/Users/injae/Desktop/code/OpenPCDet/vis/camera_coord.bin').reshape(-1,3)
with open(file='/Users/injae/Desktop/code/OpenPCDet/vis/001982.pkl', mode='rb') as f:
    bboxes = pickle.load(f)

bbox = bboxes[0]

line_set, box3d = translate_boxes_to_open3d_instance(bbox)



src = open3d.open3d.geometry.PointCloud()
src.points = open3d.utility.Vector3dVector(pcd)
src.paint_uniform_color([1, 0.706, 0])
axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
open3d.visualization.draw_geometries([axis_pcd,src,line_set])
