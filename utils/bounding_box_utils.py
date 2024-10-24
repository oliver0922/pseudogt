import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

lidar_to_camera = np.array([[0, -1, 0],[0, 0, -1],[1,0,0]])

class BoundingBox:
    def __init__(self, t=np.array([0,0,0]), s=np.array([0,0,0]), r=0):
        self.t = t
        self.s = s
        self.r = r

    def make_box(self, pcd, fit_method, angle = None):
        pcd = np.array(pcd.points)
        cam_coord_pcd = pcd @ lidar_to_camera.T
        if len(cam_coord_pcd) == 0:
            return None, None
        if fit_method == 'closeness_to_edge':
            corners, ry, area = closeness_rectangle(cam_coord_pcd[:, [0, 2]])
        elif fit_method == 'point_normal':
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(cam_coord_pcd)
            src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=50))
            src.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            corners, ry, area = point_normal_rectangle(src)
        elif fit_method == 'given_angle':
            if angle is None:
                raise ValueError('angle is None')
            corners, ry, area = given_angle_bbox_fit(cam_coord_pcd, angle)
        else:
            raise NotImplementedError(fit_method)
        ry *= -1
        l = np.linalg.norm(corners[0] - corners[1])
        w = np.linalg.norm(corners[0] - corners[-1])
        c = (corners[0] + corners[2]) / 2
        bottom = cam_coord_pcd[:, 1].max()
        # bottom = get_lowest_point_rect(full_ptc, c, l, w, ry)
        h = bottom - cam_coord_pcd[:, 1].min()

        self.t = np.array([c[0], bottom, c[1]])
        self.t = lidar_to_camera.T @ self.t
        self.s = np.array([l, w, h])
        self.r = np.pi / 2 - ry
        return self
    
    def load_gt(self, arr):
        self.t = arr[:3]
        self.s = arr[3:6]
        self.t[2] -= self.s[2] / 2
        self.r = arr[6]
        self.lidar_coord = True
        return self
    
    def get_o3d_instance(self):
        center = copy.deepcopy(self.t)
        center[2] = center[2] + self.s[2] / 2
        lwh = self.s
        axis_angles = np.array([0, 0, self.r + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = o3d.utility.Vector2iVector(lines)

        return line_set, box3d
    
    def get_np_instance(self):
        return np.concatenate((self.t, self.s, [self.r]), axis=0)

    def transform(self, transformation_matrix):
        tr_matrix = np.linalg.inv(transformation_matrix)
        new_tr_mat = np.eye(4)
        rotation = np.arctan2(tr_matrix[1, 0], tr_matrix[0, 0])
        new_tr_mat[:3, :3] = R.from_euler('z', rotation).as_matrix()
        homo_t = np.concatenate((copy.deepcopy(self.t + [0, 0, self.s[2]/2]), [1]))
        new_tr_mat[:3, 3] = (tr_matrix @ homo_t)[:3] - (new_tr_mat @ homo_t)[:3]
        self.t = (new_tr_mat @ homo_t)[:3] - [0, 0, self.t[2]/2]
        self.r += rotation
        return self




def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:,0].min(), projection[:,0].max()
        min_y, max_y = projection[:,1].min(), projection[:,1].max()
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        beta = np.vstack((Dx, Dy)).min(axis=0)
        beta = np.maximum(beta, d0)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
 
    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
 
    area = (max_x - min_x) * (max_y - min_y)
 
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area
 
def get_lowest_point_rect(ptc, xz_center, l, w, ry):
    ptc_xz = ptc[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    ys = ptc[mask, 1]
    return ys.max()

def point_normal_rectangle(src):
    normals = np.array(src.normals)
    # remove normals with strong y
    new_src = o3d.geometry.PointCloud()
    new_src.points = o3d.utility.Vector3dVector(np.array(src.points)[np.where(np.abs(normals[:, 1]) < 0.5)])
    new_src.normals = o3d.utility.Vector3dVector(normals[np.where(np.abs(normals[:, 1]) < 0.5)])
    #o3d.visualization.draw_geometries([new_src], point_show_normal=True)
    normals = normals[np.abs(normals[:, 1]) < 0.5]
    # only 0, 2 axis
    normals = normals[:, [0, 2]]
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    angles = np.where(angles < 0, angles + np.pi, angles)
    angles = np.where(angles >= np.pi / 2, angles - np.pi / 2, angles)
    # find the most frequent angle
    bins = np.arange(0, np.pi / 2, np.pi/72)
    hist, _ = np.histogram(angles, bins=bins)
    angle = bins[np.argmax(hist)]
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = np.array(src.points)[:, [0, 2]] @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    if (max_x - min_x) < (max_y - min_y):
        angle += np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = np.array(src.points)[:, [0, 2]] @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def given_angle_bbox_fit(src, angle):
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = np.array(src.points)[:, [0, 2]] @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    if (max_x - min_x) < (max_y - min_y):
        angle += np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = np.array(src.points)[:, [0, 2]] @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area