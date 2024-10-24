import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation as R
import hdbscan
import types
import math



def dbscan(pcd, eps=0.2, min_points=10, print_progress=False, debug=False ):
 
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
 
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")
 
    un_noise_idx = np.where(labels != -1)[0]
  
    return un_noise_idx



def dbscan_max_cluster(pcd, eps=0.2, min_points=10, print_progress=False, debug=False ):
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
 
    # same with dbscan, but only returns label with maximum points
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")

    labels_new = labels[np.where(labels != -1)]
    if len(labels_new) == 0:
        return np.array([])
    max_cluster_idx = np.argmax(np.bincount(labels_new))
    un_noise_idx = np.where(labels == max_cluster_idx)[0]

    return un_noise_idx



def hdbscan_idx(pcd, min_cluster_size, culster_selection_epsilion):
    
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon= culster_selection_epsilion, gen_min_span_tree=True)
    cluster.fit(np.array(pcd.points))
    labels = cluster.labels_
    
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
 
    un_noise_idx = np.where(labels != -1)[0]

    return un_noise_idx



def dbscan_cluster_filter(pcd, eps=0.2, min_points=10, max_dist = 2.0, print_progress=False, debug=False):
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
 
    # same with dbscan, but only returns label with maximum points
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")

    cluster_pc = [[] for _ in range(max_label+1)]
    for i in range(max_label+1):
        cluster_pc[i] = np.where(labels == i)[0]
    if len(cluster_pc) == 0:
        return np.array([])
    sorted_cluster_pc = sorted(cluster_pc, key=len, reverse=True)

    un_noise_idx = []
    un_noise_idx.extend(sorted_cluster_pc[0])
    for i in range(1, len(sorted_cluster_pc)):
        if np.linalg.norm(np.mean(np.array(pcd.points)[sorted_cluster_pc[i]], axis=0) - \
                          np.mean(np.array(pcd.points)[un_noise_idx]), axis=0) > max_dist:
            break
        un_noise_idx.extend(sorted_cluster_pc[i])

    return un_noise_idx
 


def transform_np_points(pcd, transformation_matrix):
    src = open3d.geometry.PointCloud()
    src.points = open3d.utility.Vector3dVector(copy.deepcopy(pcd))
    src.transform(transformation_matrix)
    return np.array(src.points)