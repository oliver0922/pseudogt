import copy
import open3d
import open3d as o3d
import numpy as np
import pandas as pd


path = '/Users/injae/Desktop/code/OpenPCDet/000000.ply'
pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])
