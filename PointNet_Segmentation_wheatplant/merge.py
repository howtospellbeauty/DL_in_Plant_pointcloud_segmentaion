import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 获取路径
file_dir_1 = './data/Plant_1_M.ply'
file_dir_2 = './data/Plant_1_S.ply'
#读取点云

ply1 = o3d.io.read_point_cloud(file_dir_1)
ply2 = o3d.io.read_point_cloud(file_dir_2)
ply_total = ply1 + ply2
ply_total = ply_total.paint_uniform_color(np.array([0, 0, 0]))

# data = np.asarray(ply_total.points)
o3d.visualization.draw_geometries([ply_total])
# plt.show()
# kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(data)

