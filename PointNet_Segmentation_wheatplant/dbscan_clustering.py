from getleaves import getLeavesWithoutPot
from sklearn.cluster import DBSCAN
import open3d as o3d
import numpy as np

leaves_points, leaves_points_pcd= getLeavesWithoutPot()
# clustering = DBSCAN(eps=2.1).fit(leaves_points)
# clustering = DBSCAN(eps=1.9).fit(leaves_points)
clustering = DBSCAN(eps=3.9).fit(leaves_points)
labels = clustering.labels_

colors = np.random.rand(np.int64(labels.max())+1, 3)
colors[0, :] = 0.0
leaves_points_pcd.colors = o3d.utility.Vector3dVector(colors[labels, :])
#画图
o3d.visualization.draw_geometries([leaves_points_pcd])

#导出
# o3d.io.write_point_cloud('./clurstering.pcd', leaves_points_pcd)