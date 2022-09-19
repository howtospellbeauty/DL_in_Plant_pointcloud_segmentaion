import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def getLeavesWithoutPot():
    with open('import-pointclouds-pcd/ann/sink_pointcloud.pcd.json','r') as f:
        data = json.load(f)

    pcd = o3d.io.read_point_cloud("./import-pointclouds-pcd/pointcloud/sink_pointcloud.pcd")
    pcd_np = np.asarray(pcd.points)

    points_indices = data['figures'][0]['geometry']['indices']
    leaf_xyz = pcd_np[points_indices, :]

    #将没盆的转换为pcd形式，方便后续画图
    #首先建立一个o3d的点云对象
    leaf_xyz_pcd = o3d.geometry.PointCloud()
    #使用Vector3dVector方法转换
    leaf_xyz_pcd.points = o3d.utility.Vector3dVector(leaf_xyz)
    
    # 画图
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(leaf_xyz[:, 0], leaf_xyz[:, 1], leaf_xyz[:, 2], s=0.5)
    # plt.title("Leaves without pot")
    # plt.show()

    return leaf_xyz, leaf_xyz_pcd

if __name__ == '__main__':
    getLeavesWithoutPot()