import json
import numpy as np
import matplotlib.pyplot as plt
from getleaves import getLeavesWithoutPot

def gettwogroups():
    _, leaves_points_pcd= getLeavesWithoutPot()

    with open('whole_incompleteleaves/import-pointclouds-pcd/ann/clurstering.pcd.json','r') as f:
        data = json.load(f)

    pcd = leaves_points_pcd
    pcd_np = np.asarray(pcd.points)

    points_indices = data['figures'][0]['geometry']['indices']
    incomplete_xyz_np = pcd_np[points_indices, :]

    #画图
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(incomplete_xyz_np[:, 0], incomplete_xyz_np[:, 1], incomplete_xyz_np[:, 2], s=0.5)
    # plt.title("incomplete")
    # plt.show()

    return pcd_np, incomplete_xyz_np, points_indices

if __name__ == '__main__':
    gettwogroups()