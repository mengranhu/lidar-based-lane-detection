import numpy as np
import open3d as o3d

a = [1, 3, 6, 8]

a1 = np.asarray(a)
print("a1:", a1)

pcd = o3d.io.read_point_cloud("frame_1_cluster_0.pcd")
xyz = np.asarray(pcd.points)
print("xyz:", xyz)
# print("xyz col 2:", xyz[:, 2])
print("xyz col 2:", xyz[:, 2].shape)
new_xz = np.hstack((xyz[:, 0], xyz[:, 2]))
new_xz2 = np.vstack((xyz[:, 0], xyz[:, 2]))
print("new_xz:", new_xz.shape)
print("new_xz2:", new_xz2.shape)

np1 = np.array([[0, 1, 5],
                [2, 3, 8]])

np2 = np1.transpose()
print("np2:", np2)
