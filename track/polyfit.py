import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

mark_resolution = 0.5

clusters_cloud = o3d.io.read_point_cloud("euclidean_cluster2.pcd")
# o3d.visualization.draw_geometries([clusters_cloud])
cluster_xyz = np.asarray(clusters_cloud.points)
cluster_xyz = cluster_xyz[cluster_xyz[:, 2].argsort()]
# print("idx:", idx, "cluster_xyz:", cluster_xyz)
max_interval = cluster_xyz[len(cluster_xyz) - 1][2] - cluster_xyz[0][2]
cluster_mark_dim = int(max_interval / mark_resolution) + 1
slice_list = [[] for i in range(cluster_mark_dim)]
print("cluster_mark_dim:", cluster_mark_dim, " ,slice_list:", slice_list)
for pt_idx, pt_xyz in enumerate(cluster_xyz):
    pt_x, pt_y, pt_z = pt_xyz
    slice_idx = int((pt_z - cluster_xyz[0][2]) / mark_resolution)
    print("slice_idx:", slice_idx, "pt_xyz:", pt_xyz)
    slice_list[slice_idx].append(pt_xyz)
if len(slice_list) == 0:
    # continue
    exit()
anchor_list = []
for slice_idx, pt_slice in enumerate(slice_list):
    print("pt slice:", pt_slice)
    np_pt_slice = np.asarray(pt_slice)
    print("np_pt_slice:", np_pt_slice)
    if len(pt_slice) == 0:
        continue
    print("anchor:", np_pt_slice.mean(axis=0))
    anchor_list.append(np_pt_slice.mean(axis=0))

z = np.asarray(anchor_list)[:, 2]
print(z)
y = np.asarray(anchor_list)[:, 1]
print(y)
if len(z) >= 2:
    parameter = np.polyfit(z, y, 3)
    print(parameter)

    y2 = parameter[0] * z ** 3 + parameter[1] * z ** 2 + parameter[2] * z + parameter[3]
    line = np.hstack((np.ones((y2.shape[-1], 1)), y2[:, np.newaxis]))
    line = np.hstack((line, z[:, np.newaxis]))

    print("line.shape：", line.shape)
    cluster_xyz = np.vstack((cluster_xyz, line))
    print("cluster_xyz.shape：", cluster_xyz.shape)
    clusters_cloud.points = o3d.utility.Vector3dVector(cluster_xyz)
    o3d.visualization.draw_geometries([clusters_cloud])

