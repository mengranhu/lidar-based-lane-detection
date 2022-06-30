import numpy as np


def get_coefficient(employee):
    return employee[0]


def road_width_update(xyz_cluster, mark_resolution, ego_yaw, ego_curvature, ego_curvature_ratio):
    # Distribution of the relative lateral position
    mark_resolution_dim = int(4.8 / mark_resolution + 1)
    # print("mark_resolution_dim: ", mark_resolution_dim)
    distribution_space = np.zeros(mark_resolution_dim)
    for i, xyz in enumerate(xyz_cluster):
        point_x, point_y, point_z = xyz
        if 0 < point_z < 10:
            y1 = ego_yaw * point_z + 0.5 * ego_curvature * point_z * point_z \
                 + ego_curvature_ratio * point_z * point_z * point_z / 6
            dealt_y_slice_num = (point_y - y1) / mark_resolution
            if dealt_y_slice_num < -mark_resolution_dim / 2:
                continue
            elif dealt_y_slice_num > mark_resolution_dim / 2:
                continue
            distribution_space[int(dealt_y_slice_num + mark_resolution_dim / 2)] = \
                distribution_space[int(dealt_y_slice_num + mark_resolution_dim / 2)] + 1

    # road width update
    Rxx = []
    for pos_idx in range(mark_resolution_dim):
        # print(" pos_idx:", pos_idx, " mark_resolution_dim:", mark_resolution_dim)
        for k in range(mark_resolution_dim - pos_idx - 1 - 10):
            # print("k:", k)
            num1 = distribution_space[pos_idx]
            num2 = distribution_space[pos_idx + k + 1 + 10]
            # left index // right index
            rxx_idx = [num1 * num2, pos_idx, pos_idx + k + 1 + 10]
            Rxx.append(rxx_idx)
    # np.savetxt("foo.csv", Rxx, delimiter=",")
    Rxx.sort(key=get_coefficient, reverse=True)
    print("Rxx[0]:", Rxx[0])
    distribution_num, left_idx, right_idx = Rxx[0]
    if distribution_num > 100:
        lane_width = (right_idx - left_idx) * mark_resolution
        print("width:", lane_width)
    else:
        lane_width = 0
        print("width:", lane_width)
    return lane_width
