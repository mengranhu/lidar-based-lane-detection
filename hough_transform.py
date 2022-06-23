import numpy as np
import open3d as o3d


def get_key(employee):
    return employee[3]


def conic_hough(xyz, offset_dim, yaw_dim, curvature_dim,
                min_offset, max_offset,
                min_yaw, max_yaw,
                min_curvature, max_curvature):

    offset_step = (max_offset - min_offset) / offset_dim
    yaw_step = (max_yaw - min_yaw) / yaw_dim
    curvature_step = (max_curvature - min_curvature) / curvature_dim
    hough_space = np.zeros((offset_dim, yaw_dim, curvature_dim))

    for idx, pt in enumerate(xyz):
        i_x, i_y, i_z = pt
        # print("individual point:", pt)
        for i_yaw in range(yaw_dim):
            yaw = min_yaw + yaw_step * i_yaw
            # print("i_yaw:", i_yaw, "yaw:", yaw)
            for i_curvature in range(curvature_dim):
                curvature = min_curvature + curvature_step * i_curvature
                # print("i_curvature:", i_curvature, "curvature:", curvature)
                offset = (i_y - i_z * yaw - i_z * i_z * curvature)
                if offset > max_offset:
                    continue
                elif offset < min_offset:
                    continue
                i_offset = (offset - min_offset) / offset_step
                # print(("offset:", offset, " i_offset: ", i_offset))
                hough_space[int(i_offset), i_yaw, i_curvature] = hough_space[int(i_offset), i_yaw, i_curvature] + 1
                # hough_space[int(i_offset), i_yaw, 5] = hough_space[int(i_offset), i_yaw, 5] + 1
    # print("\n Hough space:", hough_space, "\nspace shape:", hough_space.shape)
    vote_idx_list = []

    for i, item in enumerate(hough_space):
        # print("offset idx:", i, "item:", item)
        for j, item1 in enumerate(item):
            # print("yaw idx:", j, "item1:", item1)
            for k, votes in enumerate(item1):
                # print("curvature idx:", k, "item2:", votes)
                vote_idx = [i, j, k, votes]
                vote_idx_list.append(vote_idx)
    vote_idx_list.sort(key=get_key, reverse=True)
    opt_offset_idx, opt_yaw_idx, opt_curvature_idx, opt_votes = vote_idx_list[0]

    # print("offset idx", opt_offset_idx)
    # print("yaw idx", opt_yaw_idx)
    # print("cur idx", opt_curvature_idx)
    # print("max vote:", opt_votes)

    opt_offset = min_offset + opt_offset_idx * offset_step
    opt_yaw = min_yaw + opt_yaw_idx * yaw_step
    opt_curvature = min_curvature + opt_curvature_idx * curvature_step

    print("opt_offset: ", opt_offset)
    print("opt_yaw: ", opt_yaw)
    print("opt_curvature: ", opt_curvature)
    return opt_offset, opt_yaw, opt_curvature
