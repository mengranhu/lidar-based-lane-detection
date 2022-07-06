import numpy as np
import open3d as o3d

from lane_save import lane_save


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
                offset = (i_y - i_z * yaw - 0.5 * i_z * i_z * curvature)
                if offset >= max_offset:
                    continue
                elif offset <= min_offset:
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

    if opt_votes < 2:
        flag_positive = False
        opt_offset = 0
        opt_yaw = 0
        opt_curvature = 0
    else:
        flag_positive = True
        opt_offset = min_offset + opt_offset_idx * offset_step
        opt_yaw = min_yaw + opt_yaw_idx * yaw_step
        opt_curvature = min_curvature + opt_curvature_idx * curvature_step

    print("hough votes:", opt_offset, opt_yaw, opt_curvature, opt_votes)
    return flag_positive, opt_offset, opt_yaw, opt_curvature


def conic_hough_multi_line(xyz, offset_dim, yaw_dim, curvature_dim,
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
                offset = (i_y - i_z * yaw - 0.5 * i_z * i_z * curvature)
                if offset >= max_offset:
                    continue
                elif offset <= min_offset:
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

    opt_offset_list = []
    opt_yaw_list = []
    opt_curvature_list = []

    for index in range(len(vote_idx_list)):
        if len(opt_offset_list) >= 8:
            break
        opt_offset_idx, opt_yaw_idx, opt_curvature_idx, opt_votes = vote_idx_list[index]
        if opt_votes < 2:
            # flag_positive = False
            opt_offset = 0
            opt_yaw = 0
            opt_curvature = 0
        else:
            # flag_positive = True
            opt_offset = min_offset + opt_offset_idx * offset_step
            opt_yaw = min_yaw + opt_yaw_idx * yaw_step
            opt_curvature = min_curvature + opt_curvature_idx * curvature_step
            # 2 for lane width
            if all(abs(item - opt_offset) > 2 for item in opt_offset_list):
                print("line ", index, " -> hough votes:", opt_offset, opt_yaw, opt_curvature, opt_votes)
                opt_offset_list.append(opt_offset)
                opt_yaw_list.append(opt_yaw)
                opt_curvature_list.append(opt_curvature)
            else:
                continue

    print("1829493248948-02350-:-----------:", len(opt_offset_list))
    return opt_offset_list, opt_yaw_list, opt_curvature_list


if __name__ == "__main__":
    pcd_name = "100"
    pcd = o3d.io.read_point_cloud(pcd_name + ".pcd", format='pcd')
    # o3d.visualization.draw_geometries([pcd])
    xyz = np.asarray(pcd.points)
    offset_list, yaw_list, curvature_list = conic_hough_multi_line(xyz, 80, 60, 40, -4, 4, -3, 3, -0.002, 0.002)
    print("offset:", offset_list, "yaw:", yaw_list, "curvature:", curvature_list)

    for i in range(len(offset_list)):
        lane_save(xyz, offset_list[i], yaw_list[i], curvature_list[i], i, pcd_name)
