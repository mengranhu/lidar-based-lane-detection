import math
import numpy as np
import open3d as o3d
from model import Lane, Cluster
import matplotlib.pyplot as plt


def get_arc_len(Lane):
    return Lane.arc_len


def euclidean_cluster(cloud, tolerance=0.2, min_cluster_size=100, max_cluster_size=1000):
    """
    欧式聚类
    :param cloud:输入点云
    :param tolerance: 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
    :param min_cluster_size:设置一个聚类需要的最少的点数目
    :param max_cluster_size:设置一个聚类需要的最大点数目
    :return:聚类个数
    """

    kdtree = o3d.geometry.KDTreeFlann(cloud)  # 对点云建立kd树索引

    num_points = len(cloud.points)
    processed = [-1] * num_points  # 定义所需变量
    clusters = []  # 初始化聚类
    # 遍历各点
    for index in range(num_points):
        if processed[index] == 1:  # 如果该点已经处理则跳过
            continue
        seed_queue = []  # 定义一个种子队列
        sq_idx = 0
        seed_queue.append(index)  # 加入一个种子点
        processed[index] = 1

        while sq_idx < len(seed_queue):

            k, nn_indices, _ = kdtree.search_radius_vector_3d(cloud.points[seed_queue[sq_idx]], tolerance)

            if k == 1:  # k=1表示该种子点没有近邻点
                sq_idx += 1
                continue
            for j in range(k):

                if nn_indices[j] == num_points or processed[nn_indices[j]] == 1:
                    continue  # 种子点的近邻点中如果已经处理就跳出此次循环继续
                seed_queue.append(nn_indices[j])
                processed[nn_indices[j]] = 1

            sq_idx += 1

        if max_cluster_size > len(seed_queue) > min_cluster_size:
            clusters.append(seed_queue)

    return clusters


def lane_track(LaneMark_np, ec):
    print("[before] LaneMark len:", len(LaneMark_np))
    for cluster_idx in range(len(ec)):
        indexes = ec[cluster_idx]
        clusters_cloud = LaneMark.select_by_index(indexes)
        file_name = "frame_1_cluster_" + str(cluster_idx) + ".pcd"
        o3d.io.write_point_cloud(file_name, clusters_cloud)

        cluster_xyz = np.asarray(clusters_cloud.points)
        cluster_xyz = cluster_xyz[cluster_xyz[:, 2].argsort()]
        # print("idx:", idx, "cluster_xyz:", cluster_xyz)
        max_z = cluster_xyz[len(cluster_xyz) - 1][2]
        # print("max:", max_z)
        min_z = cluster_xyz[0][2]
        # print("min z:", min_z)
        max_interval = max_z - min_z
        min_y = float("inf")
        max_y = float("-inf")
        cluster_mark_dim = int(max_interval / mark_resolution) + 1
        slice_list = [[] for i in range(cluster_mark_dim)]
        # print("cluster_mark_dim:", cluster_mark_dim, " ,slice_list:", slice_list)
        for pt_idx, pt_xyz in enumerate(cluster_xyz):
            pt_x, pt_y, pt_z = pt_xyz
            slice_idx = int((pt_z - min_z) / mark_resolution)
            # print("slice_idx:", slice_idx, "pt_xyz:", pt_xyz)
            slice_list[slice_idx].append(pt_xyz)
            if pt_y < min_y:
                min_y = pt_y
            if pt_y > max_y:
                max_y = pt_y

        anchor_list = []
        for slice_idx, pt_slice in enumerate(slice_list):
            # print("pt slice:", pt_slice)
            np_pt_slice = np.asarray(pt_slice)
            # print("np_pt_slice:", np_pt_slice)
            if len(pt_slice) == 0:
                continue
            # print("anchor:", np_pt_slice.mean(axis=0))
            anchor_list.append(np_pt_slice.mean(axis=0))

        # Calculate arc length
        arc_len = 0
        for anchor_index in range(len(anchor_list) - 1):
            pt1 = [anchor_list[anchor_index][1], anchor_list[anchor_index][2]]
            pt2 = [anchor_list[anchor_index + 1][1], anchor_list[anchor_index + 1][2]]
            # numpy.linalg.norm(a - b)
            arc_len = arc_len + np.linalg.norm(np.asarray(pt1) - np.asarray(pt2))

        z = np.asarray(anchor_list)[:, 2]
        y = np.asarray(anchor_list)[:, 1]
        if arc_len >= 3.0 and len(z) >= 2:
            if max_interval < 30.0:
                parameter = np.polyfit(z, y, 2)
                parameter = np.hstack((0, parameter))
            else:
                parameter = np.polyfit(z, y, 3)

            # if arc_len >= 3.0:
            line = Lane(1, cluster_idx, cluster_xyz, arc_len)
            line.para = parameter
            line.scope = [min_y, max_y, min_z, max_z]
            Lanes.append(line)
            y2 = parameter[0] * z ** 3 + parameter[1] * z ** 2 + parameter[2] * z + parameter[3]
            line_np = np.hstack((20 * np.ones((y2.shape[-1], 1)), y2[:, np.newaxis]))
            line_np = np.hstack((line_np, z[:, np.newaxis]))
            print("cluster idx:", cluster_idx, " LaneMark_np shape:", LaneMark_np.shape, " line shape:", line_np.shape)
            LaneMark_np = np.vstack((LaneMark_np, line_np))
    Lanes.sort(key=get_arc_len, reverse=True)
    print("[after] LaneMark len:", len(LaneMark_np))
    return Lanes, LaneMark_np


def Update(viewer):
    global LaneMark, frame_idx, Lanes

    frame_idx = frame_idx + 1
    pc_file = "../data_set/in/mark2/" + str(frame_idx) + ".pcd"
    next_pcd = o3d.io.read_point_cloud(pc_file)

    next_xyz = np.asarray(next_pcd.points)

    # Exit lane detection
    num, dim = next_xyz.shape
    if num == 0:
        exit()

    LaneMark.points = o3d.utility.Vector3dVector(next_xyz)
    viewer.update_geometry(LaneMark)
    viewer.poll_events()
    viewer.update_renderer()


if __name__ == '__main__':
    Lanes = []
    path = "../../data_set/in/mark2/"
    frame_idx = 1
    mark_resolution = 0.5

    LaneMark = o3d.io.read_point_cloud(path + str(frame_idx) + ".pcd")
    # o3d.visualization.draw_geometries([LaneMark])
    LaneMark_np = np.asarray(LaneMark.points)
    ec = euclidean_cluster(LaneMark, tolerance=1.0, min_cluster_size=10, max_cluster_size=100000)
    Lanes, LaneMark_np = lane_track(LaneMark_np, ec)

    # print("LaneMark_np shape:", LaneMark_np.shape)
    # print("Lanes size:", len(Lanes))
    LaneMark.points = o3d.utility.Vector3dVector(LaneMark_np)
    o3d.visualization.draw_geometries([LaneMark])
    # o3d.io.write_point_cloud(str(frame_idx) + "_out.pcd", LaneMark)

    # second
    frame_idx = frame_idx + 1
    LaneMark = o3d.io.read_point_cloud(path + str(frame_idx) + ".pcd")
    LaneMark_np = np.asarray(LaneMark.points)
    ec = euclidean_cluster(LaneMark, tolerance=1.0, min_cluster_size=5, max_cluster_size=100000)
    # Optimal Matching
    cluster_list = []
    for idx in range(len(ec)):
        ind = ec[idx]
        clusters_cloud = LaneMark.select_by_index(ind)
        file_name = "frame_" + str(frame_idx) + "_cluster_" + str(idx) + ".pcd"
        o3d.io.write_point_cloud(file_name, clusters_cloud)
        cluster_xyz = np.asarray(clusters_cloud.points)
        cluster = Cluster(frame_idx, idx, cluster_xyz)
        cluster.init()
        cluster_list.append(cluster)

    # lane match to cluster
    for _, lane in enumerate(Lanes):
        last_min_y = lane.scope[0]
        last_max_y = lane.scope[1]
        last_min_z = lane.scope[2]
        last_max_z = lane.scope[3]

        for _, cluster in enumerate(cluster_list):
            now_min_z = cluster.min_z
            now_max_z = cluster.max_z
            now_min_y = cluster.min_y
            now_max_y = cluster.max_y
            pt_near_z = cluster.cluster_xyz[0][2]
            pt_near_y = cluster.cluster_xyz[0][1]
            pt_far_z = cluster.cluster_xyz[cluster.len - 1][2]
            pt_far_y = cluster.cluster_xyz[cluster.len - 1][1]
            # pt_near_z = cluster.cluster_xyz[0][2]
            # yaw * v * dealt_t + covariance
            if not (now_min_z > last_max_z + 5 or now_max_z < last_min_z - 5):
                if not (now_min_y > last_max_z + 1 or now_max_y < last_min_y - 1):
                    # 严格的最小二乘误差 Check(暂且选择now最近、最远点）
                    des_near_y = lane.para[0] * pt_near_z ** 3 \
                                 + lane.para[1] * pt_near_z ** 2 \
                                 + lane.para[2] * pt_near_z + lane.para[3]
                    des_far_y = lane.para[0] * pt_far_z ** 3 \
                                + lane.para[1] * pt_far_z ** 2 \
                                + lane.para[2] * pt_far_z + lane.para[3]

                    if abs(des_near_y - pt_near_y) < 1.0 and abs(des_far_y - pt_far_y) < 1.0:
                        lane.add_xyz(cluster.cluster_xyz)
                        print("[now] cluster idx: ", cluster.cluster_id, " [last] lane idx:", lane.cluster_id)
                        continue

    # # for debug
    # for lane_idx, lane in enumerate(Lanes):
    #     if lane.update:
    #         print("save cluster: ", lane.cluster_id)
    #         lane.pcd_save()
    #         lane.line_save()

    print("1111111111111111111111111111111111111111")
    # Polyfit 、Update [confidence] & Delete
    for lane_idx, lane in enumerate(Lanes):
        print(" lane_idx: ", lane_idx)
        lane.state_update()

    Lane_tmp = []
    Lane_output = []
    for lane_idx, lane in enumerate(Lanes):
        if lane.confidence > 0.2:
            print("confidence > 0.2, ", lane_idx)
            Lane_tmp.append(lane)
            if len(lane.para) > 0:
                z = np.linspace(lane.scope[2], lane.scope[3], 100)
                y2 = lane.para[0] * z ** 3 \
                     + lane.para[1] * z ** 2 \
                     + lane.para[2] * z \
                     + lane.para[3]
                line = np.hstack((20 * np.ones((y2.shape[-1], 1)), y2[:, np.newaxis]))
                line = np.hstack((line, z[:, np.newaxis]))
                LaneMark_np = np.vstack((LaneMark_np, line))
        if lane.confidence > 0.9:
            Lane_output.append(lane)
    Lanes = Lane_tmp

    print("Lanes size:", len(Lanes))
    LaneMark.points = o3d.utility.Vector3dVector(LaneMark_np)
    o3d.io.write_point_cloud(str(frame_idx) + "_out.pcd", LaneMark)
    o3d.visualization.draw_geometries([LaneMark])

    # # Viewer
    # viewer = o3d.visualization.VisualizerWithKeyCallback()
    # viewer.create_window("Lane Detection", 800, 800)
    # view = viewer.get_view_control()
    #
    # render = viewer.get_render_option()
    #
    # render.background_color = [2, 2, 0]
    # # render.show_coordinate_frame=True
    # render.line_width = 5
    # render.point_size = 5
    #
    # XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[2, 4, 0])
    # # viewer.add_geometry(XYZ)
    # # Lane.points = o3d.utility.Vector3dVector(all_point_cloud)
    # viewer.add_geometry(LaneMark)
    # # viewer.add_geometry(Line_set)
    # viewer.register_animation_callback(Update)
    # viewer.run()
    # viewer.destroy_window()
