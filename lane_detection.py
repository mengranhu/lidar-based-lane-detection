from pathlib import Path
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import numpy as np
import time
from hough_transform import conic_hough
from distribution_lateral_position import road_width_update
from lane_save import straight_line_save


def point_grid(radius):
    points = list()
    # lines = list()
    for z in range(-radius, radius + 1, 1):
        points.append([0, -radius, z])
        points.append([0, radius, z])
    for y in range(-radius, radius + 1, 1):
        points.append([0, y, -radius])
        points.append([0, y, radius])
    lines = [[i, i + 1] for i in range(0, len(points), 2)]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines), )
    line_set.paint_uniform_color([0.2, 0.2, 0.2])
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def update(vis):
    global lane_width, left_offset, left_yaw, left_curvature, left_curvature_ratio
    global right_offset, right_yaw, right_curvature, right_curvature_ratio
    global PCD_static, PCD_dynamic, Line_set, XYZ, KEY, IMG, pcs_path, pcd_idx
    global P, Q, H, R, A
    time.sleep(0.3)
    pcd_idx = pcd_idx + 1
    pcd = o3d.io.read_point_cloud(pcs_path + str(pcd_idx) + ".pcd")

    # print("Path:", pcs_path + str(pcd_idx) + ".pcd")
    next_xyz = np.asarray(pcd.points)
    # exit lane detection after last pcd file
    num, dim = next_xyz.shape
    if num == 0:
        exit()

    z_max = 0

    # Predict
    state_left = np.array([left_offset, left_yaw, left_curvature, left_curvature_ratio])
    state_right = np.array([right_offset, right_yaw, right_curvature, right_curvature_ratio])
    state_left_predict = np.dot(A, state_left)
    state_right_predict = np.dot(A, state_right)
    PMatPred = A * P * A.transpose() + Q

    # Update
    mea_left_lane = []
    mea_right_lane = []
    for idx, x_y_z in enumerate(next_xyz):
        next_pt_x, next_pt_y, next_pt_z = x_y_z
        if next_pt_z > 0:
            next_des_y_left = left_offset + left_yaw * next_pt_z + 0.5 * left_curvature * next_pt_z * next_pt_z
            next_des_y_right = right_offset + right_yaw * next_pt_z + 0.5 * right_curvature * next_pt_z * next_pt_z
            if abs(next_des_y_left - next_pt_y) < lane_width / 4:
                mea_left_lane.append(x_y_z)

            if abs(next_pt_y - next_des_y_right) < lane_width / 4:
                mea_right_lane.append(x_y_z)
            if next_pt_z > z_max:
                z_max = next_pt_z

            # if pcd_idx == 38:
            #     if next_pt_z > 30:
            #         print(next_pt_x, next_pt_y, next_pt_z)

    # measurement detected by last step parameter space
    # left_lane_positive, left_offset, left_yaw, left_curvature
    # measurement_left
    left_lane_positive, left_offset, left_yaw, left_curvature = conic_hough(mea_left_lane, 10, 20, 50,
                                                                            left_offset - 0.3, left_offset + 0.3,
                                                                            left_yaw - 0.5, left_yaw + 0.5,
                                                                            left_curvature - 0.002,
                                                                            left_curvature + 0.002)
    measurement_left = left_offset, left_yaw, left_curvature
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(mea_left_lane)
    o3d.io.write_point_cloud(str(pcd_idx) + "_left_mark.pcd", pcd1)

    # left_lane_positive, left_offset, left_yaw, left_curvature
    # measurement_right
    right_lane_positive, right_offset, right_yaw, right_curvature = conic_hough(mea_right_lane, 10, 20, 50,
                                                                                right_offset - 0.3, right_offset + 0.3,
                                                                                right_yaw - 0.5, right_yaw + 0.5,
                                                                                left_curvature - 0.002,
                                                                                left_curvature + 0.002)
    measurement_right = right_offset, right_yaw, right_curvature
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(mea_right_lane)
    o3d.io.write_point_cloud(str(pcd_idx) + "_right_mark.pcd", pcd2)

    print("[frame " + str(pcd_idx) + "] mea left offset: ", left_offset)
    print("[frame " + str(pcd_idx) + "] mea left yaw: ", left_yaw)
    print("[frame " + str(pcd_idx) + "] mea left curvature: ", left_curvature)
    print("[frame " + str(pcd_idx) + "] mea left curvature ratio: ", left_curvature_ratio)

    print("[frame " + str(pcd_idx) + "] mea right offset: ", right_offset)
    print("[frame " + str(pcd_idx) + "] mea right yaw: ", right_yaw)
    print("[frame " + str(pcd_idx) + "] mea right curvature: ", right_curvature)
    print("[frame " + str(pcd_idx) + "] mea right curvature ratio: ", right_curvature_ratio, " \n")

    mea_left_error = measurement_left - np.dot(H, state_left)
    mea_right_error = measurement_right - np.dot(H, state_right)
    H_t = H.transpose()
    S = np.dot(np.dot(H, PMatPred), H_t) + R
    k = np.dot(np.dot(PMatPred, H_t), np.linalg.inv(S))
    left_offset, left_yaw, left_curvature, left_curvature_ratio = state_left_predict + np.dot(k, mea_left_error)
    right_offset, right_yaw, right_curvature, right_curvature_ratio = state_right_predict + np.dot(k, mea_right_error)
    P = (np.eye(4) - np.dot(k, H)) * PMatPred

    print("\n")
    print("[frame " + str(pcd_idx) + "] opt left offset: ", left_offset)
    print("[frame " + str(pcd_idx) + "] opt left yaw: ", left_yaw)
    print("[frame " + str(pcd_idx) + "] opt left curvature: ", left_curvature)
    print("[frame " + str(pcd_idx) + "] opt left curvature ratio: ", left_curvature_ratio)

    print("[frame " + str(pcd_idx) + "] opt right offset: ", right_offset)
    print("[frame " + str(pcd_idx) + "] opt right yaw: ", right_yaw)
    print("[frame " + str(pcd_idx) + "] opt right curvature: ", right_curvature)
    print("[frame " + str(pcd_idx) + "] opt right curvature ratio: ", right_curvature_ratio)
    print("[frame " + str(pcd_idx) + "] width by offsets: ", right_offset - left_offset, " \n")

    # update road width
    ego_yaw = (left_yaw + right_yaw) / 2
    ego_curvature = (left_yaw + right_curvature) / 2
    ego_curvature_ratio = (left_curvature_ratio + right_curvature_ratio) / 2
    mea_lane_width = road_width_update(xyz_init, mark_resolution,
                                       ego_yaw, ego_curvature, ego_curvature_ratio)
    if mea_lane_width != 0:
        # measurement update
        lane_width = mea_lane_width
    print("[frame " + str(pcd_idx) + "] Update width : ", lane_width, " \n")

    # plot lane
    lane_fitting = []
    if left_lane_positive:
        for z in range(0, int(z_max), 1):
            y1 = left_offset + left_yaw * z + 0.5 * left_curvature * z * z + left_curvature_ratio * z * z * z / 6
            lane_fitting.append(float(0))
            lane_fitting.append(float(y1))
            lane_fitting.append(float(z))

    if right_lane_positive:
        for z in range(0, int(z_max), 1):
            y2 = right_offset + right_yaw * z + 0.5 * right_curvature * z * z + right_curvature_ratio * z * z * z / 6
            lane_fitting.append(float(0))
            lane_fitting.append(float(y2))
            lane_fitting.append(float(z))

    lane_fitting_np = np.asarray(lane_fitting).reshape(-1, 3)
    mark_fitting_cloud = np.vstack((next_xyz, lane_fitting_np))

    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_fitting_np)
    o3d.io.write_point_cloud(str(pcd_idx) + "_lane.pcd", pcd2)

    PCD_static.points = o3d.utility.Vector3dVector(mark_fitting_cloud)
    vis.update_geometry(PCD_static)

    # view = vis.get_view_control()
    # print("Field of view (before changing) %.2f" % view.get_field_of_view())
    # view.rotate(90.0, 0.0)
    # print("Field of view (after changing) %.2f" % view.get_field_of_view())
    # vis.add_3d_label("{}".format(pcs_path + str(pcd_idx) + ".pcd"))

    vis.poll_events()
    vis.update_renderer()


def Key_up(vis):
    view = vis.get_view_control()
    view.translate(10, 10, 0, 0)


def Key_down(vis):
    view = vis.get_view_control()
    view.translate(10, 10, 0, 0)


if __name__ == "__main__":
    global lane_width, left_offset, left_yaw, left_curvature, left_curvature_ratio
    global right_offset, right_yaw, right_curvature, right_curvature_ratio
    global PCD_static, PCD_dynamic, Line_set, XYZ, KEY, IMG, pcs_path, pcd_idx

    mark_resolution = 0.1

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Lane Detection", 800, 800)
    view = vis.get_view_control()

    render = vis.get_render_option()

    render.background_color = [2, 2, 0]
    # render.show_coordinate_frame=True
    render.line_width = 5
    render.point_size = 5

    lane_width = 3.5
    left_offset = -lane_width / 2
    left_yaw = 0.02
    left_curvature = 0.0035

    right_offset = lane_width / 2
    right_yaw = 0.02
    right_curvature = 0.0035

    left_lane = []
    right_lane = []

    # State Space Mat Initialization
    P = np.array([[10, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0.001]])
    # print("P:", P)

    Q = np.eye(4)
    # print("Q:", Q)
    H = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])
    # print("H:", H)

    R = np.eye(3)
    # print("R:", R)

    dealt_t = 0.1
    v = 0
    dz = dealt_t * v
    A = np.asarray([[1, dz, 0.5 * dz * dz, dz * dz * dz / 6],
                    [0, 1, dz, 0.5 * dz * dz],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    pcd_idx = 1
    pcs_path = "./data/mark2/"
    pcd_name = pcs_path + str(pcd_idx) + ".pcd"
    pcd = o3d.io.read_point_cloud(pcd_name, format='pcd')
    xyz_init = np.asarray(pcd.points)

    # First Width Update
    print("frame " + str(pcd_idx) + " update lane width:")
    update_width = road_width_update(xyz_init, mark_resolution, 0, 0, 0)
    if update_width != 0:
        lane_width = update_width
    print("lane width:", lane_width)

    # assign mark points to lane
    z_max = 0
    for i, xyz in enumerate(xyz_init):
        point_x, point_y, point_z = xyz
        if 0 < point_z < 50:
            des_y_left = left_offset + left_yaw * point_z + 0.5 * left_curvature * point_z * point_z
            des_y_right = right_offset + right_yaw * point_z + 0.5 * right_curvature * point_z * point_z
            # print("[", des_y_left, " ", des_y_right, " ", point_z, " ", point_y, "]")
            if abs(des_y_left - point_y) < lane_width / 3:
                left_lane.append(xyz)
                # print(""point_z, " ", point_y)

            if abs(point_y - des_y_right) < lane_width / 3:
                right_lane.append(xyz)
                # print("[", point_z, " ", point_y, "]")
            if point_z > z_max:
                z_max = point_z

    # initial mark distribution line -1
    straight_line_save(z_max, lane_width, -1, left_yaw, left_curvature)

    left_lane_positive, left_offset, left_yaw, left_curvature = conic_hough(left_lane, 25, 20, 100,
                                                                            -2.5, 0,
                                                                            -1.0, 1.0,
                                                                            -0.006, 0.006)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(left_lane)
    o3d.io.write_point_cloud(str(pcd_idx) + "_left_mark.pcd", pcd)

    right_lane_positive, right_offset, right_yaw, right_curvature = conic_hough(right_lane, 25, 20, 100,
                                                                                0, 2.5,
                                                                                -1.0, 1.0,
                                                                                -0.006, 0.006)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(right_lane)
    o3d.io.write_point_cloud(str(pcd_idx) + "_right_mark.pcd", pcd1)

    left_curvature_ratio = 0
    right_curvature_ratio = 0

    print("\n")
    print("[frame " + str(pcd_idx) + "] opt left offset: ", left_offset)
    print("[frame " + str(pcd_idx) + "] opt left yaw: ", left_yaw)
    print("[frame " + str(pcd_idx) + "] opt left curvature: ", left_curvature)
    print("[frame " + str(pcd_idx) + "] opt left curvature ratio: ", left_curvature_ratio)

    print("[frame " + str(pcd_idx) + "] opt right offset: ", right_offset)
    print("[frame " + str(pcd_idx) + "] opt right yaw: ", right_yaw)
    print("[frame " + str(pcd_idx) + "] opt right curvature: ", right_curvature)
    print("[frame " + str(pcd_idx) + "] opt right curvature ratio: ", right_curvature_ratio)
    print("[frame " + str(pcd_idx) + "] width by offsets: ", right_offset - left_offset, " \n")

    # update road width
    ego_yaw = (left_yaw + right_yaw) / 2
    ego_curvature = (left_yaw + right_curvature) / 2
    ego_curvature_ratio = (left_curvature_ratio + right_curvature_ratio) / 2
    mea_lane_width = road_width_update(xyz_init, mark_resolution,
                                       ego_yaw, ego_curvature, ego_curvature_ratio)
    if mea_lane_width != 0:
        # measurement update
        lane_width = mea_lane_width

    # generate first frame lane
    lane_point_cloud = []

    for z in range(0, int(z_max), 1):
        y1 = left_offset + left_yaw * z + 0.5 * left_curvature * z * z \
             + left_curvature_ratio * z * z * z / 6
        lane_point_cloud.append(float(0))
        lane_point_cloud.append(float(y1))
        lane_point_cloud.append(float(z))

        y2 = right_offset + right_yaw * z + 0.5 * right_curvature * z * z \
             + right_curvature_ratio * z * z * z / 6
        lane_point_cloud.append(float(0))
        lane_point_cloud.append(float(y2))
        lane_point_cloud.append(float(z))

    lane_point_cloud_np = np.asarray(lane_point_cloud).reshape(-1, 3)
    all_point_cloud = np.vstack((xyz_init, lane_point_cloud_np))
    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_point_cloud_np)
    o3d.io.write_point_cloud(str(pcd_idx) + "_lane.pcd", pcd2)

    Line_set = point_grid(30)
    PCD_static = o3d.geometry.PointCloud()

    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[2, 4, 0])
    # vis.add_geometry(XYZ)
    PCD_static.points = o3d.utility.Vector3dVector(all_point_cloud)
    vis.add_geometry(PCD_static)
    vis.add_geometry(Line_set)
    vis.register_animation_callback(update)
    vis.run()
    vis.destroy_window()
