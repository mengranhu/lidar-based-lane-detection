import open3d as o3d
import numpy as np


def init_two_lines_save(left_max_z, right_max_z, width, idx, yaw, curvature, pcs_path_out):
    # initial mark distribution line -1
    ini_lane_point_cloud = []
    for z in range(0, int(left_max_z), 1):
        y1 = -width / 2 + yaw * z + 0.5 * curvature * z * z
        ini_lane_point_cloud.append(float(0))
        ini_lane_point_cloud.append(float(y1))
        ini_lane_point_cloud.append(float(z))

    for z in range(0, int(right_max_z), 1):
        y2 = width / 2 + yaw * z + 0.5 * curvature * z * z
        ini_lane_point_cloud.append(float(0))
        ini_lane_point_cloud.append(float(y2))
        ini_lane_point_cloud.append(float(z))
    lane_point_cloud_np = np.asarray(ini_lane_point_cloud).reshape(-1, 3)
    print("lane_point_cloud_np:", lane_point_cloud_np)

    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_point_cloud_np)
    o3d.io.write_point_cloud(pcs_path_out + str(idx) + "_lane.pcd", pcd2)


def ego_lanes_save(left_min_z, left_max_z, right_min_z, right_max_z,
                   left_offset, left_yaw, left_curvature, left_curvature_ratio,
                   right_offset, right_yaw, right_curvature, right_curvature_ratio,
                   pcd_idx, pcs_path_out):
    lane_point_cloud = []
    for z in range(int(left_min_z), int(left_max_z), 1):
        y1 = left_offset + left_yaw * z + 0.5 * left_curvature * z * z \
             + left_curvature_ratio * z * z * z / 6
        lane_point_cloud.append(float(0))
        lane_point_cloud.append(float(y1))
        lane_point_cloud.append(float(z))

    for z in range(int(right_min_z), int(right_max_z), 1):
        y2 = right_offset + right_yaw * z + 0.5 * right_curvature * z * z \
             + right_curvature_ratio * z * z * z / 6
        lane_point_cloud.append(float(0))
        lane_point_cloud.append(float(y2))
        lane_point_cloud.append(float(z))

    lane_point_cloud_np = np.asarray(lane_point_cloud).reshape(-1, 3)
    # all_point_cloud = np.vstack((xyz_np, lane_point_cloud_np))
    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_point_cloud_np)
    o3d.io.write_point_cloud(pcs_path_out + str(pcd_idx) + "_lane.pcd", pcd2)
    return lane_point_cloud_np


def lane_save(xyz, offset, yaw, curvature, idx, pcd_name):
    ini_lane_point_cloud = []
    x_max, y_max, z_max = np.max(xyz, axis=0)
    # x_min, y_min, z_min = np.min(xyz, axis=0)

    for z in range(0, int(z_max), 1):
        y1 = offset + yaw * z + 0.5 * curvature * z * z
        ini_lane_point_cloud.append(float(0))
        ini_lane_point_cloud.append(float(y1))
        ini_lane_point_cloud.append(float(z))
    lane_point_cloud_np = np.asarray(ini_lane_point_cloud).reshape(-1, 3)

    # lane_point_cloud_np = np.vstack((xyz, lane_point_cloud_np))

    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_point_cloud_np)
    o3d.io.write_point_cloud(pcd_name + "_" + str(idx) + "_lane_"
                             + str(offset) + "_" + str(yaw) + "_" + str(curvature) + ".pcd", pcd2)
