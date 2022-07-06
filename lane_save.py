import open3d as o3d
import numpy as np


def lines_save(z_max, width, idx, yaw, curvature):
    # initial mark distribution line -1
    ini_lane_point_cloud = []
    for z in range(0, int(z_max), 1):
        y1 = -width/2 + yaw * z + 0.5 * curvature * z * z
        ini_lane_point_cloud.append(float(0))
        ini_lane_point_cloud.append(float(y1))
        ini_lane_point_cloud.append(float(z))

        y2 = width/2 + yaw * z + 0.5 * curvature * z * z
        ini_lane_point_cloud.append(float(0))
        ini_lane_point_cloud.append(float(y2))
        ini_lane_point_cloud.append(float(z))

    lane_point_cloud_np = np.asarray(ini_lane_point_cloud).reshape(-1, 3)

    # save lane line
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lane_point_cloud_np)
    o3d.io.write_point_cloud(str(idx) + "_lane.pcd", pcd2)


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
