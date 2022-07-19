import open3d as o3d
import numpy as np
from enum import Enum


class Cluster:
    def __init__(self, frame_idx, cluster_xyz):
        self.frame_idx = frame_idx
        self.cluster_xyz = cluster_xyz
        self.len = len(cluster_xyz)
        self.min_y = float("inf")
        self.max_y = float("-inf")
        self.min_z = float("inf")
        self.max_z = float("-inf")

    def init(self):
        self.cluster_xyz = self.cluster_xyz[self.cluster_xyz[:, 2].argsort()]
        self.min_z = self.cluster_xyz[0][2]
        self.max_z = self.cluster_xyz[len(self.cluster_xyz) - 1][2]
        for pt_idx, pt_xyz in enumerate(self.cluster_xyz):
            pt_x, pt_y, pt_z = pt_xyz
            if pt_y < self.min_y:
                self.min_y = pt_y
            if pt_y > self.max_y:
                self.max_y = pt_y


class Lane:
    def __init__(self, frame_idx, cluster_idx, cluster_xyz, arc_len):
        self.frame_idx = frame_idx
        self.cluster_id = cluster_idx
        self.confidence = 0.5  # 置信度
        self.para = []  # 列表
        self.scope = []
        self.arc_len = arc_len
        self.now_xyz = cluster_xyz
        self.next_xyz = np.ones(3)
        self.update = False
        self.is_first_mark = True
        self.mark_resolution = 0.5
        # State Space Model
        self.Q_Mat = np.eye(4)
        self.P_Mat = np.array([[10, 0, 0, 0],
                               [0, 5, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0.001]])
        self.dealt_t = 0.1
        self.v = 0
        self.dz = self.dealt_t * self.v
        self.A_Mat = np.asarray([[1, self.dz, 0.5 * self.dz * self.dz, self.dz * self.dz * self.dz / 6],
                                 [0, 1, self.dz, 0.5 * self.dz * self.dz],
                                 [0, 0, 1, self.dz],
                                 [0, 0, 0, 1]])

    def add_xyz(self, next_xyz):
        self.update = True
        if self.is_first_mark:
            self.next_xyz = next_xyz
            self.is_first_mark = False
        else:
            self.next_xyz = np.vstack((self.next_xyz, next_xyz))

    def pcd_save(self):
        last = o3d.geometry.PointCloud()
        last.points = o3d.utility.Vector3dVector(self.now_xyz)
        o3d.io.write_point_cloud(str(self.cluster_id) + "_last.pcd", last)

        if self.update:
            now = o3d.geometry.PointCloud()
            now.points = o3d.utility.Vector3dVector(self.next_xyz)
            o3d.io.write_point_cloud(str(self.cluster_id) + "_now.pcd", now)

    def line_save(self):
        z = np.linspace(self.scope[2] - 5, self.scope[3] + 5, 50)
        y2 = self.para[0] * z ** 3 + self.para[1] * z ** 2 + self.para[2] * z + self.para[3]
        line_np = np.hstack((np.ones((y2.shape[-1], 1)), y2[:, np.newaxis]))
        line_np = np.hstack((line_np, z[:, np.newaxis]))
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(line_np)
        o3d.io.write_point_cloud(str(self.cluster_id) + "_line.pcd", line_pcd)

    def update_next_scope(self):
        next_cluster_xyz = self.next_xyz[self.next_xyz[:, 2].argsort()]
        max_z = next_cluster_xyz[len(next_cluster_xyz) - 1][2]
        min_z = next_cluster_xyz[0][2]
        max_interval = max_z - min_z
        min_y = float("inf")
        max_y = float("-inf")
        cluster_mark_dim = int(max_interval / self.mark_resolution) + 1
        slice_list = [[] for i in range(cluster_mark_dim)]
        for pt_idx, pt_xyz in enumerate(next_cluster_xyz):
            pt_x, pt_y, pt_z = pt_xyz
            slice_idx = int((pt_z - min_z) / self.mark_resolution)
            slice_list[slice_idx].append(pt_xyz)
            if pt_y < min_y:
                min_y = pt_y
            if pt_y > max_y:
                max_y = pt_y
        self.scope = [min_y, max_y, min_z, max_z]
        return slice_list

    def update_next_arc_len(self, slice_list):
        assert len(slice_list) >= 2
        anchor_list = []
        for slice_idx, pt_slice in enumerate(slice_list):
            np_pt_slice = np.asarray(pt_slice)
            if len(pt_slice) == 0:
                continue
            anchor_list.append(np_pt_slice.mean(axis=0))

        # Calculate arc length
        next_arc_len = 0
        for anchor_index in range(len(anchor_list) - 1):
            pt1 = [anchor_list[anchor_index][1], anchor_list[anchor_index][2]]
            pt2 = [anchor_list[anchor_index + 1][1], anchor_list[anchor_index + 1][2]]
            next_arc_len = next_arc_len + np.linalg.norm(np.asarray(pt1) - np.asarray(pt2))
        dealt_arc_len = next_arc_len - self.arc_len
        self.arc_len = next_arc_len
        return dealt_arc_len, anchor_list

    def update_next_para(self, anchor_list):
        z = np.asarray(anchor_list)[:, 2]
        y = np.asarray(anchor_list)[:, 1]
        if self.arc_len >= 3.0 and len(z) >= 2:
            max_interval = self.scope[3] - self.scope[2]
            if max_interval < 30.0:
                para = np.polyfit(z, y, 2)
                para = np.hstack((0, para))
            else:
                para = np.polyfit(z, y, 3)
        self.para = para

    def predict(self):
        # offset, yaw, curvature, curvature_ratio
        offset = self.para[3]
        yaw = self.para[2]
        C0 = self.para[1] * 2
        C1 = self.para[0] * 6
        state = np.array([offset, yaw, C0, C1])
        state_predict = np.dot(self.A_Mat, state)
        offset, yaw, C0, C1 = state_predict
        self.para = [C1/6.0, C0/2.0, yaw, offset]
        self.P_Mat = self.A_Mat * self.P_Mat * self.A_Mat.transpose() + self.Q_Mat
        y_update = self.para[0] * self.now_xyz[:, 2] ** 3 + self.para[1] * self.now_xyz[:, 2] ** 2
        + self.para[2] * self.now_xyz[:, 2] + self.para[3]
        xy_update = np.hstack((self.now_xyz[:, 0], y_update))
        self.next_xyz = np.hstack(xy_update, self.now_xyz[:, 2])

    def update(self):
        if self.update:
            slice_list = self.update_next_scope()
            dealt_arc_len, anchor_list = self.update_next_arc_len(slice_list)
            self.update_next_para(anchor_list)
            self.confidence = self.confidence + 0.1 + dealt_arc_len / 20
            if self.confidence >= 1.0:
                self.confidence = 1.0

        else:
            # state、next_xyz & para update
            self.predict()
            slice_list = self.update_next_scope()
            self.update_next_arc_len(slice_list)
            # self.update_next_para(anchor_list)
            self.confidence = self.confidence - 0.1
            if self.confidence <= 0.0:
                self.confidence = 0.0

        self.now_xyz = self.next_xyz
        self.next_xyz = np.ones(3)
        self.update = False
        self.is_first_mark = True


# def check_validation_(Lane, xyz):

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("21_now.pcd")
    xyz_np = np.asarray(pcd.points)
    # o3d.visualization.draw_geometries([pcd])
    # line = Lane(1, 999, xyz_np, 14.0)
    # line.add_xyz(xyz_np)
    # line.poly_fit()
    # line.line_save()

