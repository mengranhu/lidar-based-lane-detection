import open3d as o3d
import numpy as np
from enum import Enum


class Type(Enum):
    LEFT = 1
    RIGHT = 2


def get_key(element):
    return element[2]


class Lane:

    def __init__(self, mark_idx, mark_xyz, mark_gap, mark_type):
        assert(isinstance(mark_xyz, np.ndarray))
        # assert (isinstance(mark_xyz, int))
        self.mark_xyz = mark_xyz
        self.mark_gap = mark_gap
        self.idx = mark_idx
        self.type = mark_type

    def mark_filtering(self):
        gap_list = []

        # for debug
        before_pcd = o3d.geometry.PointCloud()
        before_pcd.points = o3d.utility.Vector3dVector(self.mark_xyz)
        o3d.io.write_point_cloud(str(self.idx) + "_" + str(self.type) + "_origin.pcd", before_pcd)

        #  tolist()  # self.mark_xyz
        xyz_list = self.mark_xyz.tolist()
        xyz_list.sort(key=get_key)
        xyz_np = np.asarray(xyz_list)
        for idx, item in enumerate(xyz_np):
            if idx > 0 and item[2] - xyz_np[idx-1][2] > self.mark_gap:
                gap_list.append(idx)
        print("gap list:", gap_list)
        gap_list.sort(reverse=False)
        if len(gap_list) > 0:
            xyz_np = xyz_np[0:gap_list[0], :]
        self.mark_xyz = xyz_np

        # for debug
        after_pcd = o3d.geometry.PointCloud()
        after_pcd.points = o3d.utility.Vector3dVector(xyz_np)
        o3d.io.write_point_cloud(str(self.idx) + "_" + str(self.type) + "_filter.pcd", after_pcd)

        pt_num, _ = xyz_np.shape
        min_z = xyz_np[0][2]
        max_z = xyz_np[pt_num-1][2]
        return xyz_np, min_z, max_z

    def get_confidence(self):
        xyz_list = self.mark_xyz.tolist()  # self.mark_xyz
        xyz_list.sort(key=get_key)
        xyz_np = np.asarray(xyz_list)

    def update_mark(self, mark_xyz):
        assert (isinstance(mark_xyz, np.ndarray))
        self.mark_xyz = mark_xyz


if __name__ == "__main__":
    pcd_name = "100"
    pcd = o3d.io.read_point_cloud(pcd_name + ".pcd", format='pcd')
    # o3d.visualization.draw_geometries([pcd])
    xyz = np.asarray(pcd.points)

    p1 = Lane(1, xyz, 5, Type.LEFT)
    p1.mark_filtering()
    print(" type:", p1.type)
    num, _ = xyz.shape
    print("shape:", num)

