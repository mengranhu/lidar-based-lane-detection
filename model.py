import open3d as o3d
import numpy as np


def get_key(element):
    return element[2]


class Lane:
    def __init__(self, mark_xyz, mark_gap):
        self.mark_xyz = mark_xyz
        self.mark_gap = mark_gap

    def mark_filtering(self):
        gap_list = []
        xyz_list = self.mark_xyz.tolist()  # self.mark_xyz
        xyz_list.sort(key=get_key)
        xyz_np = np.asarray(xyz_list)
        for idx, item in enumerate(xyz_np):
            if idx > 0 and item[2] - xyz_np[idx-1][2] > self.mark_gap:
                gap_list.append(idx)
        print(gap_list)
        gap_list.sort(reverse=False)
        print(gap_list)
        xyz_np = xyz_np[0:gap_list[0], :]
        self.mark_xyz = xyz_np

        # for debug
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz_np)
        # o3d.io.write_point_cloud("filter.pcd", pcd)

    def get_confidence(self):
        xyz_list = self.mark_xyz.tolist()  # self.mark_xyz
        xyz_list.sort(key=get_key)
        xyz_np = np.asarray(xyz_list)


if __name__ == "__main__":
    pcd_name = "lane_mark"
    pcd = o3d.io.read_point_cloud(pcd_name + ".pcd", format='pcd')
    # o3d.visualization.draw_geometries([pcd])
    xyz = np.asarray(pcd.points)

    p1 = Lane(xyz, 5)
    # print(p1.mark_xyz)
    p1.mark_filtering()
    print(p1.mark_xyz)
    # print("11:", p1.mark_xyz[0][2])
