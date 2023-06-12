import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import json
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import sys

def dep_to_cam_coord(dep):
    y = np.arange(dep.shape[0])
    x = np.arange(dep.shape[1])
    x, y = np.meshgrid(x, y)
    px = (x - 255.5) / 256
    py = (-y + 255.5) / 256
    pz = dep[..., 0]
    return np.dstack([px * pz, -py * pz, pz])

def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]

def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]

class HoliCityDataset(Dataset):
    def __init__(self, rootdir, split, since_month=None):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/{split}-middlesplit.txt', dtype=str)

        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist.sort()
        if since_month is not None:
            take_month = lambda s: s.split('/')[-2]
            self.filelist = [f for f in self.filelist if take_month(f) >= since_month]

        self.size = len(self.filelist)
        print(f'num {split}:', self.size)
        for i in range(self.size):
            assert(os.path.exists(f"{self.filelist[i][:-3]}.jpg"))
            assert(os.path.exists(f"{self.filelist[i]}_dpth.npz"))

        self.frames = []
        self.z_near = 0 # 0m
        self.z_far = 32 # 32m
        self.sem_path = "/cluster/project/cvg/zuoyue/ViT-Adapter/segmentation/demo"

    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f'mkdir -p {save_folder}')
    
    def save_data_pano_sem_resampling(self, idx, downsample=1):

        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))
        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        filename = os.path.basename(self.filelist[idx])[:-3]
        assert(os.path.exists(f"{self.sem_path}/{filename}.npz"))

        save_name = self.filelist[idx][:-3].split("/")[-1]

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        
        self.frames.append(np.array(cam_info["loc"]))

        depth = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"][::downsample, ::downsample]

        from scipy.ndimage.filters import maximum_filter, minimum_filter
        from scipy.ndimage.morphology import generate_binary_structure
        depth_local_max = maximum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth
        depth_local_min = minimum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth

        image_org = Image.open(f"{self.filelist[idx][:-3]}.jpg")
        image = image_org.resize(
            depth.shape[::-1], resample=Image.Resampling.LANCZOS
        )
        image_org = np.array(image_org)
        sem = np.load(f"{self.sem_path}/{filename}.npz")["seg"]
        
        from vispy.util.transforms import rotate
        def panorama_to_world(pano_yaw, tilt_yaw, tilt_pitch):
            """Convert d \in S^2 (direction of a ray on the panorama) to the world space."""
            axis = np.cross([np.cos(pano_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
            R = (rotate(pano_yaw, [0, 0, 1]) @ rotate(tilt_pitch, axis))[:3, :3]
            return R

        rot_mat = panorama_to_world(cam_info["pano_yaw"], cam_info["tilt_yaw"], cam_info["tilt_pitch"])

        h, w = depth.shape
        func01 = lambda s: (np.arange(s) + 0.5) / s
        func11 = lambda s: func01(s) * 2 - 1
        lon, lat = np.meshgrid(func11(w) * np.pi, -func11(h) * np.pi / 2)

        # right x, inside y, up z
        direction = np.dstack(
            [
                np.cos(lat) * np.sin(lon),
                np.cos(lat) * np.cos(lon),
                np.sin(lat),
            ]
        )

        pts = (direction * depth[..., None]).reshape((-1, 3))
        xy_dist = np.linalg.norm(pts[:, :2], axis=-1)
        mask = ((self.z_near <= xy_dist) & (xy_dist <= self.z_far)).reshape((-1))
        mask &= ~depth_local_max.reshape((-1))
        mask &= ~depth_local_min.reshape((-1))

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        idx = w * y + x
        idx_up_left = idx[:-1]
        idx_up_right = np.roll(idx_up_left, -1, axis=1)
        idx_down_left = idx[1:]
        idx_down_right = np.roll(idx_down_left, -1, axis=1)

        if True:
            upper = np.stack([idx_up_left, idx_up_right, idx_down_left], axis=-1).reshape((-1, 3))
            lower = np.stack([idx_down_left, idx_up_right, idx_down_right], axis=-1).reshape((-1, 3))
            all_tri = np.concatenate([upper, lower], axis=0)
            # ---
            # |/|
            # ---
        else:
            upper = np.stack([idx_up_left, idx_up_right, idx_down_right], axis=-1).reshape((-1, 3))
            lower = np.stack([idx_down_left, idx_up_left, idx_down_right], axis=-1).reshape((-1, 3))
            all_tri = np.concatenate([upper, lower], axis=0)
            # ---
            # |\|
            # ---

        tri_ok = mask[all_tri].all(axis=-1)
        scene_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(pts),
            triangles=o3d.utility.Vector3iVector(all_tri[tri_ok]),
        )
        scene_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(image).reshape((-1, 3)) / 255.0)
        scene_mesh.compute_triangle_normals()

        def angle(triangles, idx):
            # The cross product of two sides is a normal vector
            v1 = triangles[:, (idx + 1) % 3] - triangles[:, idx]
            v2 = triangles[:, (idx + 2) % 3] - triangles[:, idx]
            norm1 = np.linalg.norm(v1, axis=-1)
            norm2 = np.linalg.norm(v2, axis=-1)
            return np.arccos(np.sum(v1 * v2, axis=-1) / (norm1 * norm2)) # in range [0, pi]
        
        def angles(triangles):
            return np.stack([angle(triangles, 0), angle(triangles, 1), angle(triangles, 2)], axis=1)

        # TODO: add more constraint?
        is_not_sharp = (angles(pts[np.asarray(scene_mesh.triangles)]) >= (5.0 / 180.0 * np.pi)).all(axis=-1)
        scene_mesh.triangles = o3d.utility.Vector3iVector(all_tri[tri_ok][is_not_sharp])

        def normal(triangles):
            # The cross product of two sides is a normal vector
            return np.cross(triangles[:,1] - triangles[:,0], 
                            triangles[:,2] - triangles[:,0], axis=1)

        def surface_area(triangles):
            # The norm of the cross product of two sides is twice the area
            return np.linalg.norm(normal(triangles), axis=1) / 2

        area = np.sum(surface_area(pts[np.asarray(scene_mesh.triangles)]))
        n_pts = int(area * 400)
        if n_pts < 200000:
            print(save_name, "skip")
            return
        else:
            print(save_name, n_pts)

        pc = scene_mesh.sample_points_poisson_disk(n_pts)
        pc.estimate_normals()
        pc.normalize_normals()
        pc_ground_mask = np.abs(np.asarray(pc.normals)[:, 2]) > 0.98
        pc_non_ground_mask = ~pc_ground_mask

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(pts)
        dist = pc.compute_point_cloud_distance(src_pc)

        def xyz2lonlat(coord):
            # coord: N, 3
            dist = np.linalg.norm(coord, axis=-1)
            normed_coord = coord / dist[..., np.newaxis]
            lat = np.arcsin(normed_coord[:, 2]) # -pi/2 to pi/2
            lon = np.arctan2(normed_coord[:, 0], normed_coord[:, 1]) # -pi to pi
            return lon, lat

        def xyz2uv(coord, img_h, img_w):
            # coord: N, 3
            lon, lat = xyz2lonlat(coord)
            lat /= (torch.pi / 2.0) # -1 to 1, map to h to 0
            lon /= torch.pi # -1 to 1, map to, 0 to w
            u = (-img_h * lat + img_h) / 2.0
            v = (img_w * lon + img_w) / 2.0
            return np.floor(np.stack([u, v], axis=-1)).astype(np.int32)

        uv = xyz2uv(np.asarray(pc.points), sem.shape[0], sem.shape[1])
        pc_semantics = sem[
            np.clip(uv[:, 0], 0, sem.shape[0] - 1),
            np.clip(uv[:, 1], 0, sem.shape[1] - 1),
        ]
        rotate_pc_points = np.asarray(pc.points) @ rot_mat

        np.savez_compressed(
            f"{self.save_folder}/{save_name}.npz",
            coord=rotate_pc_points,
            color=(np.asarray(pc.colors) * 255.0).astype(np.uint8),
            dist=dist,
            sem=pc_semantics,
            geo_is_not_ground=pc_non_ground_mask,
        )
        return

        with open(f"{self.save_folder}/{save_name}.txt", "w") as f:
            for (x, y, z), (r, g, b) in zip(rotate_pc_points, (np.asarray(pc.colors) * 255.0).astype(np.uint8)):
                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        pc_semantics_colored = np.array(cityscapes_palette())[pc_semantics]
        with open(f"{self.save_folder}/{save_name}_sem.txt", "w") as f:
            for (x, y, z), (r, g, b) in zip(rotate_pc_points, pc_semantics_colored.astype(np.uint8)):
                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        pc_cls_colored = np.array(cityscapes_palette())[pc_non_ground_mask.astype(np.int32) + 1]
        with open(f"{self.save_folder}/{save_name}_binary.txt", "w") as f:
            for (x, y, z), (r, g, b) in zip(rotate_pc_points, pc_cls_colored.astype(np.uint8)):
                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        pc_dist_colored = plt.get_cmap("viridis")(np.clip(dist, 0.0, 0.5) * 2)
        with open(f"{self.save_folder}/{save_name}_dist.txt", "w") as f:
            for (x, y, z), (r, g, b, _) in zip(rotate_pc_points, (pc_dist_colored * 255.0).astype(np.uint8)):
                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        return



if __name__ == "__main__":

    dataset = HoliCityDataset("/cluster/project/cvg/zuoyue/HoliCity", sys.argv[1], since_month="2018-01") # train valid test
    dataset.create_save_folder(f"/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400new_index_0_220/{sys.argv[1]}")
    for i in tqdm.tqdm(list(range(int(sys.argv[2]), int(sys.argv[3])))): # len(dataset)
        dataset.save_data_pano_sem_resampling(i, downsample=1)
