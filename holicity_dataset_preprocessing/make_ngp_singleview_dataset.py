import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import json
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
import open3d as o3d

def dep_to_cam_coord(dep):
    y = np.arange(dep.shape[0])
    x = np.arange(dep.shape[1])
    x, y = np.meshgrid(x, y)
    px = (x - 255.5) / 256
    py = (-y + 255.5) / 256
    pz = dep[..., 0]
    return np.dstack([px * pz, -py * pz, pz])

class HoliCityDataset(Dataset):
    def __init__(self, rootdir, split, since_month=None):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/filelist.txt', dtype=str)
        filter_ = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/{split}-middlesplit.txt', dtype=str)
        length = len(filter_[0])
        filter_ = set(filter_)

        self.filelist = [f'{rootdir}/image-v1/{f}' for f in filelist if f[:length] in filter_]
        self.filelist.sort()
        if since_month is not None:
            take_month = lambda s: s.split('/')[2]
            self.filelist = [f for f in self.filelist if take_month(f) >= since_month]

        self.size = len(self.filelist)
        print(f'num {split}:', self.size)

        self.frames = []
        self.z_near = 0.05 # 5cm
        self.z_far = 32 # 100m

    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f'mkdir -p {save_folder}')
        os.system(f'mkdir -p {save_folder}/images')
        os.system(f'mkdir -p {save_folder}/depths')

    def save_single_data(self, idx):
        prefix = self.filelist[idx]
        # print(idx + 1, '/', self.size, prefix)
        filename = prefix.split('/')[-1]
        basename = os.path.basename(prefix)
        camera_prefix = prefix.replace('image', 'camr')
        depth_prefix = prefix.replace('image', 'depth')

        img_pil = Image.open(f'{prefix}_imag.jpg')
        img_pil.save(f'{self.save_folder}/images/{filename}.jpg')

        with np.load(f'{camera_prefix}_camr.npz') as N:
            c2w = np.linalg.inv(N['R'])
            c2w[:3, 3] *= 0
        
        depth = np.load(f'{depth_prefix}_dpth.npz')["depth"][..., 0]
        if (depth.max() < self.z_near) or (depth.min() > self.z_far):
            print(prefix, 'invalid')
            np.savez_compressed(f'{self.save_folder}/depths/{filename}.npz', depth=depth[..., np.newaxis] * 0 + 1)
        else:
            os.system(f'cp {depth_prefix}_dpth.npz {self.save_folder}/depths/{filename}.npz')

        mask = (self.z_near < depth) & (depth < self.z_far)
        
        intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        u, v = np.meshgrid(np.arange(512)+0.5, np.arange(512)+0.5)
        pixel_exp = np.stack([depth] * 3) * np.stack([u, v, np.ones((512, 512))])
        pixel_exp = np.reshape(pixel_exp, (3, -1))
        coord_cam = np.linalg.inv(intrinsic_matrix).dot(pixel_exp)
        coord_cam[1] *= -1
        coord_cam[2] *= -1
        coord_wor = c2w[:3,:3].dot(coord_cam)

        pts = coord_wor.T[mask.flatten()]
        # np.savez_compressed(f'{self.save_folder}/coords/{filename}.npz', points=pts, mask=mask)

        # with open(f'vis/{filename}.txt', 'w') as f:
        #     for (x, y, z), (r, g, b) in zip(pts, np.array(img_pil)[mask]):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        self.frames.append({
            "file_path": f"images/{filename}.jpg",
            "dep_path": f"depths/{filename}.npz",
            "sharpness": 30.0,
            "transform_matrix": c2w.tolist(),
        })
    
    def test_function(self, idx):
        prefix = self.filelist[idx]

        filename = prefix.split('/')[-1]

        if filename not in {
            "0_TyQi6Ac2pLTtMjLtzR2A_LD_320_41",
            "18qmhehLX2ZBO6WblLWjDA_LD_229_44",
            "44G-hA5GM_ukz0SXCrtjwQ_LD_320_43",
            "7iLxnxu38hcG3huM5Wde_g_LD_259_09",
        }:
            return

        basename = os.path.basename(prefix)
        camera_prefix = prefix.replace('image', 'camr')
        depth_prefix = prefix.replace('image', 'depth')
        normal_prefix = prefix.replace('image', 'normal')

        img = np.array(Image.open(f'{prefix}_imag.jpg'))
        # img_pil.save(f'{self.save_folder}/images/{filename}.jpg')

        with np.load(f'{camera_prefix}_camr.npz') as N:
            c2w = np.linalg.inv(N['R'])
            c2w[:3, 3] *= 0
        
        depth = np.load(f'{depth_prefix}_dpth.npz')["depth"][..., 0]
        print(depth[depth>0].min(), depth.max())
        # if (depth.max() < self.z_near) or (depth.min() > self.z_far):
        #     print(prefix, 'invalid')
        #     np.savez_compressed(f'{self.save_folder}/depths/{filename}.npz', depth=depth[..., np.newaxis] * 0 + 1)
        # else:
        #     os.system(f'cp {depth_prefix}_dpth.npz {self.save_folder}/depths/{filename}.npz')

        # mask = (self.z_near < depth) & (depth < self.z_far)
        
        intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        u, v = np.meshgrid(np.arange(512)+0.5, np.arange(512)+0.5)
        pixel_exp = np.stack([depth] * 3) * np.stack([u, v, np.ones((512, 512))])
        pixel_exp = np.reshape(pixel_exp, (3, -1))
        coord_cam = np.linalg.inv(intrinsic_matrix).dot(pixel_exp)
        coord_cam[1] *= -1
        coord_cam[2] *= -1
        coord_wor = c2w[:3,:3].dot(coord_cam).T

        normal = np.load(f'{normal_prefix}_nrml.npz')["normal"].reshape((-1, 3)).T
        normal = c2w[:3,:3].dot(normal).T
        up_vec = np.array([0.0, 0.0, 1.0])
        is_ground = ((normal @ up_vec) > 0.99) & (coord_wor[:, 2] < 0)


        dist = np.sqrt((coord_wor ** 2).sum(axis=-1))
        dist_ok = (self.z_near < dist) & (dist < self.z_far)

        nrow, ncol = 512, 512

        x, y = np.meshgrid(np.arange(ncol), np.arange(nrow))
        idx = ncol * y + x
        upper = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], axis=-1).reshape((-1, 3))
        lower = np.stack([idx[1:, 1:], idx[:-1, 1:], idx[1:, :-1]], axis=-1).reshape((-1, 3))
        all_tri = np.concatenate([upper, lower], axis=0)

        same_plane = normal[all_tri]
        same_plane = np.stack([
            (same_plane[:, i] * same_plane[:, j]).sum(axis=-1) > 0.999
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ], axis=-1)
        same_plane = same_plane.all(axis=-1)

        very_close = coord_wor[all_tri]
        very_close = np.stack([
            np.sqrt(((very_close[:, i] - very_close[:, j]) ** 2).sum(axis=-1)) < 0.1
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ], axis=-1)
        very_close = very_close.all(axis=-1)

        within_range = dist_ok[all_tri].all(axis=-1)
        ground_surface = is_ground[all_tri].all(axis=-1)

        non_ground_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(coord_wor),
            triangles=o3d.utility.Vector3iVector(all_tri[(~ground_surface) & (same_plane | very_close) & within_range]),
        )
        ground_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(coord_wor),
            triangles=o3d.utility.Vector3iVector(all_tri[ground_surface & (same_plane | very_close) & within_range]),
        )

        non_ground_mesh.vertex_colors = o3d.utility.Vector3dVector(img.reshape((-1, 3)) / 255.0)
        ground_mesh.vertex_colors = o3d.utility.Vector3dVector(img.reshape((-1, 3)) / 255.0)

        n_pts = int(non_ground_mesh.get_surface_area() * 100)
        non_ground_pc = non_ground_mesh.sample_points_poisson_disk(n_pts) if n_pts > 0 else None

        n_pts = int(ground_mesh.get_surface_area() * 16)
        ground_pc = ground_mesh.sample_points_poisson_disk(n_pts) if n_pts > 0 else None

        for pc, mode in zip([ground_pc, non_ground_pc], ["w", "a"]):
            if pc is None:
                continue
            with open(f'/home/lzq/lzy/torch-ngp/data/holicity_single_view/point_clouds/{filename}.txt', mode) as f:
                for (x, y, z), (r, g, b) in zip(np.asarray(pc.points), (np.asarray(pc.colors) * 255.0).astype(np.uint8)):
                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        return


    
    def save_json(self):
        d = {
            "fl_x": 256.0,
            "fl_y": 256.0,
            "cx": 256.0,
            "cy": 256.0,
            "w": 512.0,
            "h": 512.0,
            "aabb_scale": 4,
            "z_near": self.z_near,
            "z_far": self.z_far,
            "frames": self.frames,
        }

        with open(f'{self.save_folder}/transforms.json', 'w') as f:
            f.write(json.dumps(d, indent=2))


if __name__ == '__main__':

    dataset = HoliCityDataset('.', 'train', since_month='2018-01') # valid test
    # dataset.create_save_folder('/home/lzq/lzy/torch-ngp/data/holicity_single_view')
    for i in tqdm.tqdm(list(range(1600))): # len(dataset)
        # dataset.save_single_data(i)
        dataset.test_function(i)
    # dataset.save_json()
