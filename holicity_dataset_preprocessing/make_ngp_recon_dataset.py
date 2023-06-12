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

class HoliCityDataset(Dataset):
    def __init__(self, rootdir, split, since_month=None):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/{split}-middlesplit.txt', dtype=str)

        self.filelist = [f'{rootdir}/{f}' for f in filelist]
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
        self.z_near = 1 # 1m
        self.z_far = 32 # 32m

    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f'mkdir -p {save_folder}')
        # os.system(f'mkdir -p {save_folder}/point_clouds')

    def save_data(self, idx, downsample=1):

        # def is_valid():
        #     for i in range(idx * 8, idx * 8 + 8):
        #         prefix = self.filelist[i]
        #         if i > idx * 8:
        #             assert(prefix[:-7] == self.filelist[i - 1][:-7])
        #         depth = np.load(f'{prefix}_dpth.npz')["depth"][..., 0]
        #         valid_map = (self.z_near <= depth) & (depth <= self.z_far)
        #         if valid_map.mean() < 0.1:
        #             return False
        #     return True
        
        # if not is_valid():
        #     print(idx, "not valid")
        #     return

        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))

        depth = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"][::downsample, ::downsample]
        image = Image.open(f"{self.filelist[idx][:-3]}.jpg").resize(
            depth.shape[::-1], resample=Image.Resampling.LANCZOS
        )
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

        pts = direction * depth[..., None]
        mask = (self.z_near <= depth) & (depth <= self.z_far)

        save_name = self.filelist[idx][:-3].split("/")[-1]
        # with open(f'{self.save_folder}/point_clouds/{save_name}.txt', 'w') as f:
        #     for (x, y, z), (r, g, b) in zip(pts[mask], np.array(image)[mask]):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        np.savez_compressed(f'{self.save_folder}/{save_name}.npz', coord=pts[mask], color=np.array(image)[mask])

        # all_pts = pts
        # for i in range(idx * 8, idx * 8 + 8):
        #     prefix = self.filelist[i]
        #     # print(idx + 1, '/', self.size, prefix)
        #     filename = prefix.split('/')[-1]
        #     basename = os.path.basename(prefix)


        #     with np.load(f'{camera_prefix}_camr.npz') as N:
        #         c2w = np.linalg.inv(N['R'])
        #         c2w[:3, 3] *= 0

        #     
            
        #     intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        #     u, v = np.meshgrid(np.arange(512)+0.5, np.arange(512)+0.5)
        #     pixel_exp = np.stack([depth] * 3) * np.stack([u, v, np.ones((512, 512))])
        #     pixel_exp = np.reshape(pixel_exp, (3, -1))
        #     coord_cam = np.linalg.inv(intrinsic_matrix).dot(pixel_exp)
        #     coord_cam[1] *= -1
        #     coord_cam[2] *= -1
        #     coord_wor = c2w[:3,:3].dot(coord_cam)

        #     pts = coord_wor.T[mask.flatten()]
        # np.savez_compressed(f'{self.save_folder}/coords/{filename}.npz', points=pts, mask=mask)

        
        
        # self.frames.append({
        #     "file_path": f"images/{filename}.jpg",
        #     "dep_path": f"depths/{filename}.npz",
        #     "sharpness": 30.0,
        #     "transform_matrix": c2w.tolist(),
        # })
    
    def save_data_resampling(self, idx, downsample=1):

        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))
        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        save_name = self.filelist[idx][:-3].split("/")[-1]

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        
        self.frames.append(np.array(cam_info["loc"]))
        a, b, c = cam_info["loc"]
        if a < -1500 and b < -1500:
            print(f"mv {save_name}.npz val")
        return

        
        depth = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"][::downsample, ::downsample]

        from scipy.ndimage.filters import maximum_filter, minimum_filter
        from scipy.ndimage.morphology import generate_binary_structure
        depth_local_max = maximum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth
        depth_local_min = minimum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth

        # colored_depth = plt.get_cmap("viridis")(depth_local_min.astype(np.float32)) # should be 0 to 1
        # Image.fromarray((colored_depth * 255.0).astype(np.uint8)).save(f"{self.save_folder}/{save_name}_localmax.png")
        # return

        image_org = Image.open(f"{self.filelist[idx][:-3]}.jpg")
        image = image_org.resize(
            depth.shape[::-1], resample=Image.Resampling.LANCZOS
        )
        image_org = np.array(image_org)
        
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

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(pts[mask])
        src_pc.estimate_normals()
        src_pc.normalize_normals()
        normals = np.asarray(src_pc.normals)
        mask[mask] = np.abs(normals[:, -1]) <= 0.98 # filter out ground points?

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
        is_not_ground = np.asarray(scene_mesh.triangle_normals)[:, 2] <= 0.98
        is_not_sharp = (angles(pts[np.asarray(scene_mesh.triangles)]) >= (10.0 / 180.0 * np.pi)).all(axis=-1)
        scene_mesh.triangles = o3d.utility.Vector3iVector(all_tri[tri_ok][is_not_ground & is_not_sharp])

        def normal(triangles):
            # The cross product of two sides is a normal vector
            return np.cross(triangles[:,1] - triangles[:,0], 
                            triangles[:,2] - triangles[:,0], axis=1)

        def surface_area(triangles):
            # The norm of the cross product of two sides is twice the area
            return np.linalg.norm(normal(triangles), axis=1) / 2

        area = np.sum(surface_area(pts[np.asarray(scene_mesh.triangles)]))
        n_pts = int(area * 100)
        if n_pts < 50000:
            print(save_name, "skip")
            return
        else:
            print(save_name, n_pts)
        pc = scene_mesh.sample_points_poisson_disk(n_pts)

        # with open(f'{self.save_folder}/{save_name}_orgpc.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(scene_voxel_grid.points, uv_rgb):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(pts)
        dist = pc.compute_point_cloud_distance(src_pc)

        np.savez_compressed(
            f'{self.save_folder}/{save_name}.npz',
            coord=np.asarray(pc.points),
            color=(np.asarray(pc.colors) * 255.0).astype(np.uint8),
            dist=dist,
        )
        
        # print(save_name)

        # coords = torch.from_numpy(pts[mask])
        # feats = torch.from_numpy(np.array(image).reshape((-1, 3))[mask])
        # area = np.sum(surface_area(pts[np.asarray(scene_mesh.triangles)]))
        
        # pc = scene_mesh.sample_points_poisson_disk(n_pts)

        # with open(f'{self.save_folder}/{save_name}_pc.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(np.asarray(pc.points) @ rot_mat, (np.asarray(pc.colors) * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        # def xyz2lonlat(coord):
        #     # coord: N, 3
        #     dist = np.linalg.norm(coord, axis=-1)
        #     normed_coord = coord / dist[..., np.newaxis]
        #     lat = np.arcsin(normed_coord[:, 2]) # -pi/2 to pi/2
        #     lon = np.arctan2(normed_coord[:, 0], normed_coord[:, 1]) # -pi to pi
        #     return lon, lat

        # def xyz2uv(coord, img_h, img_w):
        #     # coord: N, 3
        #     lon, lat = xyz2lonlat(coord)
        #     lat /= (torch.pi / 2.0) # -1 to 1, map to h to 0
        #     lon /= torch.pi # -1 to 1, map to, 0 to w
        #     u = (-img_h * lat + img_h) / 2.0
        #     v = (img_w * lon + img_w) / 2.0
        #     return np.floor(np.stack([u, v], axis=-1)).astype(np.int32)

        # import trimesh
        # scene_trimesh = trimesh.Trimesh(vertices=pts, faces=all_tri[tri_ok][is_not_ground])
        # scene_voxel_grid = scene_trimesh.voxelized(0.1)
        # uv = xyz2uv(scene_voxel_grid.points, image_org.shape[0], image_org.shape[1])
        # uv_rgb = image_org[
        #     np.clip(uv[:, 0], 0, image_org.shape[0] - 1),
        #     np.clip(uv[:, 1], 0, image_org.shape[1] - 1),
        # ]

        # with open(f'{self.save_folder}/{save_name}_voxelized.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(scene_voxel_grid.points, uv_rgb):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(scene_mesh, 1)
        # for voxel in voxel_grid.Voxels:
        #     print(grid_index)
        # non_ground_mesh = o3d.geometry.TriangleMesh(
        #     vertices=o3d.utility.Vector3dVector(pts),
        #     triangles=o3d.utility.Vector3iVector(all_tri[depth_ok][is_not_ground]),
        # )

        # normals_dir = np.sign(np.sum(normals * np.asarray(src_pc.points), axis=-1))
        # normals_redir = normals * normals_dir[..., np.newaxis]
        # normals_rgb = np.round(normals_redir * 127.5 + 127.5).astype(np.uint8)

        # sel = (normals_redir[..., 2] > 0.99)
        # print(normals[sel][:5])
        # print(pts[mask][sel][:5])
        # print(np.sum(normals * np.asarray(src_pc.points), axis=-1)[sel][:5])
        # print(normals_dir[sel][:5])
        # print(normals_redir[sel][:5])


        # with open(f'{self.save_folder}/{save_name}_masked.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(pts[mask] @ rot_mat, np.array(image).reshape((-1, 3))[mask]):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        # quit()

        # create_from_triangle_mesh(input, voxel_size)
        # 


        # coords = torch.cat([coords[:, :1] * 0, coords * 20], dim=-1) # 0.05m
        # feats = feats.float() / 127.5 - 1.0

        # pts_field = ME.TensorField(
        #     features=feats,
        #     coordinates=coords,
        #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        #     device=coords.device,
        # )
        # pts_sparse = pts_field.sparse()

        # pts_vis = pts_sparse
        # pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        # for k in range(4):
        #     with open(f"{self.save_folder}/{save_name}_rot_org_{k}.txt", "w") as f:
        #         for (_, x, y, z), (r, g, b) in zip(
        #             pts_vis.C.cpu().numpy(),
        #             (torch.clamp(pts_vis.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
        #         ):
        #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        #     pts_vis = pool(pts_vis)

        # with open(f'{self.save_folder}/{save_name}_rot_org.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(pts, np.array(image).reshape((-1, 3))):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        # quit()


        # with open(f'/cluster/project/cvg/zuoyue/holicity_point_cloud/512x256_resample_100_noground_index_200_220/{save_name}.txt', "w") as f:
        #     for (x, y, z), (r, g, b) in zip(np.asarray(pc.points), (np.asarray(pc.colors) * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        # colored_dist = plt.get_cmap("viridis")(np.clip(dist, 0.0, 1.0))
        # with open(f'/cluster/project/cvg/zuoyue/holicity_point_cloud/512x256_resample_100_noground_index_200_220/{save_name}_dist.txt', "w") as f:
        #     for (x, y, z), (r, g, b, _) in zip(np.asarray(pc.points), (colored_dist * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        return



if __name__ == '__main__':

    dataset = HoliCityDataset('/cluster/project/cvg/zuoyue/HoliCity', 'train', since_month='2018-01') # valid test
    # dataset.create_save_folder('/cluster/project/cvg/zuoyue/holicity_point_cloud/512x256_sphere_index_0_200')
    dataset.create_save_folder('/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_100_noground_index_0_220/')
    for i in tqdm.tqdm(list(range(int(sys.argv[1]), int(sys.argv[2])))): # len(dataset)
        dataset.save_data_resampling(i, downsample=1)
    # dataset.save_json()

    # pts = np.array(dataset.frames)
    # val_mask = (pts[:, 0] < -1500) & (pts[:, 1] < -1500)
    # plt.plot(pts[~val_mask, 0], pts[~val_mask, 1], "b*")
    # plt.plot(pts[val_mask, 0], pts[val_mask, 1], "r*")
    # plt.savefig("aaa.png")
