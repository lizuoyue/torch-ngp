import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import json
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
import glob

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
            # assert(os.path.exists(f"{self.filelist[i]}_dpth.npz"))

        self.frames = []
        self.z_near = 0.01 # 1cm
        self.z_far = 32 # 100m


    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f'mkdir -p {save_folder}')
        # os.system(f'mkdir -p {save_folder}/point_clouds')

    def save_data(self, idx, downsample=1, depth=1):

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
        # assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))

        image = Image.open(f"{self.filelist[idx][:-3]}.jpg")
        w, h = image.size
        new_w, new_h = w // downsample, h // downsample
        
        
        image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        depth = depth * np.ones((new_h, new_w), np.float32)

        func01 = lambda s: (np.arange(s) + 0.5) / s
        func11 = lambda s: func01(s) * 2 - 1
        lon, lat = np.meshgrid(func11(new_w) * np.pi, -func11(new_h) * np.pi / 2)

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
        # with open(f'{self.save_folder}/{save_name}.txt', 'w') as f:
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
    
    def save_data_new(self, idx, downsample=1, const_depth=None):

        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))
        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        filename = os.path.basename(self.filelist[idx])[:-3]

        save_name = self.filelist[idx][:-3].split("/")[-1]

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        
        self.frames.append(np.array(cam_info["loc"]))

        depth = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"][::downsample, ::downsample]
        postfix = ".jpg"
        if const_depth is not None:
            depth = depth * 0 + const_depth
            postfix = f"_{const_depth}.jpg"

        image_org = Image.open(f"{self.filelist[idx][:-3]}.jpg")
        image = image_org.resize(
            depth.shape[::-1], resample=Image.Resampling.LANCZOS
        )
        image_org = np.array(image_org)

        from vispy.util.transforms import rotate
        def panorama_to_world(pano_yaw, tilt_yaw, tilt_pitch):
            """Convert d \in S^2 (direction of a ray on the panorama) to the world space."""
            axis = np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
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
        # xy_dist = np.linalg.norm(pts[:, :2], axis=-1)
        # mask = ((self.z_near <= xy_dist) & (xy_dist <= self.z_far)).reshape((-1))
        # mask &= ~depth_local_max.reshape((-1))
        # mask &= ~depth_local_min.reshape((-1))
        rotate_pts = pts @ rot_mat
        color = np.array(image).reshape((-1, 3))

        pinhole_cams = sorted(glob.glob(f"{self.filelist[idx]}*camr.npz"))
        pinhole_deps = sorted(glob.glob(f"{self.filelist[idx]}*dpth.npz"))

        for pinhole_cam, pinhole_dep in zip(pinhole_cams, pinhole_deps):
            
            with np.load(pinhole_cam) as N:
                w2c = N["R"]
                c2w = np.linalg.inv(N["R"])
                c2w[:3, 3] *= 0

            cam_coord = rotate_pts.dot(w2c[:3, :3].T)
            depth_order = np.argsort(cam_coord[:, 2])
            cam_coord = cam_coord[depth_order]

            coord_sorted = rotate_pts[depth_order]
            color_sorted = color[depth_order]

            intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
            cam_coord[:, 1] *= -1
            cam_coord[:, 2] *= -1
            mask = cam_coord[:, 2] > 0

            cam_coord_div = cam_coord * 1.0
            cam_coord_div /= cam_coord[:, 2:]
            pix_coord = cam_coord_div.dot(intrinsic_matrix.T)
            pix_coord = np.clip(pix_coord, -1e8, 1e8)
            uv = np.floor(pix_coord[:, :2]).astype(np.int32)

            pad = 0
            mask &= ((0 - pad) <= uv[:, 0]) & (uv[:, 0] < (512 + pad))
            mask &= ((0 - pad) <= uv[:, 1]) & (uv[:, 1] < (512 + pad))

            u, v = uv[mask, 0], uv[mask, 1]
            img_recon = np.zeros((512, 512, 3), np.uint8)
            img_recon[v, u] = color_sorted[mask]
            to_save = os.path.basename(pinhole_cam).replace(".npz", postfix)
            Image.fromarray(img_recon).save(f"{self.save_folder}/{to_save}")

            print(np.load(pinhole_dep)["depth"][u, v, 0])
            print(cam_coord[mask, 2])

    def save_data_rot_correct(self, idx):

        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        filename = os.path.basename(self.filelist[idx])[:-3]

        save_name = self.filelist[idx][:-3].split("/")[-1]

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        
        from vispy.util.transforms import rotate
        def panorama_to_world_correct(pano_yaw, tilt_yaw, tilt_pitch):
            """Convert d \in S^2 (direction of a ray on the panorama) to the world space."""
            axis = np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
            R = (rotate(pano_yaw, [0, 0, 1]) @ rotate(tilt_pitch, axis))[:3, :3]
            return R
        
        def panorama_to_world_false(pano_yaw, tilt_yaw, tilt_pitch):
            """Convert d \in S^2 (direction of a ray on the panorama) to the world space."""
            axis = np.cross([np.cos(pano_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
            R = (rotate(pano_yaw, [0, 0, 1]) @ rotate(tilt_pitch, axis))[:3, :3]
            return R

        rot_mat_correct = panorama_to_world_correct(cam_info["pano_yaw"], cam_info["tilt_yaw"], cam_info["tilt_pitch"])
        rot_mat_false = panorama_to_world_false(cam_info["pano_yaw"], cam_info["tilt_yaw"], cam_info["tilt_pitch"])

        correct_mat = np.linalg.inv(rot_mat_false).dot(rot_mat_correct)
        np.savetxt(f"{self.save_folder}/{save_name}.txt", correct_mat)


if __name__ == '__main__':

    dataset = HoliCityDataset('/cluster/project/cvg/zuoyue/HoliCity', 'train', since_month='2018-01') # valid test
    dataset.create_save_folder('/cluster/project/cvg/zuoyue/holicity_point_cloud/rot_mat_correction')
    for i in tqdm.tqdm(list(range(0, 220))): # len(dataset)
        dataset.save_data_rot_correct(i)
        # dataset.save_data_new(i, downsample=32, const_depth=10)
        # dataset.save_data_new(i, downsample=32, const_depth=100)
    # dataset.save_json()
