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

class HoliCityDataset(Dataset):
    def __init__(self, rootdir, pointclouddir):
        self.rootdir = rootdir
        self.pointclouddir = pointclouddir

        self.name_to_phase = {}
        for phase in ["train", "val"]:
            with open(f"{pointclouddir}/{phase}.txt") as f:
                lines = [line.strip().split()[-1][:-4] for line in f.readlines()[1:]]
                for line in lines:
                    self.name_to_phase[line] = phase

        filelist = np.genfromtxt(f"{rootdir}/split-all-v1-bugfix/filelist.txt", dtype=str)
        self.take_name = lambda f: os.path.basename(f)[:22]

        self.filelist = [f"{rootdir}/{f}" for f in filelist if self.take_name(f) in self.name_to_phase]
        self.filelist.sort(key=lambda f: (self.name_to_phase[self.take_name(f)], f))

        self.size = len(self.filelist)
        print(f"num dataset:", self.size)

        self.frames = []
        self.z_near = 0.05 # 5cm
        self.z_far = 32 # 100m

        self.buffer_name = ""
        self.buffer_data = None

    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f"mkdir -p {save_folder}")
        os.system(f"mkdir -p {save_folder}/images")
        os.system(f"mkdir -p {save_folder}/depths")
        os.system(f"mkdir -p {save_folder}/point_clouds")
        os.system(f"mkdir -p {save_folder}/images_proj")

    def save_single_data(self, idx):
        prefix = self.filelist[idx]
        name = self.take_name(prefix)
        phase = self.name_to_phase[name]

        # print(idx + 1, "/", self.size, prefix)
        filename = prefix.split("/")[-1]
        basename = os.path.basename(prefix)

        depth = np.load(f"{prefix}_dpth.npz")["depth"][..., 0]
        valid_map = (self.z_near <= depth) & (depth <= self.z_far)
        if valid_map.mean() < 0.1:
            print(prefix, "invalid")
            # np.savez_compressed(f"{self.save_folder}/depths/{filename}.npz", depth=depth[..., np.newaxis] * 0 + 0.2)
            return

        os.system(f"ln -s {prefix}_imag.jpg {self.save_folder}/images/{filename}.jpg")
        os.system(f"ln -s {prefix}_dpth.npz {self.save_folder}/depths/{filename}.npz")

        with np.load(f"{prefix}_camr.npz") as N:
            w2c = N["R"]
            c2w = np.linalg.inv(N["R"])
            c2w[:3, 3] *= 0
        
        if name != self.buffer_name:
            with np.load(f"{self.pointclouddir}/{phase}/{name}.npz") as P:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(P["coord"])
                pc.estimate_normals()
                pc.normalize_normals()
                normal = np.asarray(pc.normals)
                normal *= np.sign((normal * P["coord"]).sum(axis=-1)[..., np.newaxis])
                normal_rgb = (normal * 127.5 + 127.5).astype(np.uint8)

                # with open(f"{self.save_folder}/{name}.txt", "w") as f:
                #     
                #     for (x, y, z), (r, g, b) in zip(P["coord"], normal_rgb):
                #         f.write("%.3lf %.3lf %.3lf %d %d %d\n" % (x, y, z, r, g, b))
                #     input("check")

                self.buffer_name = name
                self.buffer_data = (P["coord"], P["color"], normal, normal_rgb)
        
        coord, color, normal, normal_rgb = self.buffer_data

        cam_coord = coord.dot(w2c[:3, :3].T)
        depth_order = np.argsort(cam_coord[:, 2])
        cam_coord = cam_coord[depth_order]

        coord_sorted = coord[depth_order]
        color_sorted = color[depth_order]
        normal_sorted = normal[depth_order]
        normal_rgb_sorted = normal_rgb[depth_order]

        intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        cam_coord[:, 1] *= -1
        cam_coord[:, 2] *= -1
        mask = cam_coord[:, 2] > 0

        cam_coord /= cam_coord[:, 2:]
        pix_coord = cam_coord.dot(intrinsic_matrix.T)
        pix_coord = np.clip(pix_coord, -1e8, 1e8)
        uv = np.floor(pix_coord[:, :2]).astype(np.int32)

        pad = 0
        mask &= ((0 - pad) <= uv[:, 0]) & (uv[:, 0] < (512 + pad))
        mask &= ((0 - pad) <= uv[:, 1]) & (uv[:, 1] < (512 + pad))

        np.savez_compressed(
            f"{self.save_folder}/point_clouds/{filename}.npz",
            coord=coord_sorted[mask],
            color=color_sorted[mask],
            normal=normal_sorted[mask]
        )

        u, v = uv[mask, 0], uv[mask, 1]
        img_recon = np.zeros((512, 512, 3), np.uint8)
        img_recon[v, u] = color_sorted[mask]
        Image.fromarray(img_recon).save(f"{self.save_folder}/images_proj/{filename}.jpg")
        img_recon[v, u] = normal_rgb_sorted[mask]
        Image.fromarray(img_recon).save(f"{self.save_folder}/images_proj/{filename}_normal.jpg")

        # with open(f"{self.save_folder}/{filename}.txt", "w") as f:
        #     for (x, y, z), (r, g, b) in zip(coord[mask], color[mask]):
        #         f.write("%.3lf %.3lf %.3lf %d %d %d\n" % (x, y, z, r, g, b))
        
        self.frames.append({
            "phase": phase,
            "file_path": f"images/{filename}.jpg",
            "dep_path": f"depths/{filename}.npz",
            "pc_path": f"point_clouds/{filename}.npz",
            "sharpness": 30.0,
            "transform_matrix": c2w.tolist(),
        })
    
    def save_single_data_json(self, idx):
        prefix = self.filelist[idx]
        name = self.take_name(prefix)
        phase = self.name_to_phase[name]

        # print(idx + 1, "/", self.size, prefix)
        filename = prefix.split("/")[-1]
        basename = os.path.basename(prefix)
        camera_prefix = prefix#.replace("image", "camr")
        depth_prefix = prefix#.replace("image", "depth")

        depth = np.load(f"{depth_prefix}_dpth.npz")["depth"][..., 0]
        valid_map = (self.z_near <= depth) & (depth <= self.z_far)
        if valid_map.mean() < 0.1:
            print(prefix, "invalid")
            # np.savez_compressed(f"{self.save_folder}/depths/{filename}.npz", depth=depth[..., np.newaxis] * 0 + 0.2)
            return

        # os.system(f"ln -s {prefix}_imag.jpg {self.save_folder}/images/{filename}.jpg")
        # os.system(f"ln -s {depth_prefix}_dpth.npz {self.save_folder}/depths/{filename}.npz")

        with np.load(f"{camera_prefix}_camr.npz") as N:
            w2c = N["R"]
            c2w = np.linalg.inv(N["R"])
            c2w[:3, 3] *= 0
        
        self.frames.append({
            "phase": phase,
            "file_path": f"images/{filename}.png",
            "dep_path": f"depths/{filename}.npz",
            "pc_path": f"point_clouds/{filename}.npz",
            "sharpness": 30.0,
            "transform_matrix": c2w.tolist(),
        })
    
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

        with open(f"{self.save_folder}/transforms.json", "w") as f:
            f.write(json.dumps(d, indent=2))


if __name__ == "__main__":

    dataset = HoliCityDataset(
        rootdir="/cluster/project/cvg/zuoyue/HoliCity",
        pointclouddir="/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_index_0_220",
    )
    dataset.create_save_folder("/cluster/project/cvg/zuoyue/torch-ngp/data/holicity_single_view_resampled400normal")
    for i in tqdm.tqdm(list(range(len(dataset)))):
        dataset.save_single_data(i)
        # dataset.save_single_data_json(i)
    dataset.save_json()
