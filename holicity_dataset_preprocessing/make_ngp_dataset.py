import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import json

def dep_to_cam_coord(dep):
    y = np.arange(dep.shape[0])
    x = np.arange(dep.shape[1])
    x, y = np.meshgrid(x, y)
    px = (x - 255.5) / 256
    py = (-y + 255.5) / 256
    pz = dep[..., 0]
    return np.dstack([px * pz, -py * pz, pz])

class HoliCityDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/filelist.txt', dtype=str)
        filter_ = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/{split}-middlesplit.txt', dtype=str)
        length = len(filter_[0])
        filter_ = set(filter_)

        self.filelist = [f'{rootdir}/image-v1/{f}' for f in filelist if f[:length] in filter_]
        self.size = len(self.filelist)
        print(f'num {split}:', self.size)
        self.count = 0
        self.frames = []
        self.all_pts = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        plane_prefix = prefix.replace('image', 'plane')
        depth_prefix = prefix.replace('image', 'depth')
        normal_prefix = prefix.replace('image', 'normal')
        camera_prefix = prefix.replace('image', 'camr')

        image = cv2.imread(f'{prefix}_imag.jpg', -1).astype(np.float32) / 255.0
        image = image[:,:,::-1] # (512, 512, 3) float32 0.0 1.0
        plane_mask = cv2.imread(f'{plane_prefix}_plan.png', -1) # uint16 (512, 512)
        with np.load(f'{plane_prefix}_plan.npz') as N:
            plane_normal = N['ws'] # list of normals, shape: (n_plane, 3), float64
        with np.load(f'{depth_prefix}_dpth.npz') as N:
            depth = N['depth'] # (512, 512, 1) float32
            cam_coord = dep_to_cam_coord(depth)
        with np.load(f'{normal_prefix}_nrml.npz') as N:
            normal = N['normal'] # (512, 512, 3) float32 -1 to 1
        with np.load(f'{camera_prefix}_camr.npz') as N:
            # ['R', 'q', 'loc', 'yaw', 'fov', 'pitch', 'pano_yaw', 'tilt_yaw', 'tilt_pitch']
            w2c = N['R']

        return {
            'image': image,
            'plane_mask': plane_mask,
            'plane_normal': plane_normal,
            'depth': depth,
            'normal': normal,
            'world2cam': w2c,
            'cam_coord': cam_coord,
        }

    def save_single_data(self, idx):
        prefix = self.filelist[idx]
        basename = os.path.basename(prefix)
        camera_prefix = prefix.replace('image', 'camr')
        depth_prefix = prefix.replace('image', 'depth')

        img_pil = Image.open(f'{prefix}_imag.jpg')
        img_pil.save(f'/home/lzq/lzy/torch-ngp/data/holicity0001/images/{self.count:04d}.jpg')

        with np.load(f'{camera_prefix}_camr.npz') as N:
            # c2w = np.linalg.inv(np.diag([1,-1,-1,1.0]).dot(N['R']))
            c2w = np.linalg.inv(N['R']) 
            c2w[:3, 3] *= 0
        
        with np.load(f'{depth_prefix}_dpth.npz') as N:
            depth = N['depth'][..., 0]
        
        # mask = depth < -100
        # mask[::2, ::2] = True
        # mask &= (0.25 < depth) & (depth < 64.0)
        mask = (0.0625 < depth) & (depth < 64.0)
        
        intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        u, v = np.meshgrid(np.arange(512)+0.5, np.arange(512)+0.5)
        pixel_exp = np.stack([depth] * 3) * np.stack([u, v, np.ones((512, 512))])
        pixel_exp = np.reshape(pixel_exp, (3, -1))
        coord_cam = np.linalg.inv(intrinsic_matrix).dot(pixel_exp)
        coord_cam[1] *= -1
        coord_cam[2] *= -1
        coord_wor = c2w[:3,:3].dot(coord_cam)

        self.all_pts.append(coord_wor.T[mask.flatten()]) # HW * 3
        
        self.frames.append({
            "file_path": f"images/{self.count:04d}.jpg",
            "sharpness": 30.0,
            "transform_matrix": c2w.tolist(),
        })
        self.count += 1
    
    def save_json(self):
        d = {
            "fl_x": 256.0,
            "fl_y": 256.0,
            "cx": 256.0,
            "cy": 256.0,
            "w": 512.0,
            "h": 512.0,
            "aabb_scale": 4,
            "frames": self.frames,
        }

        with open('/home/lzq/lzy/torch-ngp/data/holicity0001/transforms.json', 'w') as f:
            f.write(json.dumps(d, indent=2))
        
    def save_all_pts(self):
        with open('/home/lzq/lzy/torch-ngp/data/holicity0001/pts.txt', 'w') as f:
            for pts in self.all_pts:
                for pt in pts:
                    f.write(f'{pt[0]:.9f} {pt[1]:.9f} {pt[2]:.9f}\n')
        with open('/home/lzq/lzy/torch-ngp/data/holicity0001/pts_bound2.txt', 'w') as f:
            for pts in self.all_pts:
                pts /= (128 * 1.6) # -0.5~0.5
                pts *= 4
                for pt in pts:
                    f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n')


if __name__ == '__main__':

    dataset = HoliCityDataset('.', 'train') # valid test
    # for i in tqdm.tqdm(list(range(len(dataset)-56, len(dataset)-48))):
    for i in tqdm.tqdm(list(range(len(dataset)-128, len(dataset)-120))):
        dataset.save_single_data(i)
    dataset.save_json()
    dataset.save_all_pts()
