import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm

def simple_downsample(pts, rgb, voxel_size):
    d = {}
    pts_int = np.round(pts / voxel_size).astype(np.int32)
    for (x,y,z), c in zip(pts_int, rgb):
        if (x,y,z) in d:
            d[(x,y,z)].append(c)
        else:
            d[(x,y,z)] = [c]
    vp, vc = [], []
    for (x,y,z), c_li in d.items():
        vp.append(np.array([x,y,z]))
        vc.append(np.round(np.stack(c_li).astype(np.float32).mean(axis=0)).astype(np.uint8))
    return np.stack(vp)*voxel_size, np.stack(vc)

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
    
    def create_dataset_folder(self, dataset_folder):
        self.dataset_folder = dataset_folder
        os.system(f'mkdir {dataset_folder}/{self.split}')
        os.system(f'mkdir {dataset_folder}/{self.split}/depth')
        os.system(f'mkdir {dataset_folder}/{self.split}/npz')
        os.system(f'mkdir {dataset_folder}/{self.split}/pose')
        os.system(f'mkdir {dataset_folder}/{self.split}/rgb')
        os.system(f'mkdir {dataset_folder}/{self.split}/mask')
        with open(f'{dataset_folder}/{self.split}/intrinsics.txt', 'w') as f:
            f.write('128.0 0.0 128.0 0.0\n')
            f.write('0.0 128.0 128.0 0.0\n')
            f.write('0.0 0.0 1.0 0.0\n')
            f.write('0.0 0.0 0.0 1.0\n')
        return
    

    def save_single_data(self, idx):
        prefix = self.filelist[idx]
        print(prefix)
        basename = os.path.basename(prefix)
        plane_prefix = prefix.replace('image', 'plane')
        depth_prefix = prefix.replace('image', 'depth')
        normal_prefix = prefix.replace('image', 'normal')
        camera_prefix = prefix.replace('image', 'camr')

        img_pil = Image.open(f'{prefix}_imag.jpg')
        img_pil = img_pil.resize((256, 256), Image.LANCZOS)
        img_pil.save(f'{self.dataset_folder}/{self.split}/rgb/{basename}.jpg')
        img = np.array(img_pil)

        with np.load(f'{depth_prefix}_dpth.npz') as N:
            depth = N['depth']
            cam_coord = dep_to_cam_coord(depth)
            depth = depth[::2,::2, 0]
            cam_coord = cam_coord[::2, ::2]
            np.savez_compressed(f'{self.dataset_folder}/{self.split}/depth/{basename}.npz', depth=depth)
        
        valid_mask = (0.5 < depth) & (depth < 64) # half resolution
        pts = cam_coord[valid_mask]
        rgb = img[valid_mask]
        Image.fromarray(valid_mask.astype(np.uint8) * 255).save(f'{self.dataset_folder}/{self.split}/mask/{basename}.png')

        with np.load(f'{camera_prefix}_camr.npz') as N:
            c2w = np.linalg.inv(np.diag([1,-1,-1,1.0]).dot(N['R']))
            pts = pts.dot(c2w[:3,:3].T) + c2w[:3,3]

        # with open('pts.txt', 'a') as f:
        #     for (x,y,z), (r,g,b) in zip(*simple_downsample(pts, rgb, 0.4)):
        #         f.write('%.1lf %.1lf %.1lf %d %d %d\n' % (x,y,z,r,g,b))
        
        np.savez_compressed(f'{self.dataset_folder}/{self.split}/npz/{basename}.npz', rgb=rgb, pts=pts)
        np.savetxt(f'{self.dataset_folder}/{self.split}/pose/{basename}.txt', c2w, fmt='%.9lf')
        return

if __name__ == '__main__':

    dataset = HoliCityDataset('.', 'train') # valid test
    dataset.create_dataset_folder('HoliCityFwd')
    for i in tqdm.tqdm(list(range(len(dataset)-56, len(dataset)-48))):
        dataset.save_single_data(i)
