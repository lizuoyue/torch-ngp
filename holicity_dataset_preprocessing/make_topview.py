import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import json
import torch
from z_buffer import RasterizePointsXYsBlending 

import math

import numpy as np
from scipy.optimize import least_squares

Offset = np.array(
    [
        [
            [51.48653658, -0.15787351],
            [51.49552138, -0.15750779],
            [51.50450284, -0.15714251],
            [51.5134942, -0.15678146],
            [51.522482, -0.15641665],
            [51.53147821, -0.15603593],
        ],
        [
            [51.48629708, -0.14348133],
            [51.4952862, -0.14311014],
            [51.50427476, -0.14273894],
            [51.51326159, -0.14237039],
            [51.52225695, -0.14200299],
            [51.53124625, -0.14162871],
        ],
        [
            [51.48606092, -0.12908871],
            [51.49505511, -0.12871558],
            [51.50405668, -0.12834281],
            [51.51303504, -0.1279733],
            [51.52202077, -0.12760482],
            [51.53100586, -0.12723739],
        ],
        [
            [51.485832, -0.11469519],
            [51.49482364, -0.11431929],
            [51.50381857, -0.11394519],
            [51.51280529, -0.11357625],
            [51.5217936, -0.11320131],
            [51.53077853, -0.11282795],
        ],
        [
            [51.48560681, -0.10029999],
            [51.49459821, -0.09992166],
            [51.50358628, -0.09954092],
            [51.51256733, -0.09916756],
            [51.52156189, -0.09879787],
            [51.53055204, -0.09841437],
        ],
        [
            [51.48537856, -0.08590049],
            [51.49436934, -0.08552391],
            [51.50335272, -0.08514402],
            [51.51233467, -0.08477007],
            [51.52132308, -0.08439054],
            [51.53031843, -0.08400393],
        ],
        [
            [51.48514379, -0.07149651],
            [51.49412855, -0.07111692],
            [51.50311466, -0.0707391],
            [51.51210374, -0.07036401],
            [51.52108874, -0.06998574],
            [51.53007984, -0.06960159],
        ],
        [
            [51.48490644, -0.05709119],
            [51.49389141, -0.05671215],
            [51.50287896, -0.05633446],
            [51.51186803, -0.05595725],
            [51.5208552, -0.05557937],
            [51.5298418, -0.05520037],
        ],
    ]
)


def model2gps(X):
    # [-40, -21] ~ [21, 20]
    x, y = X
    x0, y0 = math.floor(x / 1000), math.floor(y / 1000)
    xp, yp = x0 + 4, y0 + 3
    O = Offset[xp, yp]
    Ox = Offset[xp + 1, yp]
    Oy = Offset[xp, yp + 1]
    Oxy = Offset[xp + 1, yp + 1]
    dx, dy = x / 1000 - x0, y / 1000 - y0
    return (
        O * (1 - dx) * (1 - dy)
        + Ox * dx * (1 - dy)
        + Oy * (1 - dx) * dy
        + Oxy * dx * dy
    )


def gps2model(Y0):
    def risk(X):
        Y = model2gps(X)
        return Y - Y0

    result = least_squares(risk, np.r_[0, 0], gtol=1e-12, verbose=0)
    return result.x

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

        self.pts_coord = []
        self.pts_color = []
        self.pts_cam = None

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
            print(model2gps(N["loc"][:2]))
        Image.open(f'{prefix}_imag.jpg').save("vis/img.jpg")

        return {
            'image': image,
            'plane_mask': plane_mask,
            'plane_normal': plane_normal,
            'depth': depth,
            'normal': normal,
            'world2cam': w2c,
            'cam_coord': cam_coord,
        }

    def get_single_data(self, idx):
        prefix = self.filelist[idx]
        basename = os.path.basename(prefix)
        camera_prefix = prefix.replace('image', 'camr')
        depth_prefix = prefix.replace('image', 'depth')

        img_pil = Image.open(f'{prefix}_imag.jpg')

        with np.load(f'{camera_prefix}_camr.npz') as N:
            c2w = np.linalg.inv(N['R'])
            self.pts_cam = c2w[:3, 3]
        
        with np.load(f'{depth_prefix}_dpth.npz') as N:
            depth = N['depth'][..., 0]

        mask = (0.0625 < depth) & (depth < 64.0)
        
        intrinsic_matrix = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        u, v = np.meshgrid(np.arange(512)+0.5, np.arange(512)+0.5)
        pixel_exp = np.stack([depth] * 3) * np.stack([u, v, np.ones((512, 512))])
        pixel_exp = np.reshape(pixel_exp, (3, -1))
        coord_cam = np.linalg.inv(intrinsic_matrix).dot(pixel_exp)
        coord_cam[1] *= -1
        coord_cam[2] *= -1
        coord_wor = c2w[:3,:3].dot(coord_cam).T + self.pts_cam

        self.pts_coord.append(coord_wor[mask.flatten()]) # HW * 3
        self.pts_color.append(
            np.concatenate([np.array(img_pil)[mask], self.pts_coord[-1][:, -1:]], axis=1)
        )
    
    def get_data(self, idx):
        self.pts_coord = []
        self.pts_color = []
        self.pts_cam = None
        for i in range(idx * 8, idx * 8 + 8):
            self.get_single_data(i)
        return np.concatenate(self.pts_coord, axis=0), np.concatenate(self.pts_color, axis=0), self.pts_cam
        
    def save_pts(self, pts_coord, pts_color):
        with open('/home/lzq/lzy/HoliCity/vis/pts.txt', 'w') as f:
            for (x, y, z), (r, g, b) in zip(pts_coord, pts_color):
                f.write(f'{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n')
    
    def z_buffer(self, pts_coord, pts_color):
        pts_coord /= 64

        mask  = (-1 < pts_coord[:, 0]) & (pts_coord[:, 0] < 1)
        mask &= (-1 < pts_coord[:, 1]) & (pts_coord[:, 1] < 1)

        coord = torch.Tensor(pts_coord)[None, ...].cuda()
        color = torch.Tensor(pts_color)[None, ...].permute([0, 2, 1]).float().cuda()

        class Namespace(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        opts = Namespace(tau=1.0, rad_pow=2, accumulation="alphacomposite")
        layer = RasterizePointsXYsBlending(opts=opts).cuda()
        
        img = layer(coord, color)
        Image.fromarray(img[0, :3].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)[::-1]).save("vis/topview.png")
        dep = img[0, 3].cpu().numpy().astype(np.uint8)[::-1]
        mask = (dep == 0).astype(np.uint8) * 255
        dep = cv2.inpaint(dep, mask, 2, cv2.INPAINT_TELEA)
        Image.fromarray(dep).save("vis/topview_depth.png")

if __name__ == '__main__':


    dataset = HoliCityDataset('.', 'train') # valid test
    dataset[1520+65-1]
    quit()
    coord, color, cam = dataset.get_data(1519+10)
    # dataset.save_pts(coord, color)
    coord -= cam
    coord[:, 2] = (cam[2] + 256) - coord[:, 2]
    dataset.z_buffer(coord, color)
