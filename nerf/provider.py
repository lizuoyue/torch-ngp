import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def get_circle_poses(radius, focus_distance, n_poses=30):
    r, l = radius, focus_distance
    poses = []
    for i in range(n_poses):
        theta = 2 * np.pi / n_poses * i
        cos = np.cos(theta)
        sin = np.sin(theta)
        mat = np.array([
            [l, 0, -r * cos, r * cos],
            [0, l, -r * sin, r * sin],
            [r * cos, r * sin, l, 0],
            [0, 0, 0, 1],
        ])
        mat[:3, :3] /= (np.linalg.det(mat[:3, :3]) ** (1/3))
        poses.append(np.linalg.inv(mat).astype(np.float32))
    return poses


def deps_to_camera_coord(deps, intrinsics):
    # deps: [B, H, W, 1]
    fx, fy, cx, cy = intrinsics.tolist()
    B, H, W = deps.shape[:3]
    y = torch.arange(H, device=deps.device)
    x = torch.arange(W, device=deps.device)
    y, x = torch.meshgrid(y, x) # AAA, BBB = torch.meshgrid(A, B)
    px = (x + 0.5 - cx) / fx
    px = torch.stack([px] * B, dim=0)
    py = (y + 0.5 - cy) / fy
    py = torch.stack([py] * B, dim=0)
    pz = deps[..., 0]
    return torch.stack([px * pz, py * pz, pz], dim=-1)


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = False#opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        self.depths = None
        if "z_near" in transform:
            assert("z_far" in transform)
            self.z_near = transform["z_near"]
            self.z_far = transform["z_far"]
            self.depths = []
            self.coords = []
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and False:#type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
            
            circle_poses = get_circle_poses(0.03, 1.0) # local_circle to camera
            self.poses = []
            for frame in frames:
                pose = nerf_matrix_to_ngp(np.array(frame['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                self.poses += [pose @ cp for cp in circle_poses]

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    if opt.dataset_scale == 'tiny':
                        frames = frames[30:31]
                    elif opt.dataset_scale == 'resampletiny':
                        frames = frames[0:8]
                    elif opt.dataset_scale == 'small':
                        frames = frames[:80]
                    elif opt.dataset_scale == 'full':
                        frames = frames[:1860]
                    elif opt.dataset_scale.startswith('resample'):
                        frames = frames[:1550]
                    else:
                        # raise Exception(f'Do not support given dataset_scale arg: {opt.dataset_scale}')
                        a, b = opt.dataset_scale.split(",")
                        frames = frames[int(a):int(b)]
                elif type == 'val':
                    if opt.dataset_scale == 'tiny':
                        frames = frames[30:31]
                    elif opt.dataset_scale == 'resampletiny':
                        frames = frames[0:16]
                    elif opt.dataset_scale == 'small':
                        frames = frames[0:60:10] + frames[1525:1566:10]
                    elif opt.dataset_scale == 'full':
                        frames = frames[1860:] + frames[60:1860:100]
                    elif opt.dataset_scale.startswith('resample'):
                        frames = frames[0:80:8] + frames[1550:]
                    else:
                        # raise Exception(f'Do not support given dataset_scale arg: {opt.dataset_scale}')
                        a, b = opt.dataset_scale.split(",")
                        frames = frames[int(a):int(b)]
                # else 'all' or 'trainval' : use all frames
                else:
                    assert(type == 'test')
                    if opt.dataset_scale == 'tiny':
                        raise NotImplementedError
                    elif opt.dataset_scale == 'small':
                        raise NotImplementedError
                        frames = frames[:1]
                        n_poses = 1
                        circle_poses = np.eye(4)[np.newaxis, ...].astype(np.float32)
                    elif opt.dataset_scale == 'full':
                        n_poses = 24
                        circle_poses = get_circle_poses(0.04, 8.0, n_poses) # local_circle to camera
                        circle_poses.append(np.eye(4).astype(np.float32))
                        n_poses += 1
                        frames = [frames[1520-1+i] for i in [10,20,34,50,65]]#[10]*n_poses+[20]*n_poses+[34]*n_poses+[50]*n_poses+[65]*n_poses]
                        # frames = [frames[i] for i in [0, 78, 79, 80, 81, 82]]#+[79]*n_poses+[80]*n_poses+[81]*n_poses+[82]*n_poses]
                    elif opt.dataset_scale.startswith("resample"):
                        targets = [opt.dataset_scale.replace("resample_", "")]
                        temp_frames = []
                        # "0cm3FmEopleXtLauJaJ8ng"
                        # "1AF3wXsMQbL1Nb-ZxHrSDQ"
                        # "1RABiy3kVucrVKftff4oEg"
                        # "1mKXDGKTuJmnsWU_AQiv4Q"
                        for frame in frames:
                            for target in targets:
                                if target in frame["file_path"]:
                                    temp_frames.append(frame)
                        frames = temp_frames
                        n_poses = 24
                        circle_poses = [np.eye(4).astype(np.float32)]
                        circle_poses.extend(get_circle_poses(0.04, 8.0, n_poses)) # local_circle to camera
                        n_poses += 1
                    else:
                        n_poses = 12
                        circle_poses = [np.eye(4).astype(np.float32)]
                        circle_poses.extend(get_circle_poses(0.15, 10.0, n_poses)) # local_circle to camera
                        n_poses += 1
                        a, b = opt.dataset_scale.split(",")
                        frames = frames[int(a):int(b)]
                        # frames = frames[1520:1600]
                        # frames = frames[:80]
                        # frames = [frames[i] for i in [1]*11+[9]*11+[17]*11+[25]*11+[33]*11]
                        # frames = [frames[1520-1+i] for i in [10,20,34,50,65]]  

            self.org_poses = []
            self.poses = []
            self.images = []
            self.frames = frames
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                self.org_poses.append(pose)
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if opt.zero_input:
                    image *= 0
                    # image = np.random.randn(*image.shape)
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                resize = False
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    resize = True
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                if 'dep_path' in f:
                    d_path = os.path.join(self.root_path, f['dep_path'])
                    with np.load(d_path) as d_file:
                        depth = d_file['depth']
                        if opt.constant_depth:
                            depth *= 0
                            depth += 3.0 # meter
                        if resize:
                            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                            if len(depth.shape) == 2:
                                depth = depth[..., np.newaxis]
                    self.depths.append(depth)
                    if type == 'test':
                        self.depths += [depth] * (n_poses - 1)

                self.poses.append(pose)
                if type == 'test':
                    self.poses.pop()
                    self.poses += [pose @ cp for cp in circle_poses]
                self.images.append(image)
                if type == 'test':
                    self.images += [image] * (n_poses - 1)
        
        if type == 'test' and opt.dataset_scale.startswith('resample'):
            for i in range(0, len(self.poses), n_poses):
                # interp between self.poses[i], self.poses[i+n_poses]
                j = (i + n_poses) % len(self.poses)
                rots = Rotation.from_matrix(np.stack([self.poses[i][:3, :3], self.poses[j][:3, :3]]))
                slerp = Slerp([0, 1], rots)
                for k in range(1, n_poses):
                    ratio = np.sin((k / n_poses - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * self.poses[i][:3, 3] + ratio * self.poses[j][:3, 3]
                    self.poses[i + k] = pose
            self.frames *= 200
            # self.poses = self.poses[::n_poses] + self.poses
            # self.depths = self.depths[::n_poses] + self.depths
            # self.images = self.images[::n_poses] + self.images
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        if self.depths is not None:
            self.depths = torch.from_numpy(np.stack(self.depths, axis=0)) # [N, H, W, 1]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.depths is not None:
                self.depths = self.depths.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        if self.opt.random_z_rotate:

            left_mat_nerf2ngp = torch.eye(4)[[1, 2, 0, 3]].to(poses.device)
            right_mat_nerf2ngp = torch.diag(torch.Tensor([1, -1, -1, 1])).to(poses.device)
            inv_left = torch.linalg.inv(left_mat_nerf2ngp)
            inv_right = torch.linalg.inv(right_mat_nerf2ngp)

            org_poses = torch.matmul(inv_left, torch.matmul(poses, inv_right))

            # print(org_poses[0])
            # print(self.org_poses[index[0]])
            # input()
            # if self.opt.random_z_rotate:
            #     pose = self.rand_z_pose.dot(pose)

            rand_theta = torch.rand([]) * 2 * np.pi
            rand_z_pose = torch.Tensor([
                [torch.cos(rand_theta), -torch.sin(rand_theta), 0, 0],
                [torch.sin(rand_theta), torch.cos(rand_theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).to(poses.device)

            org_poses = torch.matmul(rand_z_pose, org_poses)
            poses = torch.matmul(left_mat_nerf2ngp, torch.matmul(org_poses, right_mat_nerf2ngp))


        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        if self.depths is not None:
            
            if "pc_path" not in self.frames[index[0]]:
                images = self.images[index].to(self.device) # [B, H, W, 3/4]
                depths = self.depths[index].to(self.device) # [B, H, W, 1]
                masks = (self.z_near < depths[..., 0]) & (depths[..., 0] < self.z_far) # [B, H, W]
                coords = deps_to_camera_coord(depths, self.intrinsics) # [B, H, W, 3]

                results['pts_batch'] = torch.arange(B).to(self.device).repeat(self.H, self.W, 1).permute([2, 0, 1])
                results['pts_coords'] = torch.matmul(
                    coords.view(B, self.H * self.W, 3),
                    poses[:, :3, :3].transpose(1, 2),
                ).view(B, self.H, self.W, 3)
                results['pts_coords'] = results['pts_coords'][..., [2, 0, 1]]
                results['pts_masks'] = masks
                results['pts_rgbs'] = images
            
            else:
                depths = self.depths[index].to(self.device) # [B, H, W, 1]

                pc_path = os.path.join(self.root_path, self.frames[index[0]]["pc_path"])
                pc_dict = np.load(pc_path)
                num_points = pc_dict["coord"].shape[0]
                results['pts_batch'] = torch.zeros(num_points).to(self.device).float()
                results['pts_coords'] = torch.from_numpy(
                    pc_dict["coord"]#.dot(self.rand_z_pose[:3, :3].T)
                ).to(self.device).float()
                results['pts_masks'] = torch.ones(num_points).to(self.device).bool()
                results['pts_rgbs'] = torch.from_numpy(pc_dict["color"]).to(self.device).float() / 255.0
                if "normal" in pc_dict:
                    results['pts_normal'] = torch.from_numpy(pc_dict["normal"]).to(self.device).float()

                if self.opt.random_z_rotate:
                    results['pts_coords'] = torch.matmul(results['pts_coords'], rand_z_pose[:3, :3].transpose(0, 1))

            # with open(f'vis/{index[0]:04d}.txt', 'w') as f:
            #     for (x, y, z), (r, g, b) in zip(
            #         results['pts_coords'][results['pts_masks']].cpu().numpy(),
            #         (255 * results['pts_rgbs'][results['pts_masks']].cpu().numpy()).astype(np.uint8),
            #     ):
            #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

            # input('check')

            if self.training:
                C = depths.shape[-1]
                depths = torch.gather(depths.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 1]
            results['depths'] = depths
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None#self.training
        return loader
    
    def dataloader_my(self, batch_size):
        loader = DataLoader(list(range(len(self.poses))), batch_size=batch_size, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader