import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

def dep_to_cam_coord(dep):
    y = np.arange(dep.shape[0])
    x = np.arange(dep.shape[1])
    x, y = np.meshgrid(x, y)
    px = (x - 255.5) / 256
    py = (-y + 255.5) / 256
    pz = dep[..., 0]
    return np.dstack([px * pz, py * pz, -pz])

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
        
        # with open(f'{idx}.txt', 'w') as f:
        #     for xyz, rgb in zip(cam_coord.reshape((-1, 3)), np.round(image*255.0).astype(np.uint8).reshape((-1, 3))):
        #         f.write('%.3lf %.3lf %.3lf ' % tuple(xyz.tolist()))
        #         f.write('%d %d %d\n' % tuple(rgb.tolist()))
        
        if True:
            fig = plt.figure()
            axes = []
            rows, cols = 2, 2

            axes.append( fig.add_subplot(rows, cols, 1) )
            axes[-1].set_title('image')  
            plt.imshow(image)

            axes.append( fig.add_subplot(rows, cols, 2) )
            axes[-1].set_title('depth')  
            plt.imshow(depth / 100.0)

            axes.append( fig.add_subplot(rows, cols, 3) )
            axes[-1].set_title('normal')  
            plt.imshow((normal + 1.0) / 2.0)

            axes.append( fig.add_subplot(rows, cols, 4) )
            axes[-1].set_title('plane mask')  
            plt.imshow(plane_mask)

            fig.tight_layout()    
            plt.savefig(f'{idx}.png')
            plt.clf()
            plt.close()

        return {
            'image': image,
            'plane_mask': plane_mask,
            'plane_normal': plane_normal,
            'depth': depth,
            'normal': normal,
            'world2cam': w2c,
            'cam_coord': cam_coord,
        }




if __name__ == '__main__':

    train_dataset = HoliCityDataset('.', 'train')
    valid_dataset = HoliCityDataset('.', 'valid')
    test_dataset = HoliCityDataset('.', 'test')

    xyzs, rgbs = [], []
    start = 45032 - 8 * 5
    for idx in range(start, 45032):
        data = train_dataset.__getitem__(idx)
        coord = np.concatenate([data['cam_coord'], data['cam_coord'][..., :1] * 0 + 1], axis=-1)
        w_coord = coord.dot(np.linalg.inv(data['world2cam']).T)[..., :3]
        valid  = data['depth'][..., 0] > 0.125 # (2^-3)
        valid &= data['depth'][..., 0] < 32.0 # (2^5)
        valid[::2,::2] = False
        xyzs.append(w_coord[valid])
        rgbs.append(data['image'][valid])
    xyzs = np.concatenate(xyzs, axis=0)
    rgbs = np.concatenate(rgbs, axis=0)

    with open(f'{start}-45032.txt', 'w') as f:
        for xyz, rgb in zip(xyzs, np.round(rgbs*255.0).astype(np.uint8)):
            f.write('%.3lf %.3lf %.3lf ' % tuple(xyz.tolist()))
            f.write('%d %d %d\n' % tuple(rgb.tolist()))



