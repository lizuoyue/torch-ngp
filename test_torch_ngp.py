import torch
import torch.nn as nn
import numpy as np
from gridencoder import GridEncoder
import collections
import MinkowskiEngine as ME
from gridencoder.resfieldnet import ResFieldNet50Hierarchical

def display_dict(d, prefix):
    if type(d) is dict or type(d) is collections.OrderedDict:
        for key in d:
            print(prefix, key)
            display_dict(d[key], prefix + '    ')
    else:
        print(prefix, type(d), end=' ')
        if type(d) is torch.Tensor:
            print(d.shape)
        elif type(d) is list:
            print(len(d))
        elif type(d) is bool or type(d) is int or type(d) is float:
            print(d)
        else:
            print()
    return


in_channels = 3
field_ch = 32
field_ch2 = 64
field_network = nn.Sequential(
    ME.MinkowskiSinusoidal(in_channels, field_ch),
    ME.MinkowskiBatchNorm(field_ch),
    ME.MinkowskiReLU(inplace=True),
    ME.MinkowskiLinear(field_ch, field_ch),
    ME.MinkowskiBatchNorm(field_ch),
    ME.MinkowskiReLU(inplace=True),
    ME.MinkowskiToSparseTensor(),
).cuda()
field_network2 = nn.Sequential(
    ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
    ME.MinkowskiBatchNorm(field_ch2),
    ME.MinkowskiReLU(inplace=True),
    ME.MinkowskiLinear(field_ch2, field_ch2),
    ME.MinkowskiBatchNorm(field_ch2),
    ME.MinkowskiReLU(inplace=True),
    ME.MinkowskiToSparseTensor(),
).cuda()


if __name__ == '__main__':

    ckpt = torch.load('trial_holicity0001_mink_4layers/checkpoints/ngp_ep0913.pth')
    display_dict(ckpt, '')
    df = ckpt["model"]["density_grid"].cpu().numpy()
    func = lambda x: x[(-1 < x) & (x < np.inf)]
    # df0 = func(df[0])
    # df1 = func(df[1])
    # print(df0.shape, df1.shape)
    # print(df0.min(), df0.max())
    # print(df1.min(), df1.max())
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt

    col_for_showing = ['cas0', 'cas1']
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))
    for i in range(2):
        counts, bins = np.histogram(func(df[i]), bins=100, range=(0, 1000))
        print(counts)
        print(bins)
        print()
        axes[i].hist(bins[:-1], bins, weights=counts)
    fig.savefig('aaaa.png')
    quit()

    # fts = torch.FloatTensor([
    #     [3, 4, 5, 6],
    #     [6, 7, 8, 9],
    #     [0, 1, 2, 3],
    # ]).cuda()
    # coords = torch.FloatTensor([
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 2],
    #     [0, 1, 2, 1],
    # ]).cuda()
    # query_coords = torch.FloatTensor([
    #     [0, 1, 1.5, 1.5],
    # ]).cuda()
    # nsvf = ME.SparseTensor(features=fts, coordinates=coords+0.7)
    # print(nsvf.C)

    # print(nsvf.features_at_coordinates(query_coords))
    # quit()

    pts = np.loadtxt('/home/lzq/lzy/torch-ngp/data/holicity0001/pts.txt')
    x_field = ME.TensorField(
        features=torch.from_numpy(pts).float().cuda(),
        coordinates=ME.utils.batched_coordinates([pts * 80], dtype=torch.float32),  # 0.0125m
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device=torch.device('cuda:0'),
    )

    # print(pts.shape)
    # otensor = field_network(x_field)
    # print('otensor', otensor.C.shape, otensor.F.shape)
    # ccc = otensor.cat_slice(x_field)
    # print('ccc', ccc.C.shape, ccc.F.shape)
    # otensor2 = field_network2(ccc)
    # print('otensor2', otensor2.C.shape, otensor2.F.shape)
    net = ResFieldNet50Hierarchical(3, 8).cuda()

    for item in net(x_field):
        print(item.F.shape)

    quit()

    n_voxel = 20000
    n_query = 953728

    fts = torch.rand(n_voxel, 2).cuda()
    coords = torch.cat([torch.ones(n_voxel, 1), torch.randint(128, (n_voxel, 3))], dim=1).cuda()
    # print(coords.dtype, coords.shape)
    nsvf = ME.SparseTensor(features=fts, coordinates=coords+0.5)
    # query_fts = torch.rand(n_query, 9).cuda()
    # tfield = ME.TensorField(coordinates=query_coords, features=query_fts, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE, coordinate_manager=nsvf.coordinate_manager)

    # print(nsvf.slice(tfield))
    import time
    for _ in range(10):
        query_coords = torch.rand(n_query, 4).cuda()
        query_coords[:, 0] = 1.0
        query_coords[:, 1:] *= 128.0

        tic = time.time()
        a = nsvf.features_at_coordinates(query_coords)
        toc = time.time()
        print(toc - tic)

    print()

    quit()

    pts = np.loadtxt('data/holicity0000/pts.txt')
    print(pts.shape)
    print(pts.min(axis=0), pts.max(axis=0))

    sizes = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
    resolutions = [128, 256, 512, 1024, 2048, 4096]
    index_encoders, voxelized_pts_list = [], []
    to_save_arr = []
    d = {}
    offsets = [0]
    for size, resolution in zip(sizes, resolutions):
        voxelized_pts = ME.utils.sparse_quantize(torch.from_numpy(pts), quantization_size=size).numpy()
        diff = voxelized_pts.max(axis=0) - voxelized_pts.min(axis=0)
        voxelized_pts_list.append(voxelized_pts + resolution // 2)
        print(size, voxelized_pts.shape, diff)

        encoder = GridEncoder(
            input_dim=3,
            num_levels=1,
            level_dim=1,
            base_resolution=resolution,
            log2_hashmap_size=19,
            # desired_resolution=128,
            per_level_scale=1,
            align_corners=True,
        )
        encoder.offsets = encoder.offsets.cuda()
        encoder = encoder.cuda()
        encoder.embeddings.data *= 0
        encoder.embeddings.data[:, 0] += torch.arange(524288).cuda()

        index_encoders.append(encoder)

        res_1 = resolution - 1
        coords = (voxelized_pts_list[-1] * 2 - res_1) / res_1

        print(voxelized_pts.shape, voxelized_pts.dtype, voxelized_pts)

        fts = encoder(torch.Tensor(coords).cuda())
        index = torch.round(fts).int()
        diff = index.float() - fts.float()
        print("diff", diff.min(), diff.max())

        to_save = np.hstack([voxelized_pts, index.cpu().numpy() + offsets[-1]])
        print("to_save", to_save.shape)
        to_save_arr.append(to_save)

        idx, cnt = torch.unique(index, return_counts=True)
        to_see = torch.argsort(cnt, descending=True)[:5]
        print(idx.shape, idx[to_see], cnt[to_see])
        print()

        offsets.append(offsets[-1] + 524288)

    # np.savez_compressed('data/holicity0000/pts_idx.npz', sizes=np.array(sizes), resolutions=np.array(resolutions), points=np.vstack(to_save_arr), offsets=np.array(offsets))
    
    quit()



    # coords, li = [], [-126/127, 0/127, 126/127] # 1 64 127
    coords, li = [], [-127/127, 0/127, 127/127] # 1 64 127
    for i in li:
        for j in li:
            for k in range(-127, 128, 2):
                # coords.append([i, j, k/127])
                coords = [[i, j, k/127]]
                fts = encoder(torch.Tensor(coords).cuda())
                for coord, ft in zip(coords, fts):
                    print(np.array(coord)*127, ft)
                    input()

    fts = encoder(torch.Tensor(coords).cuda())
    for coord, ft in zip(coords, fts):
        print(np.array(coord)*127, ft)
        input()
    quit()

    # n = 1000000
    # np.random.seed(1993)
    # coords = np.random.randint(1, 128, size=(n, 3))
    x, y, z = np.meshgrid(np.arange(1, 128), np.arange(1, 128), np.arange(1, 128))
    coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    res = encoder(torch.Tensor((coords - 0.5) / 127).cuda()).detach().cpu().numpy()
    d = {}
    for i in range(coords.shape[0]):
        key = int(np.round(res[i, 0]))
        if key in d:
            d[key].append(coords[i].tolist())
        else:
            d[key] = [coords[i].tolist()]
    for key in sorted(d.keys())[::-1]:
        print(d[key])
        input()
    f = {}  # len to num
    for key in d:
        if len(d[key]) in f:
            f[len(d[key])] += 1
        else:
            f[len(d[key])] = 1
    s = 0
    for l in sorted(list(f.keys())):
        print(l, f[l])
        s += f[l]
    print(s)
