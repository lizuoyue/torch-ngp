import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

import MinkowskiEngine as ME
from .resfieldnet import ResFieldNet50Hierarchical
from .resfieldnet import ResNet50
from .minkvae import MinkVAE, MinkVQVAE
from .minkfieldunet import MinkFieldUNet50Small, MinkFieldUNet14A, MinkFieldUNet34A
from .style_unet import StyleFieldUNet50Small, StyleFieldUNet14A
# from .stylegan2_3d import StyleGAN2_3D_Generator
from .stylegan1_3d import StyleGAN2_3D_Generator
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None
        


grid_encode = _grid_encode.apply


class GridEncoderGeometry(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            print("resolution", resolution)
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        d = np.load('/home/lzq/lzy/torch-ngp/data/holicity0000/pts_idx.npz')
        n_points = d["points"].shape[0]
        self.embeddings = nn.Parameter(torch.empty(n_points, level_dim))  # for occupied points
        # self.fs_embeddings = nn.Parameter(torch.empty(num_levels, level_dim))  # for free space
        self.fs_embeddings = nn.Parameter(torch.empty(offset, level_dim))  # for free space

        scatter_index = torch.stack(
            [torch.from_numpy(d["points"][:, -1]).long()] * (level_dim + 1),
        dim=-1)
        embedding_ones = torch.ones(n_points, 1).float()

        self.register_buffer('scatter_index', scatter_index)
        self.register_buffer('embedding_ones', embedding_ones)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)
        self.fs_embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        level_nums = self.offsets[1:] - self.offsets[:-1]

        # fs_embeddings = torch.repeat_interleave(self.fs_embeddings, level_nums, dim=0)

        src = torch.cat([self.embeddings, self.embedding_ones], dim=-1)

        embeddings_with_num = torch.scatter_add(
            torch.zeros(
                self.offsets[-1], self.level_dim + 1, dtype=self.embeddings.dtype, device=self.embeddings.device
            ),
            dim=0, index=self.scatter_index, src=src
        )

        sel = torch.clip(embeddings_with_num[:, -1:], max=1.0)

        embedding_num = torch.clip(embeddings_with_num[:, -1:], min=1.0)
        embeddings = embeddings_with_num[:, :-1] / embedding_num
        # embeddings = embeddings * sel + fs_embeddings * (1 - sel)
        embeddings = embeddings * sel + self.fs_embeddings * (1 - sel)

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs


def replace_features(x, features):
    assert(x.F.shape[0] == features.shape[0])
    if isinstance(x, ME.TensorField):
        return ME.TensorField(
            features=features.to(x.device),
            coordinate_field_map_key=x.coordinate_field_map_key,
            coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode,
            device=x.device,
        )
    elif isinstance(x, ME.SparseTensor):
        return ME.SparseTensor(
            features=features.to(x.device),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device,
        )
    else:
        pass
    raise ValueError("Input tensor is not ME.TensorField nor ME.SparseTensor.")
    return None


class GridEncoderMinkowski(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners

        # allocate parameters
        pts = np.loadtxt('/home/lzq/lzy/torch-ngp/data/holicity0001/pts.txt')
        pts = torch.from_numpy(pts).cuda()

        std = 1e-4
        voxel_size = 0.8
        voxel_sizes, resolutions = [], []
        self.embeddings = []
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            coords = ME.utils.sparse_quantize(pts / voxel_size)
            print("minkowski resolution & voxel size & num:", resolution, voxel_size, coords.shape[0])
            coords = torch.cat([torch.zeros(coords.shape[0], 1), coords], dim=-1)
            emb = nn.Parameter(torch.empty(coords.shape[0], level_dim))
            emb.data.uniform_(-std, std)
            self.register_parameter(f'emb_{i}', emb)
            self.embeddings.append(ME.SparseTensor(
                coordinates=coords.cuda(),
                features=torch.zeros(coords.shape[0], level_dim).cuda(),
            ))
            voxel_sizes.append(voxel_size)
            resolutions.append(resolution)
            voxel_size /= 2.0

        self.register_buffer('voxel_sizes', torch.IntTensor(voxel_sizes))
        self.register_buffer('resolutions', torch.IntTensor(resolutions))

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={[(name, tuple(param.shape)) for name, param in self.named_parameters()]} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        outputs = []
        prefix_shape = list(inputs.shape[:-1])
        level = 0
        inputs = inputs[..., [2, 0, 1]]

        # with open(f'vis/query.txt', 'w') as f:
        #     for (x, y, z) in inputs.cpu().numpy():
        #         f.write('%.3lf %.3lf %.3lf\n' % (x, y, z))
        # input("check")

        for resolution, embedding in zip(self.resolutions, self.embeddings):
            emb = getattr(self, f'emb_{level}')
            emb_with_ft = replace_features(embedding, emb)
            query = inputs / bound / 2.0 * resolution
            query = torch.cat([torch.zeros(*(prefix_shape + [1]), device=inputs.device), query], dim=-1)
            query = query.view(-1, 4)
            outputs.append(emb_with_ft.features_at_coordinates(query).view(*prefix_shape, -1))
            level += 1
        outputs = torch.cat(outputs, dim=-1)
        return outputs




class GridEncoderMinkowskiHierarchical(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, z_dim=128, 
                initseed_dim=8, num_gan_blocks=5, per_level_scale=2, base_resolution=16,
                log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, 
                embedding_net='unet', embedding_regu='none', scale=20, opt=None):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.align_corners = align_corners
        self.dual = False
        self.embedding_net = embedding_net
        self.embedding_regu = embedding_regu

        self.aabb_size = opt.aabb_size # meter, front 32m back 32m left 32m right 32m
        # self.scale = 80
        # for voxel sizes 0.1, 0.2, 0.4, and 0.8
        # originally 1/80, become 1/10 after 3 downsamples
        self.scale = scale
        # for voxel sizes 0.05, 0.1, 0.2, and 0.4
        # originally 1/160, become 1/20 after 3 downsamples
        # self.scale = 240
        # for voxel sizes
        # originally 1/240, become 1/30 after 3 downsamples

        
        
        if self.dual:
            self.net = ResFieldNet50Hierarchical(3, 4)
            self.net_dual = ResFieldNet50Hierarchical(3, 4)
            self.scale_dual = self.scale * np.sqrt(2)
            # self.output_dim *= 2
        else:

            self.aggregation = "concat"
            if "_" in self.embedding_net:
                self.embedding_net, self.aggregation = self.embedding_net.split("_")
            if self.aggregation != "concat":
                self.output_dim //= num_levels

            if self.embedding_net == 'resnet':
                self.net = ResFieldNet50Hierarchical(3, level_dim)
            elif self.embedding_net == 'unet14':
                print('********************** Using UNet14 **********************')
                self.net = MinkFieldUNet14A(3, level_dim)
            elif self.embedding_net == 'minkvae':
                print('********************** Using MinkVAE **********************')
                self.net = MinkVAE(3, 64, 64, 8)
            elif self.embedding_net == 'minkvqvae':
                print('********************** Using MinkVQ-VAE **********************')
                self.net = MinkVQVAE(
                    in_channels=3,
                    z_channels=256,
                    num_emb=8192,
                    gaussian_channels=4,
                    emb_channels=8,
                )
            elif self.embedding_net == 'unet34':
                print('********************** Using UNet34 **********************')
                self.net = MinkFieldUNet34A(3, level_dim)
            elif self.embedding_net == 'unet50':
                print('********************** Using UNet50 **********************')
                self.net = MinkFieldUNet50Small(3, level_dim)
            elif self.embedding_net == 'styleunet':
                print('********************** Using StyleUNet14 **********************')
                self.net = StyleFieldUNet14A(3, 8, z_dim=z_dim, w_dim=z_dim, D=3)
            elif self.embedding_net == 'stylegan':
                print('********************** Using StyleGAN2_3D **********************')
                self.net = StyleGAN2_3D_Generator(z_dim=z_dim, w_dim=z_dim, num_blocks=num_gan_blocks, 
                                num_latent_mapping_layers=4, feature_out_channels=level_dim, 
                                init_seed_channels=initseed_dim)
            elif self.embedding_net == 'bicyclegan':
                print('********************** Using Bicycle-StyleGAN **********************')
                self.net = StyleGAN2_3D_Generator(z_dim=z_dim, w_dim=z_dim, num_blocks=num_gan_blocks,
                                num_latent_mapping_layers=4, feature_out_channels=level_dim, 
                                init_seed_channels=initseed_dim)
            else:
                assert(False)

        self.pts_data = None
        self.embeddings = None
        self.posterior = None
        self.vqloss = None
        self.density = {}
        self.vis_count = 0
        self.emb_count = 0
        self.time_count = 900
        self.z_mu, self.z_logvar = None, None
        self.func = lambda x: torch.cat([x, x[:1]], dim=0)
        self.opt = opt

    def __repr__(self):
        return f"GridEncoderMinkowskiHierarchical: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={[(name, tuple(param.shape)) for name, param in self.named_parameters()]} gridtype={self.gridtype} align_corners={self.align_corners}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        assert(self.pts_data is not None)
        # print(self.pts_data['pts_batch'].shape) # [B, H, W]
        # print(self.pts_data['pts_coords'].shape) # [B, H, W, 3]
        # print(self.pts_data['pts_masks'].shape) # [B, H, W]
        # print(self.pts_data['pts_rgbs'].shape) # [B, H, W, 3]
        if self.embeddings is None:
            pts = self.pts_data['pts_coords'][self.pts_data['pts_masks']]
            pts_batch = self.pts_data['pts_batch'][self.pts_data['pts_masks']].unsqueeze(-1) * 0
            pts_field = ME.TensorField(
                features=self.func(self.pts_data['pts_rgbs'][self.pts_data['pts_masks']]),
                coordinates=self.func(torch.cat([pts_batch, pts * self.scale], dim=-1)),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=pts.device,
            )

            # self-defined data
            if self.opt.feed_pc is not None:
                cat_batch = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x], dim=1)
                mat = np.loadtxt(self.opt.feed_pc)
                pts_field = ME.TensorField(
                    features=self.func(torch.from_numpy(mat[:, 3:]).to(pts.device).float() / 255.0),
                    coordinates=self.func(cat_batch(torch.from_numpy(mat[:, :3]).to(pts.device)) * self.scale),
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=pts.device,
                )

            # print('  Compute embeddings')
            self.pts_sparse = pts_field#.sparse()
            if self.embedding_net == 'stylegan':
                z = torch.randn(pts_batch.max()+1, self.net.z_dim).to(pts.device)
                self.embeddings = self.net(self.pts_sparse.sparse(), z=z, update_emas=False)
            elif self.embedding_net == 'bicyclegan':
                if self.z_mu != None and self.z_logvar != None:
                    std = torch.exp(self.z_logvar / 2)
                    z = torch.normal(mean=self.z_mu, std=std).to(pts.device)
                else:
                    z = torch.randn(pts_batch.max()+1, self.net.z_dim).to(pts.device)
                self.embeddings = self.net(self.pts_sparse.sparse(), z=z, update_emas=False)
            elif self.embedding_net == 'styleunet':
                z = torch.randn(pts_batch.max()+1, self.net.z_dim).to(pts.device)
                self.embeddings = self.net(self.pts_sparse, z=z)
            elif self.embedding_net == 'minkvae':
                self.embeddings, self.posterior = self.net(self.pts_sparse)
            elif self.embedding_net == 'minkvqvae':
                self.embeddings, self.vqloss, _ = self.net(self.pts_sparse)
            else:
                self.embeddings = self.net(self.pts_sparse)
            # for emb in self.embeddings:
            #     print(emb.F.shape)

            if self.embedding_regu == 'tanh':
                self.embeddings = tuple([replace_features(embedding, 10 * F.tanh(embedding.F / 10.0)) for embedding in self.embeddings])
            elif self.embedding_regu == 'normal':
                def gaussian_norm(net_output):
                    mu = torch.mean(net_output, dim=0)
                    sigma = torch.std(net_output, dim=0)
                    return torch.clamp((net_output - mu) / sigma, -3.0, 3.0)
                self.embeddings = tuple([replace_features(embedding, gaussian_norm(embedding.F)) for embedding in self.embeddings])
            elif self.embedding_regu == 'normal_loss':
                self.emb_mus = [torch.mean(embedding.F, dim=0) for embedding in self.embeddings]
                self.emb_vars = [torch.var(embedding.F, dim=0) for embedding in self.embeddings]
            else:
                assert(self.embedding_regu == 'none')

            if False:
                to_save = {
                    "features": [emb.F for emb in self.embeddings],
                    "coordinates": [emb.C for emb in self.embeddings],
                }

                torch.save(to_save, f'embeddings+minkunet14_gau_20/{self.emb_count}.pt')
                self.emb_count += 1
            
            if False:
                print("Load emb from file")
                if self.time_count == -100:
                    self.emb_count += 1
                    self.time_count = 900
                filename = f'embeddings+minkunet14_20/pred_{self.emb_count}_{self.time_count}1stlayer.pt'
                low_to_high = False
                # filename = f'embeddings+holicity+fineres+fieldnet/overfit0/pred_0_clip8_{self.emb_count}.pt'
                # filename = f'embeddings+minkunet14_gau_20/{self.emb_count}.pt'
                # filename = f'embeddings+minkunet14_tanh_20/pred_4_{self.emb_count}_overfit5.pt'
                d = torch.load(filename)
                self.time_count -= 100
                if self.time_count == 0:
                    self.time_count = 1
                if self.time_count == -99:
                    self.time_count = 0
                # self.emb_count += 1
                embs = []
                stride = 8 if low_to_high else 1
                for fts, coords in zip(d["features"], d["coordinates"]):
                    embs.append(
                        ME.SparseTensor(
                            fts, coordinates=coords, device=d["features"][0].device, tensor_stride=stride,
                            coordinate_manager=embs[0].coordinate_manager if len(embs) else None,
                        )
                    )
                    # print(embs[-1].F.min(dim=0))
                    # print(embs[-1].F.max(dim=0))
                    # print(embs[-1].F.shape)
                    # input()
                    if low_to_high:
                        stride *= 2
                    else:
                        stride //= 2
                
                if low_to_high:
                    self.embeddings = tuple(embs)
                else:
                    print("here")
                    # self.embeddings = tuple(embs[::-1])
                    self.embeddings = self.embeddings[:-1] + (embs[0], )

            if self.dual:
                pts_field = ME.TensorField(
                    features=self.func(self.pts_data['pts_rgbs'][self.pts_data['pts_masks']]),
                    coordinates=self.func(torch.cat([pts_batch, pts * self.scale_dual], dim=-1)),
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=pts.device,
                )
                self.embeddings_dual = self.net_dual(pts_field)
        else:
            # print('  Use precomputed embeddings')
            pass
    
        if self.opt.visualize_point_cloud is not None:
            import os
            os.system(f"mkdir -p {self.opt.workspace}/{self.opt.visualize_point_cloud}")

            # img = (255 * self.pts_data['pts_rgbs'][0].cpu().numpy()).astype(np.uint8)
            # from PIL import Image
            # Image.fromarray(img).save(f'vis/{self.count:04d}.jpg')

            with open(f'{self.opt.workspace}/{self.opt.visualize_point_cloud}/{self.vis_count:04d}_pc.txt', 'w') as f:
                for (_, x, y, z), (r, g, b) in zip(
                    self.pts_sparse.C.cpu().numpy() / self.scale,
                    (255 * self.pts_sparse.F.cpu().numpy()).astype(np.uint8),
                ):
                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
            
            vox = self.pts_sparse.sparse()
            with open(f'{self.opt.workspace}/{self.opt.visualize_point_cloud}/{self.vis_count:04d}_vox.txt', 'w') as f:
                for (_, x, y, z), (r, g, b) in zip(
                    vox.C.cpu().numpy() / self.scale,
                    (255 * vox.F.cpu().numpy()).astype(np.uint8),
                ):
                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs[..., [2, 0, 1]]
        query = inputs / bound / 2.0 * self.aabb_size * self.scale
        query = torch.cat([torch.zeros(*(prefix_shape + [1]), device=inputs.device), query], dim=-1)
        query = query.view(-1, 4)
        outputs = []

        if self.aggregation == "concat":
            for embedding in self.embeddings:
                outputs.append(embedding.features_at_coordinates(query).view(*prefix_shape, -1))
            outputs = torch.cat(outputs, dim=-1)
        elif self.aggregation == "single":
            outputs = self.embeddings[-1].features_at_coordinates(query).view(*prefix_shape, -1)
        elif self.aggregation == "mean":
            for embedding in self.embeddings:
                outputs.append(embedding.features_at_coordinates(query).view(*prefix_shape, -1))
            outputs = torch.stack(outputs).mean(dim=0)
        elif self.aggregation == "alpha":
            # xxx in a shape of (N, C)
            add_ones = lambda tensor: torch.cat([tensor, torch.ones(tensor.shape[0], 1).to(tensor.device)], dim=1)
            weights = []
            for embedding in self.embeddings[::-1]: # fine to coarse
                emb_cat_one = replace_features(embedding, add_ones(embedding.F))
                queried = emb_cat_one.features_at_coordinates(query).view(*prefix_shape, -1)
                outputs.append(queried[:, :-1])
                weights.append(queried[:, -1:])
            
            rests = [torch.ones(query.shape[0], 1).to(query.device)]
            new_weights = []
            for i in range(len(weights)):
                new_weights.append(rests[-1] * weights[i])
                rests.append(rests[-1] * (1 - weights[i]))
            
            outputs = (torch.stack(outputs) * torch.stack(new_weights)).sum(dim=0)
        elif self.aggregation == "point":
            import frnn
            output_field = self.embeddings[-1].slice(self.pts_sparse)
            pts1 = query[:, 1:].unsqueeze(0)
            pts2 = output_field.C[:, 1:].unsqueeze(0)
            frnn_dists, frnn_idxs, frnn_nn, frnn_grid = frnn.frnn_grid_points(
                pts1, pts2,
                torch.Tensor([pts1.shape[1]]).to(pts1.device).long(),
                torch.Tensor([pts2.shape[1]]).to(pts2.device).long(),
                K=8, r=2.0, grid=None, return_nn=False, return_sorted=True,
            )
            # frnn_dists (N, P1, K)
            # frnn_idxs (N, P1, K)
            to_be_selected = torch.cat([output_field.F, output_field.F[:1] * 0.0], dim=0)
            frnn_nnft = to_be_selected[frnn_idxs.flatten()].reshape(
                frnn_idxs.shape[1], frnn_idxs.shape[2], output_field.F.shape[1]
            ) # (P1, K, C)
            frnn_weights = 1.0 / ((frnn_dists[0] + 1e-9) / self.scale / self.scale) # (P1, K) 1/d^2
            frnn_weights = torch.nn.functional.relu(frnn_weights)
            frnn_weights = torch.sqrt(frnn_weights)
            frnn_weights = frnn_weights / (frnn_weights.sum(dim=-1, keepdim=True) + 1e-9)

            outputs = torch.einsum("pkc, pk -> pc", frnn_nnft, frnn_weights)
        
        elif self.aggregation == "pointnormal":
            import frnn
            search_r = 2.0

            normals = self.func(self.pts_data['pts_normal'][self.pts_data['pts_masks']])

            output_field = self.embeddings[-1].slice(self.pts_sparse)
            pts1 = query[:, 1:].unsqueeze(0)
            pts2 = output_field.C[:, 1:].unsqueeze(0)
            frnn_dists2, frnn_idxs, frnn_nn, frnn_grid = frnn.frnn_grid_points(
                pts1, pts2,
                torch.Tensor([pts1.shape[1]]).to(pts1.device).long(),
                torch.Tensor([pts2.shape[1]]).to(pts2.device).long(),
                K=8, r=search_r, grid=None, return_nn=True, return_sorted=True,
            )
            # pts1 (N, P1, 3)
            # frnn_dists2 (N, P1, K)  -1 for invalid
            # frnn_idxs (N, P1, K)  -1 for invalid
            # frnn_nn (N, P1, K, 3) (0 0 0) for invalid
            ft_to_sel = torch.cat([output_field.F, output_field.F[:1] * 0.0], dim=0) # for index -1
            frnn_nnft = ft_to_sel[frnn_idxs.flatten()].reshape(
                frnn_idxs.shape[1], frnn_idxs.shape[2], output_field.F.shape[1]
            ) # (P1, K, C)

            nml_to_sel = torch.cat([normals, normals[:1] * 0.0], dim=0) # for index -1
            frnn_nnnml = nml_to_sel[frnn_idxs.flatten()].reshape(
                frnn_idxs.shape[1], frnn_idxs.shape[2], 3
            ) # (P1, K, 3) (0 0 0) for invalid
            frnn_valid = (frnn_idxs[0] >= 0).float() # (P1, K)
            frnn_invalid = 1.0 - frnn_valid
            #           (P1, 1, 3)             (P1, K, 3)
            pts1_diff = pts1[0].unsqueeze(1) - frnn_nn[0] # (P1, K, 3)
            dir_norm2 = (pts1_diff * frnn_nnnml).sum(dim=-1) ** 2 # (P1, K) 0 for invalid or in-plane
            pln_norm2 = frnn_dists2[0] - dir_norm2 # (P1, K) <=-1 for invalid

            dir_norm = frnn_valid * torch.sqrt(dir_norm2) + frnn_invalid * search_r # search_r for invalid
            pln_norm2 = frnn_valid * (pln_norm2 + 1e-9) - frnn_invalid # -1 for invalid

            frnn_weights = 1.0 / (pln_norm2 / self.scale / self.scale) # (P1, K) 1/d^2
            frnn_weights = torch.sqrt(torch.nn.functional.relu(frnn_weights)) # (P1, K) 1/d
            frnn_weights = frnn_weights / (frnn_weights.sum(dim=-1, keepdim=True) + 1e-9)

            weak_weights = torch.clamp((1 - (dir_norm / search_r) ** 8) ** 8, 0.0, 1.0)

            weak_nnft = frnn_nnft * weak_weights.unsqueeze(-1)
            outputs = torch.einsum("pkc, pk -> pc", weak_nnft, frnn_weights)

        else:
            assert(False)
        
        if self.dual:
            query *= float(np.sqrt(2))
            for embedding in self.embeddings_dual:
                outputs.append(embedding.features_at_coordinates(query).view(*prefix_shape, -1))

        return outputs

    def density_query(self, inputs, bound, n_voxel):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]
        assert(self.pts_data is not None)
        assert(type(self.density) == dict)
        # offset = torch.Tensor([
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 1],
        #     [0, 0, 1, 0],
        #     [0, 0, 1, 1],
        #     [0, 1, 0, 0],
        #     [0, 1, 0, 1],
        #     [0, 1, 1, 0],
        #     [0, 1, 1, 1],
        # ]).to(inputs.device)
        if n_voxel not in self.density:
            pts = self.pts_data['pts_coords'][self.pts_data['pts_masks']]
            pts_batch = self.pts_data['pts_batch'][self.pts_data['pts_masks']].unsqueeze(-1) * 0
            coords = torch.cat([pts_batch, pts / self.aabb_size * n_voxel], dim=-1)

            # self-defined data
            if self.opt.feed_pc is not None:
                mat = np.loadtxt(self.opt.feed_pc)
                coords = torch.from_numpy(mat[:, :3]).to(pts.device).float()
                coords = self.func(torch.cat([torch.zeros_like(coords[:, :1]), coords / self.aabb_size * n_voxel], dim=-1))

            # coords = torch.repeat_interleave(torch.floor(coords), 8, dim=0) + offset.repeat(coords.shape[0], 1)

            self.density[n_voxel] = ME.TensorField(
                features=self.func(torch.ones_like(coords[:, :1])),
                coordinates=self.func(coords),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=pts.device,
            )

            try:
                self.density[n_voxel] = self.density[n_voxel].sparse()
            except:
                print(coords)
                with open(f'query_density.txt', 'w') as f:
                    for (_, x, y, z) in self.density[n_voxel].C.cpu().numpy():
                        f.write('%.3lf %.3lf %.3lf\n' % (x, y, z))
                input("see see")
        
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs[..., [2, 0, 1]]
        query = inputs / bound / 2.0 * n_voxel
        query = torch.cat([torch.zeros(*(prefix_shape + [1]), device=inputs.device), query], dim=-1)
        query = query.view(-1, 4)
        query_density = self.density[n_voxel].features_at_coordinates(query).view(*prefix_shape, -1)

        if self.opt.visualize_point_cloud is not None:
            # hist, edges = np.histogram(query_density[query_density > 0].cpu().numpy(), bins=100, range=(0.0, 1.0))
            # plt.plot(edges[:-1], hist)
            # plt.savefig(f"{self.opt.workspace}/{self.opt.visualize_point_cloud}/density{n_voxel}.jpg")
            # plt.clf()
            import os
            os.system(f"mkdir -p {self.opt.workspace}/{self.opt.visualize_point_cloud}")
            vis_query = query / n_voxel * self.aabb_size
            vis_mask = (query_density > 0)[..., 0]
            density_colored = np.clip(query_density[..., 0][vis_mask].cpu().numpy(), 0.0, 1.0)
            density_colored = plt.get_cmap("viridis")(density_colored)
            density_colored = (density_colored * 255.0).astype(np.uint8)
            with open(f'{self.opt.workspace}/{self.opt.visualize_point_cloud}/{self.vis_count:04d}_query_density{n_voxel}.txt', 'w') as f:
                for (_, x, y, z), (r, g, b, _) in zip(vis_query[vis_mask].cpu().numpy(), density_colored):
                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        return query_density


class GridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear'):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation] # "linear" or "smoothstep"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.
        
        D = self.input_dim
        C = self.embeddings.shape[1] # embedding dim for each level
        L = self.offsets.shape[0] - 1 # level
        S = np.log2(self.per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')

        _backend.grad_total_variation(inputs, self.embeddings, self.embeddings.grad, self.offsets, weight, B, D, C, L, S, H, self.gridtype_id, self.align_corners)