import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import tqdm
import glob, json, os

def set_feature(x, features):
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


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class UnbatchedMinkowski(nn.Module):

    def __init__(self, module):
        super().__init__()
        # a PyTorch 1D module/layer
        self.module = module

    def forward(self, x):
        list_permutations = x.decomposition_permutations
        list_feats = list(map(
            lambda perm: self.module(
                x.F[perm].transpose(0, 1).unsqueeze(0) # shape of (1 (batch), C, N)
            )[0].transpose(0, 1), # shape of (N, C)
            list_permutations,
        ))
        ft = torch.cat(list_feats, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)
        return set_feature(x, ft[inv_perm])


class BatchedMinkowski(nn.Module):

    def __init__(self, module):
        super().__init__()
        # a PyTorch 1D module/layer
        self.module = module

    def forward(self, x):
        return set_feature(x, self.module(x.F))


class ConvBlock(nn.Module):

    def __init__(self, ch_in, ch_out, ks=3, groups=16, act="SiLU", conv_first=True):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            ch_in, ch_out, kernel_size=ks, stride=1, dilation=1, dimension=3,
        )
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=groups, num_channels=ch_out if conv_first else ch_in)
        )
        if act is None or act == "":
            self.act = Identity()
        else:
            assert(type(act) == str)
            if act == "leaky_relu":
                self.act = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            else:
                self.act = getattr(torch.nn.functional, act.lower())
            
        self.conv_first = conv_first

    def forward(self, x, scale_shift=None):
        if self.conv_first:
            x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            idx = x.C[:, 0].long() # batch index
            x = set_feature(x, x.F * (scale[idx] + 1) + shift[idx])
        x = set_feature(x, self.act(x.F))
        if not self.conv_first:
            x = self.conv(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, ch_in, ch_out, ch_time=None, act="SiLU"):
        super().__init__()
        self.mlp = nn.Sequential(
            getattr(nn, act)(),
            nn.Linear(ch_time, ch_out * 2)
        ) if ch_time is not None else None
        self.block1 = ConvBlock(ch_in, ch_out, act=act)
        self.block2 = ConvBlock(ch_out, ch_out, act=act)
        self.res_conv = ME.MinkowskiConvolution(
            ch_in, ch_out, kernel_size=1, stride=1, dilation=1, dimension=3, bias=True
        ) if ch_in != ch_out else Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb) # B, ch_out * 2
            scale_shift = time_emb.chunk(2, dim=1) # (B, ch_out) and (B, ch_out)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

# https://github.com/lucidrains/linear-attention-transformer
class LinearAttention(nn.Module):
    def __init__(self, ch, num_head=4, ch_head=32, norm=True, residual=True):
        super().__init__()
        self.num_head = num_head
        self.ch_head = ch_head
        self.scale = ch_head ** -0.5
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
        ) if norm else Identity()
        self.to_qkv = ME.MinkowskiConvolution(
            ch, ch_head * num_head * 3,
            kernel_size=1, stride=1, dilation=1, dimension=3,
        )
        self.to_out = nn.Sequential(
            ME.MinkowskiConvolution(
                ch_head * num_head, ch,
                kernel_size=1, stride=1, dilation=1, dimension=3,
            ),
            UnbatchedMinkowski(
                nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
            )
        )
        self.residual = residual

    def forward(self, x):
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x)
        list_permutations = qkv.decomposition_permutations
        
        q, k, v = map(
            lambda t: t.reshape(-1, self.num_head, self.ch_head),
            qkv.F.chunk(3, dim=1)
        ) # (spatial, num_head, ch_head)

        list_context = list(map(
            lambda perm: torch.einsum('ihk, ihv -> hkv', (k[perm] / np.sqrt(perm.size(0))).softmax(dim=0), v[perm]),
            list_permutations
        )) # list of (num_head, ch_head_key, ch_head_value)

        q = q * self.scale
        list_out = list(map(
            lambda context, perm: torch.einsum('ihk, hkv -> ihv', q[perm].softmax(dim=-1), context)
            .reshape(-1, self.num_head * self.ch_head),
            list_context, list_permutations
        )) # list of (spatial, num_head * ch_head)

        ft = torch.cat(list_out, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)
        
        out = self.to_out(set_feature(qkv, ft[inv_perm]))
        if self.residual:
            out = out + x
        
        return out


class Attention(nn.Module):

    def __init__(self, ch, num_head=4, ch_head=32, norm=True, residual=True):
        super().__init__()
        self.num_head = num_head
        self.ch_head = ch_head
        self.scale = ch_head ** -0.5
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
        ) if norm else Identity()
        self.to_qkv = ME.MinkowskiConvolution(
            ch, num_head * ch_head * 3,
            kernel_size=1, stride=1, dilation=1, dimension=3,
        )
        self.to_out = ME.MinkowskiConvolution(
            num_head * ch_head, ch,
            kernel_size=1, stride=1, dilation=1, dimension=3, bias=True
        )
        self.residual = residual

    def forward(self, x):
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x)
        list_permutations = qkv.decomposition_permutations
        
        q, k, v = map(
            lambda t: t.reshape(-1, self.num_head, self.ch_head),
            qkv.F.chunk(3, dim=1)
        ) # (spatial, num_head, ch_head)
        q = q * self.scale

        list_attn = list(map(
            lambda perm: torch.einsum('ihd, jhd -> hij', q[perm], k[perm]).softmax(dim=-1),
            list_permutations
        )) # list of (num_head, spatial, spatial)

        list_out = list(map(
            lambda attn, perm: torch.einsum('hij, jhd -> ihd', attn, v[perm]).reshape(-1, self.num_head * self.ch_head),
            list_attn, list_permutations
        )) # list of (spatial, num_head * ch_head)

        ft = torch.cat(list_out, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)

        out = self.to_out(set_feature(qkv, ft[inv_perm]))
        if self.residual:
            out = out + x

        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, times):
        half_dim = self.dim // 2
        emb = np.log(1000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
        emb = times[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


ATTN_LAYER = {
    "L": LinearAttention,
    "A": Attention,
    "N": Identity,
}


class MinkEncoder(nn.Module):
    BLOCK = ResNetBlock
    PLANES = (32, 64, 128, 256, 512)
    REPEATS = (2, 2, 2, 2, 2)
    ATTNS = "N N N N N".split()
    # "L", "A", "N" for linear/general/no attention

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(self.REPEATS) == len(self.PLANES)
        self.levels = len(self.PLANES) - 1
        self.init_network(in_channels, out_channels)
        self.init_weight()

    def init_network(self, in_channels, out_channels):
        self.init_conv = ConvBlock(in_channels, self.PLANES[0], ks=5)

        # Encoder
        self.enc_blocks = nn.ModuleList([])
        self.downs = nn.ModuleList([])
        for i in range(self.levels + 1):
            ch = self.PLANES[i]
            ch_next = self.PLANES[i + 1] if i < self.levels else None
            blocks = nn.ModuleList([])
            for j in range(self.REPEATS[i]):
                blocks.append(nn.ModuleList([
                    ResNetBlock(ch, ch),
                    ATTN_LAYER[self.ATTNS[i]](ch) if j == self.REPEATS[i] - 1 else Identity(),
                ]))
            self.enc_blocks.append(blocks)
            self.downs.append(
                ME.MinkowskiConvolution(ch, ch_next, kernel_size=2, stride=2, dimension=3)
                if i < self.levels else Identity()
            )
        
        # Mid
        mid_dim = self.PLANES[self.levels]
        self.mid_block1 = ResNetBlock(mid_dim, mid_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResNetBlock(mid_dim, mid_dim)

        self.conv_out = ConvBlock(mid_dim, out_channels, conv_first=False)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out")#, nonlinearity="relu")

    def forward(self, x):
        x = self.init_conv(x)
        for blocks, down in zip(self.enc_blocks, self.downs):
            for block, attn in blocks:
                x = block(x)
                x = attn(x)
            x = down(x)
        
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
    
        return self.conv_out(x)



class MinkFieldEncoder(MinkEncoder):

    def init_network(self, in_channels, out_channels):
        field_ch1 = 32
        field_ch2 = 32
        self.field_network1 = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch1),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=16, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
            ME.MinkowskiLinear(field_ch1, field_ch1),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=16, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch1 + in_channels, field_ch2),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=16, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=16, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
        )
        MinkEncoder.init_network(self, field_ch2, out_channels)

    def forward(self, x: ME.TensorField):
        otensor1 = self.field_network1(x)
        otensor1 = ME.cat(otensor1, x)
        otensor2 = self.field_network2(otensor1)
        otensor2 = otensor2.sparse()
        return MinkEncoder.forward(self, otensor2)


class MinkDecoder(nn.Module):
    BLOCK = ResNetBlock
    PLANES = (512, 256, 128, 64, 32)#(256, 128, 64, 32)#
    REPEATS = (2, 2, 2, 2, 2)
    ATTNS = "N N N N N".split()
    # "L", "A", "N" for linear/general/no attention

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(self.REPEATS) == len(self.PLANES)
        self.levels = len(self.PLANES) - 1
        self.init_network(in_channels, out_channels)
        self.init_weight()

    def init_network(self, in_channels, out_channels):
        self.conv_in = ME.MinkowskiConvolution(in_channels, self.PLANES[0], kernel_size=3, stride=1, dimension=3)
        
        # Mid
        mid_dim = self.PLANES[0]
        self.mid_block1 = ResNetBlock(mid_dim, mid_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResNetBlock(mid_dim, mid_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.conv_outs = nn.ModuleList([])
        for i in range(len(self.PLANES)):
            ch = self.PLANES[i]
            ch_next = self.PLANES[i + 1] if i < self.levels else None
            blocks = nn.ModuleList([])
            for j in range(self.REPEATS[i]):
                blocks.append(nn.ModuleList([
                    ResNetBlock(ch, ch),
                    ATTN_LAYER[self.ATTNS[i]](ch) if j == self.REPEATS[i] - 1 else Identity(),
                ]))
            self.dec_blocks.append(blocks)
            self.conv_outs.append(ConvBlock(ch, out_channels, conv_first=False) if i > (self.levels - 4) else Identity())
            self.ups.append(
                ME.MinkowskiConvolutionTranspose(ch, ch_next, kernel_size=2, stride=2, dimension=3)
                if i < self.levels else Identity()
            )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out")#, nonlinearity="relu")

    def forward(self, x):
        x = self.conv_in(x)
        
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        outputs = []
        for blocks, conv_out, up in zip(self.dec_blocks, self.conv_outs, self.ups):
            for block, attn in blocks:
                x = block(x)
                x = attn(x)
            outputs.append(conv_out(x))
            x = up(x)

        return outputs[-4:]



class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.F, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return set_feature(self.parameters, x)

    def kl(self, other=None):
        if self.deterministic:
            raise NotImplementedError
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            else:
                raise NotImplementedError
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                )

    def nll(self, sample):
        if self.deterministic:
            raise NotImplementedError
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
        )

    def mode(self):
        return self.mean



class MinkVAE(nn.Module):

    def __init__(self, in_channels, z_channels, gaussian_channels, emb_channels):
        super().__init__()
        self.encoder = MinkFieldEncoder(in_channels, 2 * z_channels)
        self.enc_quant_conv = ME.MinkowskiConvolution(2 * z_channels, 2 * gaussian_channels, kernel_size=1, dimension=3)
        self.dec_quant_conv = ME.MinkowskiConvolution(gaussian_channels, z_channels, kernel_size=1, dimension=3)
        self.decoder = MinkDecoder(z_channels, emb_channels)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.enc_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.dec_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return tuple(dec), posterior


from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

class MinkVQVAE(nn.Module):

    def __init__(self, in_channels, z_channels, num_emb, gaussian_channels, emb_channels):
        super().__init__()
        self.encoder = MinkFieldEncoder(in_channels, z_channels)
        self.enc_quant_conv = ME.MinkowskiConvolution(z_channels, gaussian_channels, kernel_size=1, dimension=3)
        self.quantize = VectorQuantizer(num_emb, gaussian_channels, beta=0.25,
                                        remap=None,#remap,
                                        sane_index_shape=True)
        self.dec_quant_conv = ME.MinkowskiConvolution(gaussian_channels, z_channels, kernel_size=1, dimension=3)
        self.decoder = MinkDecoder(z_channels, emb_channels)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.enc_quant_conv(h)
        mF = moments.F.transpose(0, 1)[None, ..., None] # 1, C, N, 1
        quant, emb_loss, info = self.quantize(mF)
        quant = set_feature(moments, quant[0, ..., 0].transpose(0, 1))
        return quant, emb_loss, info

    def decode(self, z):
        z = self.dec_quant_conv(z)
        return self.decoder(z)

    def forward(self, x):
        quant, diff, (_, _, ind) = self.encode(x)
        dec = self.decode(quant)
        return dec, diff, ind
