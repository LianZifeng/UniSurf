import torch.nn as nn
import torch
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple
from torch import Tensor
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_sparse import SparseTensor, matmul
from typing_extensions import Final
from monai.utils.deprecate_utils import deprecated_arg
from collections.abc import Sequence
from monai.utils import ensure_tuple_rep, look_up_option
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets import PatchMerging, PatchMergingV2
import numpy as np
from monai.networks.nets.swin_unetr import SwinTransformer


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


"""
Swin UNETR based on: "Hatamizadeh et al.,
Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
<https://arxiv.org/abs/2201.01266>"
"""
class SwinUNETR(nn.Module):
    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


class SDFNet(nn.Module):
    def __init__(self, in_channels=3):
        super(SDFNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        self.enc5 = self.conv_block(256, 512)

        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        self.wm = nn.Sequential(
            self.conv_block(32, 32),
            self.conv_block(32, 16),
            nn.Conv3d(16, 1, kernel_size=1)
        )
        self.pial = nn.Sequential(
            self.conv_block(32, 32),
            self.conv_block(32, 16),
            nn.Conv3d(16, 1, kernel_size=1)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(x5), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        wm = self.wm(d1)
        pial = self.pial(d1)
        x = torch.cat([wm, pial], dim=1)

        return x


def compute_normal(v, f):
    normal = Meshes(verts=list(v), faces=list(f)).verts_normals_list()[0].unsqueeze(0)
    return normal


"""
ACKNOWLEDGEMENT AND CITATION:

The PialNet is adapted from the official implementation of 
"PialNN: A Fast Deep Learning Framework for Cortical Pial Surface Reconstruction".
- Repository: https://github.com/m-qiang/PialNN
- Paper: Qiang Ma et al., International Workshop on Machine Learning in Clinical Neuroimaging 2021

If you use this code in your research, please consider citing the original works:

@inproceedings{ma2021pialnn,
  title={PialNN: A fast deep learning framework for cortical pial surface reconstruction},
  author={Ma, Qiang and Robinson, Emma C and Kainz, Bernhard and Rueckert, Daniel and Alansary, Amir},
  booktitle={Machine Learning in Clinical Neuroimaging: 4th International Workshop, MLCN 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27, 2021, Proceedings 4},
  pages={73--81},
  year={2021},
  organization={Springer}
"""
class PialNet(nn.Module):
    def __init__(self, nc=128, K=5, n_scale=3):
        super(PialNet, self).__init__()
        self.block1 = DeformBlock(nc, K, n_scale)
        self.block2 = DeformBlock(nc, K, n_scale)
        self.block3 = DeformBlock(nc, K, n_scale)
        self.smooth = LaplacianSmooth(3, 3, aggr='mean')

    def forward(self, v, f, volume, n_smooth=1, lambd=1.0):
        x = self.block1(v, f, volume)
        x = self.block2(x, f, volume)
        x = self.block3(x, f, volume)
        edge_list = torch.cat([f[0, :, [0, 1]], f[0, :, [1, 2]], f[0, :, [2, 0]]], dim=0).transpose(1, 0)

        for i in range(n_smooth):
            x = self.smooth(x, edge_list, lambd=lambd)

        return x

    def initialize(self, L=256, W=256, H=256):
        self.block1.initialize(L, W, H)
        self.block2.initialize(L, W, H)
        self.block3.initialize(L, W, H)


class DeformBlock(nn.Module):
    def __init__(self, nc=128, K=5, n_scale=3):
        super(DeformBlock, self).__init__()

        # MLP layers
        self.fc1 = nn.Linear(6, nc)
        self.fc2 = nn.Linear(nc * 2, nc * 4)
        self.fc3 = nn.Linear(nc * 4, nc * 2)
        self.fc4 = nn.Linear(nc * 2, 3)

        # for local convolution operation
        self.localconv = nn.Conv3d(n_scale, nc, (K, K, K))
        self.localfc = nn.Linear(nc, nc)

        self.n_scale = n_scale
        self.nc = nc
        self.K = K

    def forward(self, v, f, volume):
        coord = v.clone()
        normal = compute_normal(v, f)

        # point feature
        x = torch.cat([v, normal], 2)
        x = F.leaky_relu(self.fc1(x), 0.15)

        # local feature
        cubes = self.cube_sampling(v, volume)
        x_local = self.localconv(cubes)
        x_local = x_local.view(1, v.shape[1], self.nc)
        x_local = self.localfc(x_local)

        # fusion
        x = torch.cat([x, x_local], 2)
        x = F.leaky_relu(self.fc2(x), 0.15)
        x = F.leaky_relu(self.fc3(x), 0.15)
        x = torch.tanh(self.fc4(x)) * 0.1

        return coord + x

    def initialize(self, L, W, H):
        LWHmax = max([L, W, H])
        self.LWHmax = LWHmax
        # rescale to [-1, 1]
        self.rescale = torch.Tensor([L / LWHmax, W / LWHmax, H / LWHmax]).cuda()

        # shape of mulit-scale image pyramid
        self.pyramid_shape = torch.zeros([self.n_scale, 3]).cuda()
        for i in range(self.n_scale):
            self.pyramid_shape[i] = torch.Tensor([L / (2 ** i), W / (2 ** i), H / (2 ** i)]).cuda()

        # for threshold
        self.lower_bound = torch.tensor([(self.K - 1) // 2, (self.K - 1) // 2, (self.K - 1) // 2]).cuda()

        # for storage of sampled cubes
        self.cubes_holder = torch.zeros([1, self.n_scale, self.K, self.K, self.K]).cuda()

    def cube_sampling(self, v, volume):
        # for storage of sampled cubes
        cubes = self.cubes_holder.repeat(v.shape[1], 1, 1, 1, 1)

        # 3D MRI volume
        vol_ = volume.clone()
        for n in range(self.n_scale):
            if n > 0:
                vol_ = F.avg_pool3d(vol_, 2)
            vol = vol_[0, 0]

            # find corresponding position
            indices = (v[0] + self.rescale) * self.LWHmax / (2 ** (n + 1))
            indices = torch.round(indices).long()
            indices = torch.max(torch.min(indices, self.pyramid_shape[n] - 3), self.lower_bound).long()

            # sample values of each cube
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    for k in [-2, -1, 0, 1, 2]:
                        cubes[:, n, 2 + i, 2 + j, 2 + k] = vol[indices[:, 0] + i, indices[:, 1] + j, indices[:, 2] + k]

        return cubes


class LaplacianSmooth(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: str = 'add',
                 bias: bool = True,
                 **kwargs):
        super(LaplacianSmooth, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

    def forward(self, x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_weight: OptTensor = None,
                size: Size = None,
                lambd=0.5) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = lambd * out
        x_r = x[1]
        if x_r is not None:
            out += (1 - lambd) * x_r

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,  x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)