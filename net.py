import torch.nn as nn
import torch
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple
from torch import Tensor
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_sparse import SparseTensor, matmul


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

This network architecture is adapted from "PialNN: A Fast Deep Learning Framework 
for Cortical Pial Surface Reconstruction" by Ma Qiang et al.

Original Project Repository: https://github.com/m-qiang/PialNN
Original Paper: https://arxiv.org/abs/2109.03693 (MLCN 2021)

If you use this code in your research, please cite the original paper:
Ma, Q., Robinson, E.C., Kainz, B., Rueckert, D., Alansary, A. (2021). 
PialNN: A Fast Deep Learning Framework for Cortical Pial Surface Reconstruction. 
In: Machine Learning in Clinical Neuroimaging. MLCN 2021.
"""
class PialNet(nn.Module):
    def __init__(self, nc, K, n_scale):
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