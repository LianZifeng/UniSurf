import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
from seg2surf import topo_correct, laplacian_smooth
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import time
import argparse


class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()
        self.enc1 = self.conv_block(1, 32)
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

        self.final = nn.Sequential(
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
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        d4 = self.dec4(torch.cat([self.up4(x5), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        x = self.final(d1)
        return x


def processvolume(tissue):
    return np.where(tissue == 250, 1, 0).astype(np.int64)


def processSDF(SDF):
    return np.clip(SDF, a_min=-5, a_max=5)


def processsurface(v, affine, shape=(224, 256, 192)):
    v_ras_h = np.hstack([v, np.ones((v.shape[0], 1))])
    affine_inv = np.linalg.inv(affine)
    v_voxel = (affine_inv @ v_ras_h.T).T[:, :3]
    D1, D2, D3 = shape
    D = max(D1, D2, D3)
    v = (2 * v_voxel - [D1, D2, D3]) / D
    return v


class LoadSurface(Dataset):
    def __init__(self, surf_path, surf_hemi):
        self.surf_path = surf_path
        self.surf_hemi = surf_hemi
        self.surf_list = [folder for folder in os.listdir(surf_path)]

    def __len__(self):
        return len(self.surf_list)

    def __getitem__(self, idx):
        filename = self.surf_list[idx]
        filepath = os.path.join(self.surf_path, filename)
        brain = nib.load(os.path.join(filepath, "brain.nii.gz")).get_fdata()
        affine = nib.load(os.path.join(filepath, "brain.nii.gz")).affine
        if self.surf_hemi == 'left':
            tissue = nib.load(os.path.join(filepath, "lh.nii.gz")).get_fdata().astype(np.int64)
        elif self.surf_hemi == 'right':
            tissue = nib.load(os.path.join(filepath, "rh.nii.gz")).get_fdata().astype(np.int64)
        tissue = processvolume(tissue)
        input = np.stack([brain, tissue], axis=0)
        SDF = nib.load(os.path.join(filepath, "lh.white.SDF.nii.gz")).get_fdata()
        SDF = processSDF(SDF)
        if self.surf_hemi == 'left':
            v_ras, f = nib.freesurfer.read_geometry(os.path.join(filepath, "lh.white.surf"))
        elif self.surf_hemi == 'right':
            v_ras, f = nib.freesurfer.read_geometry(os.path.join(filepath, "rh.white.surf"))
        f = f.astype(np.int64)
        v = processsurface(v_ras, affine)
        item = {
            'image': torch.Tensor(input).float(),
            'SDF': torch.Tensor(SDF).unsqueeze(0).float(),
            'v': torch.Tensor(v).float(),
            'f': torch.LongTensor(f),
            'filename': filename
        }
        return item


def sdf2surf(sdf, alpha=16, level=0, n_smooth=2, lambd=1.):
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lewiner')
    v_mc, f_mc = v_mc.copy(), f_mc.copy()
    D1, D2, D3 = sdf.shape
    D = max(D1, D2, D3)
    v_mc = (2 * v_mc - [D1, D2, D3]) / D
    v_mc, f_mc = torch.Tensor(v_mc).unsqueeze(0).cuda(), torch.LongTensor(f_mc).unsqueeze(0).cuda()
    for j in range(n_smooth):
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=lambd)
    v_mc, f_mc = v_mc[0].cpu().numpy(), f_mc[0].cpu().numpy()
    return v_mc, f_mc


class DiffMarchingCubes(nn.Module):
    def __init__(self):
        super(DiffMarchingCubes, self).__init__()

    def forward(self, sdf):
        # Detach from graph: The mesh extraction is just a forward pass function
        sdf_np = sdf.squeeze().detach().cpu().numpy().astype(float)
        v_mc, f_mc = sdf2surf(sdf_np)
        return v_mc, f_mc


def calc_SDF_grad(SDF, v, v_grad, normals):
    dL_ds_v = -torch.sum(v_grad * normals, dim=1)

    grid_size = torch.tensor(SDF.shape[2:], device=v.device)
    D = torch.max(grid_size).item()
    offsets = grid_size.float()
    v_grid = ((v * D + offsets) / 2).long()
    v_grid = torch.clamp(v_grid, torch.zeros_like(grid_size - 1), grid_size - 1)

    SDF_grad = torch.zeros_like(SDF)
    # for i in range(v.size(0)):
    #     idx = v_grid[i]
    #     SDF_grad[0, 0, idx[0], idx[1], idx[2]] += dL_ds_v[i]
    idx_x, idx_y, idx_z = v_grid[:, 0], v_grid[:, 1], v_grid[:, 2]
    indices = idx_x * (grid_size[1] * grid_size[2]) + idx_y * grid_size[2] + idx_z
    SDF_grad_flat = SDF_grad.view(1, 1, -1)
    SDF_grad_flat.scatter_add_(2, indices.view(1, 1, -1), dL_ds_v.view(1, 1, -1))

    return SDF_grad


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def cosine_lr(optimizer, warmup_length, steps, warmup_lr, base_lr):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = warmup_lr
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


"""
ACKNOWLEDGEMENT AND CITATION:

This is a demo application of the differential iso-surface extraction algorithm for training WM surface reconstruction.
This differentiable iso-surface extraction algorithm is adapted from the official 
implementation of "MeshSDF: Differentiable Iso-Surface Extraction".
- Repository: https://github.com/cvlab-epfl/MeshSDF
- Paper: Remelli et al., NeurIPS 2020

If you use this code in your research, please consider citing the original works:

@inproceedings{remelli2020meshsdf,
  title={MeshSDF: Differentiable Iso-Surface Extraction},
  author={Remelli, Edoardo and Lukoianov, Artem and Richter, Stephan R and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
"""
def trainer(config):
    train_dataset = LoadSurface(config.train_path, config.hemi)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    model = SDFNet().cuda()
    MC = DiffMarchingCubes()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.warmup_lr)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup * len(train_loader)
    scheduler = cosine_lr(optimizer, warmup_steps, total_steps, config.warmup_lr, config.base_lr)
    L2loss = nn.MSELoss()
    model.train()
    global_step = 0
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        for idx, data in enumerate(train_loader):
            batch_start_time = time.time()
            image, sdf_pseudo, v_gt, f_gt, ID = data['image'].cuda(), data['SDF'].cuda(), data['v'].cuda(), data['f'].cuda(), data['filename'][0]
            optimizer.zero_grad()
            output = model(image)
            current_lr = scheduler(global_step)
            global_step += 1
            # warm up learning for SDF using pseudo ground truth supervision and L2loss
            if epoch < config.warmup:
                loss = L2loss(output, sdf_pseudo)
                loss.backward()
                print(f"epoch {epoch + 1}/{config.epochs} batch {ID}/{idx+1}/{len(train_loader)} took {time.time()-batch_start_time:.2f}s, l2_loss={loss.item():.6f}, lr={current_lr:.8f}")
            else:
                v_mc, f_mc = MC(output)

                v_0 = torch.tensor(v_mc, dtype=torch.float32, requires_grad=True, device='cuda')
                f_0 = torch.tensor(f_mc, dtype=torch.long, requires_grad = False, device='cuda')

                pred_mesh = Meshes(verts=[v_0], faces=[f_0])
                gt_mesh = Meshes(verts=[v_gt.squeeze(0)], faces=[f_gt.squeeze(0)])

                pred_points = sample_points_from_meshes(pred_mesh, num_samples=200000)
                gt_points = sample_points_from_meshes(gt_mesh, num_samples=200000)

                loss = 1e3 * chamfer_distance(pred_points, gt_points)[0]
                loss.backward()

                dL_dv = v_0.grad

                # Compute normals
                optimizer.zero_grad()

                D1, D2, D3 = output.shape[2:]
                D = max(D1, D2, D3)
                v_1 = ((v_mc * D) + [D1, D2, D3]) / 2.0
                v_1 = v_1 / np.array([D1 - 1, D2 - 1, D3 - 1]) * 2.0 - 1.0

                v_1 = torch.tensor(v_1, dtype=torch.float32, requires_grad=True, device='cuda')

                v_SDF = F.grid_sample(output.permute(0, 1, 4, 3, 2), v_1.unsqueeze(0).unsqueeze(-2).unsqueeze(-2), mode='bilinear', padding_mode='border', align_corners=True)

                normals_loss = torch.sum(v_SDF)

                normals_loss.backward(retain_graph = True)

                normals = v_1.grad / torch.norm(v_1.grad, dim=-1, keepdim=True).clamp(min=1e-6)

                # Compute gradient w.r.t. SDF
                optimizer.zero_grad()

                v_2 = torch.tensor(v_mc, dtype=torch.float32, requires_grad=True, device='cuda')

                SDF_grad = calc_SDF_grad(output, v_2, dL_dv, normals)

                # Backpropagate gradient to network parameters
                dummy_loss = torch.sum(output * SDF_grad)

                dummy_loss.backward(retain_graph=True)

                print(f"epoch {epoch + 1}/{config.epochs} batch {ID}/{idx+1}/{len(train_loader)} took {time.time()-batch_start_time:.2f}s, chamfer={loss.item():6f}")

            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        print(f"epoch {epoch+1} completed in {epoch_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.train_path = r"./train"
    config.hemi = 'left'
    config.warmup_lr = 1e-4
    config.save_path = r'./model/demo/experiment_0'
    config.epochs = 100
    config.warmup = 20
    config.base_lr = 1e-6
    trainer(config)