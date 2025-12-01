from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torch
import pandas as pd


def process_affine(affine, surf_hemi):
    p_affine = affine.copy()

    if surf_hemi == 'right':
        offset_voxel = np.array([96, 0, 0, 1])
        offset_physical = affine @ offset_voxel
        p_affine[:3, 3] = offset_physical[:3]

    return p_affine


def process_volume(brain, surf_hemi):
    if surf_hemi == 'left':
        p_brain = brain[:128, :, :]
    elif surf_hemi == 'right':
        p_brain = brain[96:, :, :]
    else:
        raise ValueError("surf_hemi must be 'left' or 'right'")

    mask = p_brain > 0
    mean = p_brain[mask].mean()
    std = p_brain[mask].std()
    return (p_brain - mean) / std


def process_label(tissue):
    tissue[tissue == 10] = 0
    tissue[tissue == 150] = 1
    tissue[tissue == 250] = 2
    return tissue


def process_edge(edge, surf_hemi):
    if surf_hemi == 'left':
        p_edge = edge[:128, :, :]
    elif surf_hemi == 'right':
        p_edge = edge[96:, :, :]
    else:
        raise ValueError("surf_hemi must be 'left' or 'right'")

    return p_edge


def process_SDF(SDF):
    return np.clip(SDF, a_min=-5, a_max=5)


# Transform coordinates from world space to voxel space
# Normalized coordinates
def process_surface(v, affine, shape):
    v = np.hstack([v, np.ones((v.shape[0], 1))])
    affine_inv = np.linalg.inv(affine)
    v = (affine_inv @ v.T).T[:, :3]
    D1, D2, D3 = shape
    D = max(D1, D2, D3)
    v = (2 * v - [D1, D2, D3]) / D
    return v


class LoadDataset(Dataset):
    def __init__(self, data_path, excel_path, surf_hemi, mode='training'):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.surf_hemi = surf_hemi

        self.surf_list = pd.read_excel(excel_path, sheet_name=mode)['FolderName'].astype(str).tolist()

    def __len__(self):
        return len(self.surf_list)

    def __getitem__(self, idx):
        filename = self.surf_list[idx]
        filepath = os.path.join(self.data_path, filename)

        affine = nib.load(os.path.join(filepath, "brain.nii.gz")).affine
        affine = process_affine(affine, self.surf_hemi)

        brain = nib.load(os.path.join(filepath, "brain.nii.gz")).get_fdata()
        brain = process_volume(brain, self.surf_hemi)
        shape = brain.shape

        if self.surf_hemi == 'left':
            tissue = nib.load(os.path.join(filepath, "lh.nii.gz")).get_fdata().astype(np.int32)
            tissue = tissue[:128, :, :]
        elif self.surf_hemi == 'right':
            tissue = nib.load(os.path.join(filepath, "rh.nii.gz")).get_fdata().astype(np.int32)
            tissue = tissue[96:, :, :]
        else:
            raise ValueError("surf_hemi must be 'left' or 'right'")
        
        tissue = process_label(tissue)

        edge = nib.load(os.path.join(filepath, "edge.nii.gz")).get_fdata()
        edge = process_edge(edge, self.surf_hemi)

        image = np.stack([brain, edge], axis=0)

        # Load pseudo-SDF GT
        if self.surf_hemi == 'left':
            inner_SDF = nib.load(os.path.join(filepath, "lh.white.SDF.nii.gz")).get_fdata()
            outer_SDF = nib.load(os.path.join(filepath, "lh.pial.SDF.nii.gz")).get_fdata()
            inner_SDF = inner_SDF[:128, :, :]
            outer_SDF = outer_SDF[:128, :, :]
            inner_SDF = process_SDF(inner_SDF)
            outer_SDF = process_SDF(outer_SDF)
        elif self.surf_hemi == 'right':
            inner_SDF = nib.load(os.path.join(filepath, "rh.white.SDF.nii.gz")).get_fdata()
            outer_SDF = nib.load(os.path.join(filepath, "rh.pial.SDF.nii.gz")).get_fdata()
            inner_SDF = inner_SDF[96:, :, :]
            outer_SDF = outer_SDF[96:, :, :]
            inner_SDF = process_SDF(inner_SDF)
            outer_SDF = process_SDF(outer_SDF)
        else:
            raise ValueError("surf_hemi must be 'left' or 'right'")

        # Load surface GT
        if self.surf_hemi == 'left':
            inner_v, inner_f = nib.freesurfer.read_geometry(os.path.join(filepath, "lh.white"))
            outer_v, outer_f = nib.freesurfer.read_geometry(os.path.join(filepath, "lh.pial"))
            inner_v = process_surface(inner_v, affine, shape)
            outer_v = process_surface(outer_v, affine, shape)
            inner_f = inner_f.astype(np.int32)
            outer_f = outer_f.astype(np.int32)
        elif self.surf_hemi == 'right':
            inner_v, inner_f = nib.freesurfer.read_geometry(os.path.join(filepath, "rh.white"))
            outer_v, outer_f = nib.freesurfer.read_geometry(os.path.join(filepath, "rh.pial"))
            inner_v = process_surface(inner_v, affine, shape)
            outer_v = process_surface(outer_v, affine, shape)
            inner_f = inner_f.astype(np.int32)
            outer_f = outer_f.astype(np.int32)
        else:
            raise ValueError("surf_hemi must be 'left' or 'right'")

        data = {
            'affine': affine,
            'image': torch.Tensor(image).float(),
            'label': torch.Tensor(tissue).long(),
            'inner_SDF': torch.Tensor(inner_SDF).unsqueeze(0).float(),
            'outer_SDF': torch.Tensor(outer_SDF).unsqueeze(0).float(),
            'inner_v': torch.Tensor(inner_v).float(),
            'outer_v': torch.Tensor(outer_v).float(),
            'inner_f': torch.LongTensor(inner_f),
            'outer_f': torch.LongTensor(outer_f),
            'ID': filename
        }

        return data