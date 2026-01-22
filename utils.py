import torch
import torch.nn as nn
from tca import topology
from skimage.measure import marching_cubes
import numpy as np
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes

point_face_distance = _PointFaceDistance.apply


def point_to_mesh_dist(pcls, meshes):
    points = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face.sqrt()


def compute_mesh_distance(v_pred, v_gt, f_pred, f_gt, n_pts):
    mesh_pred = Meshes(verts=list(v_pred), faces=list(f_pred))
    mesh_gt = Meshes(verts=list(v_gt), faces=list(f_gt))
    pts_pred = sample_points_from_meshes(mesh_pred, num_samples=n_pts)
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=n_pts)
    pcl_pred = Pointclouds(pts_pred)
    pcl_gt = Pointclouds(pts_gt)
    x_dist = point_to_mesh_dist(pcl_pred, mesh_gt)
    y_dist = point_to_mesh_dist(pcl_gt, mesh_pred)
    assd = (x_dist.mean().item() + y_dist.mean().item()) / 2
    x_quantile = torch.quantile(x_dist, 0.9).item()
    y_quantile = torch.quantile(y_dist, 0.9).item()
    hd = max(x_quantile, y_quantile)
    return assd, hd


"""
ACKNOWLEDGEMENT AND CITATION:

This SDF→Surface algorithm is proposed in:
"CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs".

Original Project Repository: https://github.com/m-qiang/CortexODE
Paper: Ma, Q. et al., "CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs", 
       IEEE Transactions on Medical Imaging, 2022.

If you find this code useful or use it in your research, please cite the original paper:

@article{ma2022cortexode,
  title={CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs},
  author={Ma, Qiang and Li, Liu and Robinson, Emma C and Kainz, Bernhard and Rueckert, Daniel and Alansary, Amir},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={10},
  pages={2942--2953},
  year={2022},
  publisher={IEEE}
}
"""

# initialize topology correction
topo_correct = topology()


def laplacian_smooth(verts, faces, method="uniform", lambd=1.):
    """
    Laplacian smoothing based on pytorch3d.loss.mesh_laplacian_smoothing.
    For the original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    """

    v = verts[0]
    f = faces[0]

    with torch.no_grad():
        if method == "uniform":
            V = v.shape[0]
            edge = torch.cat([f[:, [0, 1]],
                              f[:, [1, 2]],
                              f[:, [2, 0]]], dim=0).T
            L = torch.sparse_coo_tensor(edge, torch.ones_like(edge[0]).float(), (V, V))
            norm_w = 1.0 / torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)

        elif method == "cot":
            L = laplacian_cot(v, f)
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]

    v_bar = L.mm(v) * norm_w  # new vertices
    return ((1 - lambd) * v + lambd * v_bar).unsqueeze(0)


def laplacian_cot(verts, faces):
    """
    Laplacian cotangent weights based on pytorch3d.ops.cot_laplacian.
    For the original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html

    Note that in previous version (v0.4.0) this function is defined
    in pytorch3d.loss.mesh_laplacian_smoothing.
    """

    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L += L.t()

    return L


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
        sdf_np = sdf.squeeze().detach().cpu().numpy().astype(float)
        v_mc, f_mc = sdf2surf(sdf_np)
        return v_mc, f_mc


"""
ACKNOWLEDGEMENT AND CITATION:

This is the core part of the differential iso-surface extraction algorithm.
It adapted from the official implementation of "MeshSDF: Differentiable Iso-Surface Extraction".
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


def denormalize(v, shape):
    """
    v: (N,3) in [-1,1]
    shape: (D1, D2, D3)
    returns: (N, 3) voxel coords
    """
    D1, D2, D3 = shape
    size = np.array([D1, D2, D3], dtype=np.float32)
    D = max(shape)
    return (v * D + size) / 2


def voxel2ras(v, affine):
    """
    v: (N,3)
    affine: 4x4
    returns: (N,3) in RAS mm
    """
    N = v.shape[0]
    hom = np.hstack([v, np.ones((N,1), dtype=np.float32)])
    ras = (affine @ hom.T).T
    return ras[:, :3]