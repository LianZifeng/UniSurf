import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from seg2surf import topo_correct, laplacian_smooth
from skimage.measure import marching_cubes
import numpy as np


def softmax_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    input_ls = input.log_softmax(1)
    loss = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss
    return loss


def sigmoid_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss
    if alpha is not None:
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss
    return loss


class FocalLoss3D(nn.Module):
    def __init__(self,
                 include_background=True,
                 gamma=2.0,
                 alpha=None,
                 use_softmax=False,
                 reduction='mean'):
        super(FocalLoss3D, self).__init__()
        self.include_background = include_background
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, input, target):
        n_pred_ch = input.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        input = input.float()
        target = target.float()
        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape}), It may require one hot encoding")
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss_3d(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss_3d(input, target, self.gamma, self.alpha)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class DiceLoss3D(nn.Module):
    def __init__(
            self,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard = False,
            reduction='mean',
            smooth_nr=1e-5,
            smooth_dr=1e-5,
    ):
        super(DiceLoss3D, self).__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: # [1, 1, H, W, D] softmax后的概率图和Sigmoid图
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape}), It may require one hot encoding")
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target * input, dim=reduce_axis)
        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)
        f = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        if self.reduction == "mean":
            return torch.mean(f)
        elif self.reduction == "sum":
            return torch.sum(f)
        elif self.reduction == "none":
            return f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class DiceFocalLoss3D(nn.Module):
    def __init__(
            self,
            n_classes=4,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard=False,
            reduction="mean",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
    ):
        super(DiceFocalLoss3D, self).__init__()
        self.dice = DiceLoss3D(
            include_background=include_background,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction='mean',
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.focal = FocalLoss3D(
            include_background=include_background,
            gamma=gamma,
            use_softmax=softmax,
            reduction='mean',
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.reduction = reduction
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target):
        target = self._one_hot_encoder(target)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        if self.reduction == "mean":
            return torch.mean(total_loss)
        elif self.reduction == "sum":
            return torch.sum(total_loss)
        elif self.reduction == "none":
            return total_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


def compute_dice(pred, gt):
    # pred = pred[:, 1:, :, :, :]
    gt = gt[:, 1:, :, :, :]
    num_classes = pred.size(1)
    dice_scores = torch.zeros((pred.size(0), num_classes), device=pred.device)
    for i in range(num_classes):
        intersection = (pred[:, i] * gt[:, i]).sum(dim=[1, 2, 3])
        pred_sum = pred[:, i].sum(dim=[1, 2, 3])
        gt_sum = gt[:, i].sum(dim=[1, 2, 3])
        union = pred_sum + gt_sum
        dice_scores[:, i] = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice_scores


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


"""
ACKNOWLEDGEMENT AND CITATION:

This differentiable iso-surface extraction algorithm is adapted from the official 
implementation of "MeshSDF: Differentiable Iso-Surface Extraction".
- Repository: https://github.com/cvlab-epfl/MeshSDF
- Paper: Remelli et al., NeurIPS 2020
- Paper: Guillard et al., 2021 (arXiv:2106.11795 / IEEE TPAMI)

If you use this code in your research, please consider citing the original works:

@inproceedings{remelli2020meshsdf,
  title={MeshSDF: Differentiable Iso-Surface Extraction},
  author={Remelli, Edoardo and Lukoianov, Artem and Richter, Stephan R and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@article{guillard2021deepmesh,
  title={DeepMesh: Differentiable Iso-Surface Extraction},
  author={Guillard, Benoit and Remelli, Edoardo and Lukoianov, Artem and others},
  journal={arXiv preprint arXiv:2106.11795},
  year={2021}
}
"""
class DiffMarchingCubes(nn.Module):
    def __init__(self):
        super(DiffMarchingCubes, self).__init__()

    def forward(self, sdf):
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