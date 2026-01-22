from dataloader import LoadDataset
from torch.utils.data import DataLoader
from network import SwinUNETR, SDFNet, PialNet
import torch
import os
from utils import DiffMarchingCubes, denormalize, voxel2ras, compute_mesh_distance
from tqdm import tqdm
from nibabel.freesurfer import write_geometry
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import time


def compute_dice(pred, gt):
    pred = pred[:, 1:, :, :, :]
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


def Inference(config):
    # load data
    dataset = LoadDataset(data_path=config.data_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # load model
    Segmodel = SwinUNETR(img_size=(128, 256, 192), in_channels=2, out_channels=3, feature_size=48, use_checkpoint=True, use_v2=True).cuda()
    SDFmodel = SDFNet(in_channels=4).cuda()
    Pialmodel = PialNet(config.nc, config.K, config.n_scale).cuda()

    # load weights
    Segmodel.load_state_dict(torch.load(os.path.join(config.model_path, 'Segmentation.pth')))
    SDFmodel.load_state_dict(torch.load(os.path.join(config.model_path, 'LSDF.pth')))
    Pialmodel.load_state_dict(torch.load(os.path.join(config.model_path, 'LPialSurface.pth')))
    Pialmodel.initialize(128, 256, 192)

    Segmodel.eval()
    SDFmodel.eval()
    Pialmodel.eval()

    torch.backends.cudnn.benchmark = True

    times_voxel = []
    times_wm = []
    times_pial = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Inference")):
            image = data['image'].cuda()
            label = data['label'].cuda()
            affine = data['affine'][0].numpy()
            ID = data['ID'][0]

            # Star segmentation
            torch.cuda.synchronize()
            t_start_voxel = time.perf_counter()

            # If you have a pre-segmented tissue map (for other MRI contrast images or thicker clinical data), you can directly use it as input to the SDF model.
            # seg_prob = torch.zeros(1, 2, *label.shape[1:], dtype=torch.float32, device='cuda')
            # seg_prob[0, 0, :, :, :] = (label == 1).float()
            # seg_prob[0, 1, :, :, :] = (label == 2).float()

            seg_pred = Segmodel(image)
            seg_prob = torch.softmax(seg_pred, dim=1)  # (1, 3, 128, 256, 192)
            seg_prob = seg_prob[:, 1:, :, :, :]  # (1, 2, 128, 256, 192)

            torch.cuda.synchronize()
            t_end_voxel = time.perf_counter()
            time_voxel = t_end_voxel - t_start_voxel
            if i > 2: # Skip the first three iterations to exclude potentially inaccurate time measurements caused by system load, hardware initialization, or caching effects
                times_voxel.append(time_voxel)

            # Star WM surface reconstruction
            torch.cuda.synchronize()
            t_start_wm = time.perf_counter()

            input = torch.cat((image, seg_prob), dim=1)  # (1, 4, 128, 256, 192)
            SDF_pred = SDFmodel(input)

            inner_v_pred, inner_f_pred = DiffMarchingCubes()(SDF_pred[:, 0:1, ...])
            inner_v_pred_ras = denormalize(inner_v_pred, shape=SDF_pred[0, 0, ...].shape)
            inner_v_pred_ras = voxel2ras(inner_v_pred_ras, affine)

            torch.cuda.synchronize()
            t_end_wm = time.perf_counter()
            time_wm = t_end_wm - t_start_wm
            if i > 2:
                times_wm.append(time_wm)

            # Star pial surface reconstruction
            # load predicted pial SDF
            pial_SDF = torch.clamp(SDF_pred[:, 1:2, ...], min=-5, max=5)  # (1, 1, 128, 256, 192)

            # load predicted WM surface
            inner_v_pred_input = torch.Tensor(inner_v_pred).unsqueeze(0).cuda()
            inner_f_pred_input = torch.LongTensor(inner_f_pred).unsqueeze(0).cuda()

            torch.cuda.synchronize()
            t_start_pial = time.perf_counter()

            outer_v_pred = Pialmodel(v=inner_v_pred_input, f=inner_f_pred_input, volume=pial_SDF, n_smooth=config.n_smooth, lambd=config.lambd)
            outer_v_pred = denormalize(outer_v_pred.squeeze(0).cpu().numpy(), shape=(128, 256, 192))
            outer_v_pred = voxel2ras(outer_v_pred, affine)

            torch.cuda.synchronize()
            t_end_pial = time.perf_counter()
            time_pial = t_end_pial - t_start_pial
            if i > 2:
                times_pial.append(time_pial)

            # calculate metrics
            # calculate segmentation performance
            seg_out = torch.argmax(seg_pred, dim=1)  # (1, 128, 256, 192)
            seg = F.one_hot(seg_out, num_classes=3).permute(0, 4, 1, 2, 3)  # (1, 128, 256, 192)→(1, 128, 256, 192, 3)→(1, 3, 128, 256, 192)
            label = F.one_hot(label, num_classes=3).permute(0, 4, 1, 2, 3)  # (1, 128, 256, 192)→(1, 128, 256, 192, 3)→(1, 3, 128, 256, 192)
            dice = compute_dice(seg, label)  # (1, 2)

            # calculate WM surface performance
            inner_v_pred_ras = torch.Tensor(inner_v_pred_ras).unsqueeze(0).cuda()
            inner_f_pred_ras = torch.LongTensor(inner_f_pred).unsqueeze(0).cuda()

            # load WM surface GT
            inner_v_gt, inner_f_gt = nib.freesurfer.read_geometry(os.path.join(config.data_path, ID, "lh.white"))
            inner_v_gt = torch.Tensor(inner_v_gt).unsqueeze(0).cuda()
            inner_f_gt = torch.LongTensor(inner_f_gt.astype(np.int32)).unsqueeze(0).cuda()

            inner_mesh_pred = Meshes(verts=[inner_v_pred_ras.squeeze(0)], faces=[inner_f_pred_ras.squeeze(0)])
            inner_mesh_gt = Meshes(verts=[inner_v_gt.squeeze(0)], faces=[inner_f_gt.squeeze(0)])
            inner_points_pred = sample_points_from_meshes(inner_mesh_pred, num_samples=200000)
            inner_points_gt = sample_points_from_meshes(inner_mesh_gt, num_samples=200000)

            cd_wm = chamfer_distance(inner_points_pred, inner_points_gt)[0]
            assd_wm, hd90_wm = compute_mesh_distance(inner_v_pred_ras, inner_v_gt, inner_f_pred_ras, inner_f_gt, 200000)

            # calculate pial surface performance
            outer_v_pred = torch.Tensor(outer_v_pred).unsqueeze(0).cuda()

            # load pial surface GT
            outer_v_gt, outer_f_gt = nib.freesurfer.read_geometry(os.path.join(config.data_path, ID, "lh.pial"))
            outer_v_gt = torch.Tensor(outer_v_gt).unsqueeze(0).cuda()
            outer_f_gt = torch.LongTensor(outer_f_gt.astype(np.int32)).unsqueeze(0).cuda()

            outer_mesh_pred = Meshes(verts=[outer_v_pred.squeeze(0)], faces=[inner_f_pred_ras.squeeze(0)])
            outer_mesh_gt = Meshes(verts=[outer_v_gt.squeeze(0)], faces=[outer_f_gt.squeeze(0)])
            outer_points_pred = sample_points_from_meshes(outer_mesh_pred, num_samples=200000)
            outer_points_gt = sample_points_from_meshes(outer_mesh_gt, num_samples=200000)

            cd_pial = chamfer_distance(outer_points_pred, outer_points_gt)[0]
            assd_pial, hd90_pial = compute_mesh_distance(outer_v_pred, outer_v_gt, inner_f_pred_ras, outer_f_gt, 200000)

            # save results
            # save segmentation
            seg_out = seg_out.squeeze(0).cpu().numpy().astype(np.uint8)
            pad = [(0, 96), (0, 0), (0, 0)]
            seg_out = np.pad(seg_out, pad, mode='constant', constant_values=0)
            seg_out = nib.Nifti1Image(seg_out, affine)
            nib.save(seg_out, os.path.join(config.data_path, ID, f"pred_seg.nii.gz"))

            # save WM surface
            inner_v, inner_f = inner_v_pred_ras[0].cpu().numpy(), inner_f_pred_ras[0].cpu().numpy()
            write_geometry(os.path.join(config.data_path, ID, f"pred.white"), inner_v, inner_f)

            # save pial surface
            outer_v = outer_v_pred[0].cpu().numpy()
            write_geometry(os.path.join(config.data_path, ID, f"pred.pial"), outer_v, inner_f)

            print(f"ID {ID}: "
                  f"Voxel-level Time: {time_voxel:.4f}s, "
                  f"Surface-level (WM) Time: {time_wm:.4f}s, "
                  f"Surface-level (pial) Time: {time_pial:.4f}s, "
                  f"Total: {time_voxel + time_wm + time_pial:.4f}s, "
                  f"GM: {dice[0, 0].item():.4f}, WM: {dice[0, 1].item():.4f}; "
                  f"WM_CD: {cd_wm.item():.6f}, WM_ASSD: {assd_wm}, WM_HD90: {hd90_wm}"
                  f"Pial_CD: {cd_pial.item():.6f}, Pial_ASSD: {assd_pial}, Pial_HD90: {hd90_pial}")

    avg_voxel = np.mean(times_voxel)
    avg_wm = np.mean(times_wm)
    avg_pial = np.mean(times_pial)
    print("=" * 50)
    print(f"Average Inference Time (over {len(times_voxel)} subjects):")
    print(f"Voxel-level Processing      : {avg_voxel:.4f} s")
    print(f"WM Surface Reconstruction   : {avg_wm:.4f} s")
    print(f"Pial Surface Reconstruction : {avg_pial:.4f} s")
    print(f"Total Pipeline              : {avg_voxel + avg_wm + avg_pial:.4f} s")
    print("=" * 50)
    print('Inference Finished!')


if __name__ == '__main__':
    import argparse

    # args
    parser = argparse.ArgumentParser(description="UniSurf")
    # data
    parser.add_argument('--data_path', default=r"/your/data/path", type=str, help="path to data")
    parser.add_argument('--excel_path', default=r"/your/data/list/path.xlsx", type=str, help="path to data list")
    # model
    parser.add_argument('--nc', default=256, type=int, help="num of channels")
    parser.add_argument('--K', default=7, type=int, help="kernal size")
    parser.add_argument('--n_scale', default=5, type=int, help="num of scales for image pyramid")
    parser.add_argument('--n_smooth', default=1, type=int, help="num of Laplacian smoothing layers")
    parser.add_argument('--lambd', default=1.0, type=float, help="Laplacian smoothing weights")
    # inference
    parser.add_argument('--surf_hemi', default="left", type=str, help="left or right hemisphere")
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--model_path', default=r"/your/model/weights/path", type=str, help='path to model weights')

    config = parser.parse_args()

    Inference(config)