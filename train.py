from SDataLoader import LoadDataset, process_surface
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR
from network import SDFNet, PialNet
from utils import DiceFocalLoss3D, compute_dice, sdf2surf, DiffMarchingCubes, calc_SDF_grad, denormalize, voxel2ras
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from config import load_config
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency
import nibabel as nib
from seg2surf import topo_correct


# Using the pseudo-SDF GT for pretraining
def PretrainSDF(config):
    train_dataset = LoadDataset(data_path=config.train_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    Segmodel = SwinUNETR(img_size=(128, 256, 192), in_channels=2, out_channels=3, feature_size=48, use_checkpoint=True, use_v2=True).cuda()
    SDFmodel = SDFNet(in_channels=4).cuda()

    SegLoss = DiceFocalLoss3D(n_classes=3, softmax=True)
    SDFLoss = nn.MSELoss()

    optimizer = optim.Adam(list(Segmodel.parameters()) + list(SDFmodel.parameters()), lr=config.base_lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=1e-6)

    best_loss = float('inf')

    for epoch in range(config.n_epochs):
        Segmodel.train()
        SDFmodel.train()
        total_loss = 0
        for idx, data in enumerate(train_loader):
            image, label = data['image'].cuda(), data['label'].cuda()
            inner_SDF, outer_SDF = data['inner_SDF'].cuda(), data['outer_SDF'].cuda()

            optimizer.zero_grad()

            seg_pred = Segmodel(image)
            seg_prob = torch.softmax(seg_pred, dim=1)[:, 1:, :, :, :]
            input = torch.cat((image, seg_prob), dim=1)
            SDF_pred = SDFmodel(input)
            SDF = torch.cat((inner_SDF, outer_SDF), dim=1)

            seg_loss = SegLoss(seg_pred, label)
            SDF_loss = SDFLoss(SDF_pred, SDF)
            loss = seg_loss + SDF_loss

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(train_dataset)}]: "
                  f"Seg loss: {seg_loss.item():.4f}; SDF loss: {SDF_loss.item():.6f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{config.n_epochs}] current LR: {current_lr:.8f}")

        avg_train_loss = total_loss / len(train_loader)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
        torch.save(
            {
                'epoch': epoch,
                'Segmodel': Segmodel.state_dict(),
                'SDFmodel': SDFmodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'best_loss': best_loss
            }, os.path.join(config.output_dir, "pretrainedSDF.pth")
        )
        print(f"Saved model with best training loss: {best_loss:.6f}")


# Using the pseudo-SDF and surface GT for pretraining
def PretrainPial(config):
    train_dataset = LoadDataset(data_path=config.train_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    _, L, W, H = train_dataset[0]['outer_SDF'].shape

    model = PialNet(config.nc, config.K, config.n_scale).cuda()
    model.initialize(L, W, H)

    optimizer = optim.Adam(model.parameters(), lr=config.base_lr)

    best_loss = float('inf')

    for epoch in range(config.n_epochs):
        total_loss = 0
        for idx, data in enumerate(train_loader):
            outer_SDF = data['outer_SDF'].cuda()
            inner_v_gt, outer_v_gt, inner_f_gt, outer_f_gt = data['inner_v'].cuda(), data['outer_v'].cuda(), data['inner_f'].cuda(), data['outer_f'].cuda()

            optimizer.zero_grad()

            v_out = model(v=inner_v_gt, f=inner_f_gt, volume=outer_SDF, n_smooth=config.n_smooth, lambd=config.lambd)

            loss = nn.MSELoss()(v_out, outer_v_gt) * 1e3

            total_loss += loss.item()

            print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(train_dataset)}]: "
                  f"loss: {loss.item():.6f}")

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss
            }, os.path.join(config.output_dir, "pretrainedPial.pth")
        )
        print(f"Saved model with best training loss: {best_loss:.6f}")


def TrainSDF(config):
    train_dataset = LoadDataset(data_path=config.train_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi)
    valid_dataset = LoadDataset(data_path=config.valid_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi, mode='validiation')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    Segmodel = SwinUNETR(img_size=(128, 256, 192), in_channels=2, out_channels=3, feature_size=48, use_checkpoint=True, use_v2=True).cuda()
    SDFmodel = SDFNet(in_channels=4).cuda()

    Segmodel.load_state_dict(torch.load(os.path.join(config.output_dir, "pretrainedSDF.pth"))['Segmodel'])
    SDFmodel.load_state_dict(torch.load(os.path.join(config.output_dir, "pretrainedSDF.pth"))['SDFmodel'])

    # If your GPU memory is limited and joint training is not possible
    # You can uncomment this code to fine-tune only the model

    # for name, param in Segmodel.named_parameters():
    #     if name.startswith("swinViT") or name.startswith("encoder"):
    #         param.requires_grad = False

    # for param in Segmodel.parameters():
    #     param.requires_grad = False
    # Segmodel.eval()

    # for name, param in SDFmodel.named_parameters():
    #     if name.startswith("enc") or name.startswith("pool"):
    #         param.requires_grad = False

    SegLoss = DiceFocalLoss3D(n_classes=3, softmax=True)

    MC = DiffMarchingCubes()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(Segmodel.parameters()) + list(SDFmodel.parameters())),
                           lr=config.base_lr)

    total_steps = config.n_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    best_train_loss = float('inf')
    best_val_chamfer = float('inf')

    # ===== Checkpoint Resume Support =====
    start_epoch = 0
    # start_batch = 0
    if config.resume:
        print(f"Resuming from checkpoint")
        checkpoint = torch.load(os.path.join(config.output_dir, "TrainSDF.pth"))
        Segmodel.load_state_dict(checkpoint['Segmodel'])
        SDFmodel.load_state_dict(checkpoint["SDFmodel"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        # start_batch = checkpoint["batch"] + 1
        best_train_loss = checkpoint["loss"]

    for epoch in range(start_epoch, config.n_epochs):
        Segmodel.train()
        SDFmodel.train()
        total_loss = 0
        for idx, data in enumerate(train_loader):
            # if epoch == start_epoch and idx < start_batch:
            #     continue
            start_time = time.time()

            image, label = data['image'].cuda(), data['label'].cuda()
            inner_v_gt, outer_v_gt, inner_f_gt, outer_f_gt = data['inner_v'].cuda(), data['outer_v'].cuda(), data['inner_f'].cuda(), data['outer_f'].cuda()
            ID = data['ID'][0]

            optimizer.zero_grad()

            seg_pred = Segmodel(image) # (1, 2, 128, 256, 192) → (1, 3, 128, 256, 192)
            seg_prob = torch.softmax(seg_pred, dim=1) # (1, 3, 128, 256, 192) → (1, 3, 128, 256, 192)
            seg_prob = seg_prob[:, 1:, :, :, :] # (1, 3, 128, 256, 192) → (1, 2, 128, 256, 192)
            input = torch.cat((image, seg_prob), dim=1) # (1, 4, 128, 256, 192)
            SDF_pred = SDFmodel(input) # (1, 4, 128, 256, 192) → (1, 2, 128, 256, 192)

            seg_loss = SegLoss(seg_pred, label)

            inner_v_pred, inner_f_pred = MC(SDF_pred[:, 0:1, ...]) # (1, 1, 128, 256, 192)
            outer_v_pred, outer_f_pred = MC(SDF_pred[:, 1:2, ...]) # (1, 1, 128, 256, 192)

            inner_v_pred_0 = torch.tensor(inner_v_pred, dtype=torch.float32, requires_grad=True, device='cuda')
            inner_f_pred_0 = torch.tensor(inner_f_pred, dtype=torch.long, requires_grad=False, device='cuda')
            outer_v_pred_0 = torch.tensor(outer_v_pred, dtype=torch.float32, requires_grad=True, device='cuda')
            outer_f_pred_0 = torch.tensor(outer_f_pred, dtype=torch.long, requires_grad=False, device='cuda')

            inner_mesh_pred = Meshes(verts=[inner_v_pred_0], faces=[inner_f_pred_0])
            outer_mesh_pred = Meshes(verts=[outer_v_pred_0], faces=[outer_f_pred_0])
            inner_mesh_gt = Meshes(verts=[inner_v_gt.squeeze(0)], faces=[inner_f_gt.squeeze(0)])
            outer_mesh_gt = Meshes(verts=[outer_v_gt.squeeze(0)], faces=[outer_f_gt.squeeze(0)])

            inner_points_pred = sample_points_from_meshes(inner_mesh_pred, num_samples=200000)
            outer_points_pred = sample_points_from_meshes(outer_mesh_pred, num_samples=200000)
            inner_points_gt = sample_points_from_meshes(inner_mesh_gt, num_samples=200000)
            outer_points_gt = sample_points_from_meshes(outer_mesh_gt, num_samples=200000)

            inner_chamfer_loss = 1e3 * chamfer_distance(inner_points_pred, inner_points_gt)[0]
            outer_chamfer_loss = 1e3 * chamfer_distance(outer_points_pred, outer_points_gt)[0]
            surf_loss = inner_chamfer_loss + outer_chamfer_loss
            surf_loss.backward()

            inner_dL_dv = inner_v_pred_0.grad
            outer_dL_dv = outer_v_pred_0.grad

            optimizer.zero_grad()

            D1, D2, D3 = SDF_pred.shape[2:]
            D = max(D1, D2, D3)
            inner_v_pred_1 = ((inner_v_pred * D) + [D1, D2, D3]) / 2.0
            outer_v_pred_1 = ((outer_v_pred * D) + [D1, D2, D3]) / 2.0
            inner_v_pred_1 = inner_v_pred_1 / np.array([D1 - 1, D2 - 1, D3 - 1]) * 2.0 - 1.0
            outer_v_pred_1 = outer_v_pred_1 / np.array([D1 - 1, D2 - 1, D3 - 1]) * 2.0 - 1.0

            inner_v_pred_1 = torch.tensor(inner_v_pred_1, dtype=torch.float32, requires_grad=True, device='cuda')
            outer_v_pred_1 = torch.tensor(outer_v_pred_1, dtype=torch.float32, requires_grad=True, device='cuda')

            inner_v_SDF = F.grid_sample(SDF_pred[:, 0:1, ...].permute(0, 1, 4, 3, 2), inner_v_pred_1.unsqueeze(0).unsqueeze(-2).unsqueeze(-2), mode='bilinear', padding_mode='border', align_corners=True)
            outer_v_SDF = F.grid_sample(SDF_pred[:, 1:2, ...].permute(0, 1, 4, 3, 2), outer_v_pred_1.unsqueeze(0).unsqueeze(-2).unsqueeze(-2), mode='bilinear', padding_mode='border', align_corners=True)

            inner_normals_loss = torch.sum(inner_v_SDF)
            outer_normals_loss = torch.sum(outer_v_SDF)

            normals_loss = inner_normals_loss + outer_normals_loss
            normals_loss.backward(retain_graph=True)

            inner_normals = inner_v_pred_1.grad / torch.norm(inner_v_pred_1.grad, dim=-1, keepdim=True).clamp(min=1e-6)
            outer_normals = outer_v_pred_1.grad / torch.norm(outer_v_pred_1.grad, dim=-1, keepdim=True).clamp(min=1e-6)

            optimizer.zero_grad()

            inner_v_pred_2 = torch.tensor(inner_v_pred, dtype=torch.float32, requires_grad=True, device='cuda')
            outer_v_pred_2 = torch.tensor(outer_v_pred, dtype=torch.float32, requires_grad=True, device='cuda')

            inner_SDF_grad = calc_SDF_grad(SDF_pred[:, 0:1, ...], inner_v_pred_2, inner_dL_dv, inner_normals)
            outer_SDF_grad = calc_SDF_grad(SDF_pred[:, 1:2, ...], outer_v_pred_2, outer_dL_dv, outer_normals)

            inner_dummy_loss = torch.sum(SDF_pred[:, 0:1, ...] * inner_SDF_grad)
            outer_dummy_loss = torch.sum(SDF_pred[:, 1:2, ...] * outer_SDF_grad)
            dummy_loss = inner_dummy_loss + outer_dummy_loss + seg_loss
            dummy_loss.backward(retain_graph=True)

            optimizer.step()
            scheduler.step()

            total_loss += surf_loss.item()

            print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(train_dataset)}], time: {time.time() - start_time:.2f}s, ID {ID}: "
                  f"inner chamfer loss: {inner_chamfer_loss.item():.6f}, outer chamfer loss: {outer_chamfer_loss.item():.6f}; "
                  f"SDFLR: {optimizer.param_groups[0]['lr']:.2e}")

            if (idx + 1) % 1000 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': idx,
                    'Segmodel': SegModel.state_dict(),
                    'SDFmodel': SDFmodel.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(config.output_dir, f"epoch{epoch + 1}_batch{idx + 1}.pth"))
                print(f"Checkpoint saved at batch {idx + 1} of epoch {epoch + 1}.")

        avg_train_loss = total_loss / len(train_loader)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'Segmodel': SegModel.state_dict(),
                'SDFmodel': SDFmodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_train_loss
            }, os.path.join(config.output_dir, "TrainSDF.pth"))
            print(f"Saved model with best training loss: {best_train_loss:.6f}")

        if (epoch + 1) % config.valid_interval == 0:
            SDFmodel.eval()
            chamfer = []
            with torch.no_grad():
                for idx, data in enumerate(valid_loader):
                    start_time = time.time()

                    image = data['image'].cuda()
                    inner_v_gt, outer_v_gt, inner_f_gt, outer_f_gt = data['inner_v'].cuda(), data['outer_v'].cuda(), data['inner_f'].cuda(), data['outer_f'].cuda()
                    ID = data['ID'][0]

                    seg_pred = Segmodel(image)
                    seg_prob = torch.softmax(seg_pred, dim=1)
                    seg_prob = seg_prob[:, 1:, :, :, :]
                    input = torch.cat((image, seg_prob), dim=1)
                    SDF_pred = SDFmodel(input)

                    inner_SDF_pred = SDF_pred[0, 0, ...].detach().cpu().numpy().astype(float)
                    outer_SDF_pred = SDF_pred[0, 1, ...].detach().cpu().numpy().astype(float)
                    inner_v_pred, inner_f_pred = sdf2surf(inner_SDF_pred)
                    outer_v_pred, outer_f_pred = sdf2surf(outer_SDF_pred)
                    inner_v_pred = torch.tensor(inner_v_pred, dtype=torch.float32, requires_grad=True, device='cuda')
                    inner_f_pred = torch.tensor(inner_f_pred, dtype=torch.long, requires_grad=False, device='cuda')
                    outer_v_pred = torch.tensor(outer_v_pred, dtype=torch.float32, requires_grad=False, device='cuda')
                    outer_f_pred = torch.tensor(outer_f_pred, dtype=torch.long, requires_grad=False, device='cuda')
                    inner_mesh_pred = Meshes(verts=[inner_v_pred], faces=[inner_f_pred])
                    outer_mesh_pred = Meshes(verts=[outer_v_pred], faces=[outer_f_pred])
                    inner_mesh_gt = Meshes(verts=[inner_v_gt.squeeze(0)], faces=[inner_f_gt.squeeze(0)])
                    outer_mesh_gt = Meshes(verts=[outer_v_gt.squeeze(0)], faces=[outer_f_gt.squeeze(0)])

                    inner_points_pred = sample_points_from_meshes(inner_mesh_pred, num_samples=200000)
                    outer_points_pred = sample_points_from_meshes(outer_mesh_pred, num_samples=200000)
                    inner_points_gt = sample_points_from_meshes(inner_mesh_gt, num_samples=200000)
                    outer_points_gt = sample_points_from_meshes(outer_mesh_gt, num_samples=200000)

                    inner_chamfer_loss = 1e3 * chamfer_distance(inner_points_pred, inner_points_gt)[0]
                    outer_chamfer_loss = 1e3 * chamfer_distance(outer_points_pred, outer_points_gt)[0]
                    chamfer_loss = inner_chamfer_loss + outer_chamfer_loss

                    chamfer.append(chamfer_loss.item())

                    print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(valid_dataset)}], time: {time.time() - start_time:.2f}s, ID {ID}: "
                          f"inner chamfer distance: {inner_chamfer_loss:.6f}, outer chamfer distance: {outer_chamfer_loss:.6f}")

            avg_val_chamfer = np.mean(chamfer)
            if avg_val_chamfer < best_val_chamfer:
                best_val_chamfer = avg_val_chamfer
                torch.save({
                    'epoch': epoch,
                    'SDFmodel': SDFmodel.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'chamfer distance': best_val_chamfer
                }, os.path.join(config.output_dir, "ValidSDF.pth"))
                print(f"Saved model with best validation Chamfer Distance: {best_val_chamfer:.4f}")


def TrainPial(config):
    train_dataset = LoadDataset(data_path=config.data_path, excel_path=config.excel_path, surf_hemi=config.surf_hemi)
    valid_dataset = LoadDataset(data_path=config.valid_path,  excel_path=config.excel_path, surf_hemi=config.surf_hemi, mode='validiation')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    SDF_path = '/path/for/your/predicted/pial/surface/SDF/from/SDFs/prediction/network'
    WM_path = '/path/for/your/predicted/WM/surface/SDF/from/SDFs/prediction/network'

    model = PialNet(config.nc, config.K, config.n_scale).cuda()
    model.initialize(128, 256, 192)

    model.load_state_dict(torch.load(os.path.join(config.output_dir, "pretrainedPial.pth"))['model'])

    optimizer = optim.Adam(model.parameters(), lr=config.base_lr)

    best_train_loss = float('inf')
    best_val_cd = float('inf')

    start_epoch = 0
    checkpoint_path = os.path.join(config.output_dir, "TrainPial.pth")
    if config.resume:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_train_loss = checkpoint['best_train_loss']
        print(f"Resumed from checkpoint at epoch {start_epoch}, best train loss {best_train_loss:.6f}")

    for epoch in range(start_epoch, config.n_epochs):
        model.train()
        total_loss = 0
        for idx, data in enumerate(train_loader):
            outer_v_gt, outer_f_gt = data['outer_v'].cuda(), data['outer_f'].cuda()
            ID = data['ID'][0]

            outer_SDF = nib.load(os.path.join(SDF_path, ID + '.nii.gz')).get_fdata()
            outer_SDF = outer_SDF[:128, :, :]
            outer_SDF = topo_correct.apply(outer_SDF, threshold=5)
            outer_SDF = np.clip(outer_SDF, a_min=-5, a_max=5)
            outer_SDF = torch.tensor(outer_SDF, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
            inner_v, inner_f = nib.freesurfer.read_geometry(os.path.join(WM_path, ID + ".white"))
            inner_f = inner_f.astype(np.int32)
            inner_v = torch.Tensor(inner_v).unsqueeze(0).float().cuda()
            inner_f = torch.LongTensor(inner_f).unsqueeze(0).cuda()

            optimizer.zero_grad()

            v_out = model(v=inner_v, f=inner_f, volume=outer_SDF, n_smooth=config.n_smooth, lambd=config.lambd)

            outer_mesh_pred = Meshes(verts=[v_out.squeeze(0)], faces=[inner_f.squeeze(0)])
            outer_mesh_gt = Meshes(verts=[outer_v_gt.squeeze(0)], faces=[outer_f_gt.squeeze(0)])
            outer_points_pred = sample_points_from_meshes(outer_mesh_pred, num_samples=200000)
            outer_points_gt = sample_points_from_meshes(outer_mesh_gt, num_samples=200000)

            distance_loss = 1e3 * chamfer_distance(outer_points_pred, outer_points_gt)[0]
            edge_loss = 0.5 * 1e3 * mesh_edge_loss(outer_mesh_pred)
            normal_loss = 0.001 * 1e3 * mesh_normal_consistency(outer_mesh_pred)

            loss = distance_loss + edge_loss + normal_loss
            total_loss += loss.item()

            print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(train_dataset)}], "
                  # f"ID: {ID}: "
                  f"loss: {loss.item():.6f}, "
                  f"cd loss: {distance_loss.item():.6f}, edge loss: {edge_loss.item():.6f}, normal loss: {normal_loss.item():.6f} ")

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_train_loss': best_train_loss
            }, os.path.join(config.output_dir, "TrainPial.pth"))
            print(f"Saved model with best training loss: {best_train_loss:.6f}")

        if (epoch + 1) % config.valid_interval == 0:
            model.eval()

            cd = 0

            with torch.no_grad():
                for idx, data in enumerate(valid_loader):
                    affine = data['affine'][0].numpy()
                    ID = data['ID'][0]

                    outer_v_gt, outer_f_gt = nib.freesurfer.read_geometry(os.path.join(config.data_path, ID, "lh.pial"))
                    outer_v_gt = torch.Tensor(outer_v_gt).cuda()
                    outer_f_gt = torch.LongTensor(outer_f_gt.astype(np.int32)).cuda()

                    outer_SDF = nib.load(os.path.join(SDF_path, ID + '.nii.gz')).get_fdata()
                    outer_SDF = outer_SDF[:128, :, :]
                    outer_SDF = topo_correct.apply(outer_SDF, threshold=5)
                    outer_SDF = np.clip(outer_SDF, a_min=-5, a_max=5)
                    outer_SDF = torch.tensor(outer_SDF, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
                    inner_v, inner_f = nib.freesurfer.read_geometry(os.path.join(WM_path, ID + ".white"))
                    inner_f = inner_f.astype(np.int32)
                    inner_v = torch.Tensor(inner_v).unsqueeze(0).float().cuda()
                    inner_f = torch.LongTensor(inner_f).unsqueeze(0).cuda()

                    v_out = model(v=inner_v, f=inner_f, volume=outer_SDF, n_smooth=config.n_smooth, lambd=config.lambd)

                    v_out = denormalize(v_out.squeeze(0).cpu().numpy(), shape=(128, 256, 192))
                    v_out = voxel2ras(v_out, affine)
                    v_out = torch.Tensor(v_out).cuda()

                    outer_mesh_pred = Meshes(verts=[v_out], faces=[inner_f.squeeze(0)])
                    outer_mesh_gt = Meshes(verts=[outer_v_gt], faces=[outer_f_gt])

                    outer_points_pred = sample_points_from_meshes(outer_mesh_pred, num_samples=200000)
                    outer_points_gt = sample_points_from_meshes(outer_mesh_gt, num_samples=200000)

                    chamfer = chamfer_distance(outer_points_pred, outer_points_gt)[0]
                    cd += chamfer.item()

                    print(f"epoch: [{epoch + 1}/{config.n_epochs}], batch: [{idx + 1}/{len(valid_dataset)}], ID {ID}: "
                          f"chamfer distance: {chamfer.item():.6f}")

            avg_val_cd = cd / len(valid_loader)

            if avg_val_cd < best_val_cd:
                best_val_cd = avg_val_cd
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_cd': best_val_cd
                }
                torch.save(checkpoint, os.path.join(config.output_dir, "ValidPial.pth"))
                print(f"Saved the best validation model at epoch {epoch + 1} with val loss {best_val_cd:.6f}")


if __name__ == '__main__':
    config = load_config()
    os.makedirs(config.output_dir, exist_ok=True)
    PretrainSDF(config)
    PretrainPial(config)
    # TrainSDF(config)
    # TrainPial(config)