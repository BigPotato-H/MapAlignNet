import theseus as th
import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from theseus.core.cost_function import ErrFnType
from typing import cast
import random
import math
import time
import cv2
import os

pooling_size = 5
img_pooling = torch.nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=(pooling_size - 1) // 2)

def get_class_weights():
    class_weights = np.array([1, 50])
    class_weights = torch.from_numpy(class_weights.astype(np.float32))
    class_weights = class_weights.cuda()
    return class_weights


def cull_to_view_frustum(uv,H, W, points_cam):
    is_valid_x = torch.logical_and(0 <= uv[:, 0], uv[:, 0] < W - 1)
    is_valid_y = torch.logical_and(0 <= uv[:, 1], uv[:, 1] < H - 1)
    #is_valid_z = points_cam[:, 2] > 0
    is_valid_z = torch.logical_and(0 < points_cam[:, 2], points_cam[:, 2] < 60)
    is_valid_points = torch.logical_and(torch.logical_and(is_valid_x, is_valid_y), is_valid_z) #bool
    return is_valid_points


def transfoorm_hd_to_image(hd, img, K, P):
    N = hd.shape[2]
    B, H, W = img.shape[0], img.shape[1], img.shape[2]

    hd_homo = torch.cat((hd,
                         torch.ones((B, 1, N), dtype=torch.float32, device=hd.device)),
                        dim=1)  # Bx4xN
    P = P.reshape(B,3,4)   # B*3*4
    P_hd_homo = torch.bmm(P, hd_homo)  # Bx3xN
    KP_hd_homo = torch.bmm(K, P_hd_homo)  # Bx3xN

    KP_hd_pxpy = KP_hd_homo[:, 0:2, :] / KP_hd_homo[:, 2:3, :]  # Bx2xN
    #KP_hd_pxpy = KP_hd_pxpy.to(dtype=torch.long)
    #return KP_hd_pxpy

    '''is_valid_points = cull_to_view_frustum(KP_hd_pxpy, H, W, KP_hd_homo)
    is_valid_points = is_valid_points.bool().unsqueeze(1)
    is_valid_points = is_valid_points.repeat(1,2,1)

    mask = torch.zeros_like(KP_hd_pxpy)
    KP_hd_pxpy = torch.where(is_valid_points, KP_hd_pxpy, mask)'''

    # Example of a soft clipping using a sigmoid function
    soft_mask_x = torch.sigmoid(10 * (KP_hd_pxpy[:, 0] - 0)) * torch.sigmoid(10 * (W - 1 - KP_hd_pxpy[:, 0]))
    soft_mask_y = torch.sigmoid(10 * (KP_hd_pxpy[:, 1] - 0)) * torch.sigmoid(10 * (H - 1 - KP_hd_pxpy[:, 1]))
    soft_mask_z = torch.sigmoid(10 * (KP_hd_homo[:, 2] - 0)) * torch.sigmoid(10 * (60 - KP_hd_homo[:, 2]))
    soft_mask = soft_mask_x * soft_mask_y * soft_mask_z
    KP_hd_pxpy = KP_hd_pxpy * soft_mask.unsqueeze(1)

    return KP_hd_pxpy


from scipy.ndimage import distance_transform_edt
def normalize_array(array):
    max_val = np.max(array)
    min_val = np.min(array)
    if max_val - min_val == 0:  # To handle cases where all values are the same
        return np.zeros_like(array)
    else:
        return (array - min_val) / (max_val - min_val)


def project_error(hd, img, K, P):
    B, H, W = img.shape[0], img.shape[1], img.shape[2]
    # Project HD map points to image plane
    KP_hd_pxpy = transfoorm_hd_to_image(hd, img, K, P)
    img = img.unsqueeze(1)  # Adjust to [1, 1, 256, 256] if img has a single channel

    KP_hd_pxpy_normalized = KP_hd_pxpy.clone().float()
    KP_hd_pxpy_normalized[:, 0, :] = 2 * (KP_hd_pxpy[:, 0, :] / (W - 1)) - 1  # Normalizing width
    KP_hd_pxpy_normalized[:, 1, :] = 2 * (KP_hd_pxpy[:, 1, :] / (H - 1)) - 1  # Normalizing height

    grid = KP_hd_pxpy_normalized.permute(0, 2, 1).unsqueeze(2)  # [1, 1024, 1, 2]
    # Use grid_sample
    img_values = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    img_values = img_values.squeeze(1).squeeze(-1)  # To remove unnecessary dimensions

    # Create a tensor of ones for hd_values (assuming you want to compare with image values)
    hd_values = torch.ones(B, hd.shape[2]).cuda()  # Shape: (B, N)
    #err = torch.abs(hd_values - img_values)
    #loss = err.mean(dim=1).unsqueeze(1)  # Shape: (B, 1)

    loss = nn.functional.mse_loss(img_values,hd_values, reduction="none")
    loss = loss.mean(dim=1, keepdim=True)#.unsqueeze(1)
    return loss

def pixel_error(hd, img, K, P):
    B, H, W = img.shape[0], img.shape[1], img.shape[2]
    # Project HD map points to image plane
    KP_hd_pxpy = transfoorm_hd_to_image(hd, img, K, P)
    img = img.unsqueeze(1)  # Adjust to [1, 1, 256, 256] if img has a single channel

    KP_hd_pxpy_normalized = KP_hd_pxpy.clone().float()
    KP_hd_pxpy_normalized[:, 0, :] = 2 * (KP_hd_pxpy[:, 0, :] / (W - 1)) - 1  # Normalizing width
    KP_hd_pxpy_normalized[:, 1, :] = 2 * (KP_hd_pxpy[:, 1, :] / (H - 1)) - 1  # Normalizing height

    grid = KP_hd_pxpy_normalized.permute(0, 2, 1).unsqueeze(2)  # [1, 1024, 1, 2]
    # Use grid_sample
    img_values = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    img_values = img_values.squeeze(1).squeeze(-1)  # To remove unnecessary dimensions

    # Create a tensor of ones for hd_values (assuming you want to compare with image values)
    hd_values = torch.ones(B, hd.shape[2]).cuda()  # Shape: (B, N)
    #err = torch.abs(hd_values - img_values)
    #loss = err.mean(dim=1).unsqueeze(1)  # Shape: (B, 1)

    loss = nn.functional.mse_loss(img_values,hd_values, reduction="none")
    loss = loss.mean(dim=1, keepdim=True)#.unsqueeze(1)

    return loss

def dist_error(hd, dist_img, K, P):
    B, H, W = dist_img.shape[0], dist_img.shape[1], dist_img.shape[2]
    # Project HD map points to image plane
    KP_hd_pxpy = transfoorm_hd_to_image(hd, dist_img, K, P)
    dist_img = dist_img.unsqueeze(1)  # Adjust to [1, 1, 256, 256] if img has a single channel

    '''reprojection_error_mean = []
    for b in range(B):
        proj_x = KP_hd_pxpy[b, 0, :].round().long().clamp(0, W - 1)
        proj_y = KP_hd_pxpy[b, 1, :].round().long().clamp(0, H - 1)
        reprojection_error = 1 - dist_img[b, 0, proj_y, proj_x]
        reprojection_error_mean.append(reprojection_error.mean().item())
    reproject_error = torch.tensor(reprojection_error_mean).cuda().unsqueeze(1)'''

    KP_hd_pxpy_normalized = KP_hd_pxpy.clone().float()
    KP_hd_pxpy_normalized[:, 0, :] = 2 * (KP_hd_pxpy[:, 0, :] / (W - 1)) - 1  # Normalizing width
    KP_hd_pxpy_normalized[:, 1, :] = 2 * (KP_hd_pxpy[:, 1, :] / (H - 1)) - 1  # Normalizing height

    grid = KP_hd_pxpy_normalized.permute(0, 2, 1).unsqueeze(2)  # [1, 1024, 1, 2]
    # Use grid_sample
    img_values = torch.nn.functional.grid_sample(dist_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    img_values = img_values.squeeze(1).squeeze(-1)  # To remove unnecessary dimensions

    # Create a tensor of ones for hd_values (assuming you want to compare with image values)
    hd_values = torch.ones(B, hd.shape[2]).cuda()  # Shape: (B, N)
    loss = nn.functional.mse_loss(img_values, hd_values, reduction="none")
    loss = loss.mean(dim=1, keepdim=True)
    return loss


def cauchy_fn(x):
    return torch.sqrt(6* torch.log(1 + x ** 2))

def cauchy_pixel_error_fn(optim_vars, aux_vars):
    P = optim_vars[0].tensor
    # th.Variable
    hd = aux_vars[0].tensor
    img = aux_vars[1].tensor
    K = aux_vars[2].tensor
    err = pixel_error(hd, img, K, P)
    #err = cauchy_fn(err)
    return err

def cauchy_dist_error_fn(optim_vars, aux_vars):
    P = optim_vars[0].tensor
    # th.Variable
    hd = aux_vars[0].tensor
    img = aux_vars[1].tensor
    K = aux_vars[2].tensor
    err = dist_error(hd, img, K, P)
    #err = cauchy_fn(err)
    return err

def cal_rotate_matrix(angles):
    axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    r_x = angles[0]
    r_y = angles[1]
    r_z = angles[2]
    rot_x = linalg.expm(np.cross(np.eye(3), axis_x / linalg.norm(axis_x) * r_x))
    rot_y = linalg.expm(np.cross(np.eye(3), axis_y / linalg.norm(axis_y) * r_y))
    rot_z = linalg.expm(np.cross(np.eye(3), axis_z / linalg.norm(axis_z) * r_z))

    rot = np.dot(np.dot(rot_x, rot_y), rot_z)
    return rot


def generate_random_transform(P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
    """

    :param pc_np: pc in NWU coordinate
    :return:
    """
    t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
         random.uniform(-P_ty_amplitude, P_ty_amplitude),
         random.uniform(-P_tz_amplitude, P_tz_amplitude)]

    angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
              random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
              random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

    #rotation_mat = augmentation.angles2rotation_matrix(angles)
    rotation_mat = cal_rotate_matrix(angles)
    P_random = np.identity(4, dtype=float)
    P_random[0:3, 0:3] = rotation_mat
    P_random[0:3, 3] = t

    return P_random

def jit_gt_pose(P, batch_size):
    if P is not None:
        batch_size, rows, cols = P.shape
        tf_jitter = generate_random_transform(0.5, 0.5,0.5,
                                       1.0 * math.pi / 90.0,
                                       1.0 * math.pi / 90.0,
                                       1.0 * math.pi / 90.0)
        tf_jitter = torch.from_numpy(tf_jitter.astype(np.float32)).repeat(batch_size,1, 1).cuda()
        zero_row = torch.zeros(1, cols, dtype=P.dtype).repeat(batch_size,1,1).cuda()
        P_with_zero_row = torch.cat((P, zero_row), dim=1)
        P_jitter = torch.matmul(tf_jitter, P_with_zero_row)
        P_jitter = P_jitter[:,0:3,:]
    else:
        P_init = torch.eye(3)
        P_init = torch.cat((P_init,
                   torch.zeros((3, 1), dtype=torch.float32)),
                  dim=1)
        P_jitter = P_init.repeat(batch_size, 1,1).cuda()
    return P_jitter


def diff_to_gt(gt_Rt, P):
    #gt_Rt = np.array(gt_Rt.cpu())
    #P = np.array(P.cpu())

    B = gt_Rt.shape[0]

    new_row = torch.tensor([0, 0, 0, 1], device=gt_Rt.device).view(1, 1, 4).repeat(B, 1, 1)
    #new_row = torch.zeros((B, 1, 4), device=gt_Rt.device)  # Shape: [B, 1, 4]

    # Concatenate the original tensor with the new row along the 1st dimension
    gt = torch.cat((gt_Rt, new_row), dim=1)  # Shape: [B, 4, 4]
    p = torch.cat((P.view(-1, 3, 4), new_row), dim=1)  # Shape: [B, 4, 4]

    # Compute the pseudo-inverse of gt for the entire batch
    gt_inv = torch.linalg.pinv(gt)  # Shape: [B, 4, 4]

    # Perform matrix multiplication for the entire batch
    dif = torch.matmul(gt_inv, p)  # Shape: [B, 4, 4]

    if 0:
        for b in range(B):
            dif_b = dif[b, :, :]
            rotation_matrix = dif_b[0:3, 0:3]  # Extract the rotation matrix
            t = dif_b[0:3, 3]  # Extract the translation vector
            r = R.from_matrix(rotation_matrix.cpu().numpy())  # Convert to numpy for SciPy
            euler_angles = r.as_euler('xyz', degrees=True)
            print('t:', t)
            print('euler', euler_angles)

    return dif

def view_hd_img(hd, img,P, K, clr):
    KP_hd_pxpy = transfoorm_hd_to_image(hd, img, K, P)
    KP_hd_pxpy = KP_hd_pxpy.detach().cpu().numpy()

    B, H, W = img.shape[0], img.shape[1], img.shape[2]
    N = KP_hd_pxpy.shape[1]

    b = 0
    seg_img = img[b]
    #img_label = torch.argmax(seg_img.squeeze(), dim=0).detach().cpu().numpy()
    img_label = seg_img.detach().cpu().numpy()
    view_img = np.zeros((3, H, W))
    view_img[0, :, :] = 255  # Red channel
    view_img[1, :, :] = 192  # Green channel
    view_img[2, :, :] = 203  # Blue channel

    view_img[:, (img_label == 0)] = 150
    view_img[:, (img_label > 0)] = 255

    y_indices = KP_hd_pxpy[b,1, :].astype(int)
    x_indices = KP_hd_pxpy[b,0, :].astype(int)
    view_img[0, y_indices, x_indices] = clr[0]
    view_img[1, y_indices, x_indices] = clr[1]
    view_img[2, y_indices, x_indices] = clr[2]

    view_img = view_img.transpose(1,2,0)

    return view_img

def get_view_hd_img(hd, img,P, K):
    KP_hd_pxpy = transfoorm_hd_to_image(hd, img, K, P)
    B, H, W = img.shape[0], img.shape[1], img.shape[2]
    N = KP_hd_pxpy.shape[1]

    #batch_proj_img = torch.zeros_like(img)
    batch_proj_img = torch.zeros(B, H, W, dtype=torch.float32).cuda()
    for bi in range(B):
        proj_xrev = KP_hd_pxpy[bi, 0, :].round().long()
        proj_yrev = KP_hd_pxpy[bi, 1, :].round().long()
        batch_proj_img[bi * torch.ones_like(proj_xrev), proj_yrev, proj_xrev] = 1
    batch_proj_img = img_pooling(batch_proj_img)

    return batch_proj_img


def run(hd, img, K, init_Rt, gt_Rt=None):
    #torch.autograd.set_detect_anomaly(True)

    batch_size = hd.shape[0]

    img = torch.argmax(img, dim=1).float()
    dist_img = torch.zeros_like(img).float().cuda()
    for b in range(0,batch_size):
        image_np = img[b].cpu().numpy()
        #image_np = img[b].cpu().numpy() *255 # Convert PyTorch tensor to NumPy array
        #cv2.imwrite('/mnt/e/data/hn/temp/pose_solve/seg.jpg', image_np)
        dist_np = distance_transform_edt(image_np == 1)
        #dist_np = distance_transform_edt(image_np == 0)*255
        #cv2.imwrite('/mnt/e/data/hn/temp/pose_solve/dt.jpg', dist_np)
        dist_np = normalize_array(dist_np)
        dist_img[b] = torch.from_numpy(dist_np).type(torch.FloatTensor).cuda()

    objective = th.Objective()
    objective.device = torch.device("cuda:0")

    # data is of type Variable
    hd_var = th.Variable(tensor=hd, name="hd")
    img_var = th.Variable(tensor=img, name="img")
    dist_img_var = th.Variable(tensor=dist_img, name="dist_img")
    K_var = th.Variable(tensor=K, name="K")

    #P_init_mat = jit_gt_pose(gt_Rt, batch_size)
    P_init_mat = init_Rt
    P_init = P_init_mat.reshape(batch_size, -1)#.requires_grad_(True)
    P_init_var = th.Vector(tensor=P_init, name='P')

    if 1:
        wt_cost_function = th.AutoDiffCostFunction(
            optim_vars = [P_init_var],
            err_fn=cast(ErrFnType, cauchy_pixel_error_fn),
            dim=1,
            aux_vars = [hd_var, img_var, K_var],
            name="cauchy_quad_cost_fn",
            cost_weight=th.ScaleCostWeight(torch.ones(1).cuda()),
            #autograd_mode="vmap"
            autograd_mode="DENSE"
        )
        objective.add(wt_cost_function)

    if 0:
        dist_cost_function = th.AutoDiffCostFunction(
            optim_vars=[P_init_var],
            err_fn=cast(ErrFnType, cauchy_dist_error_fn),
            dim=1,
            aux_vars=[hd_var, dist_img_var, K_var],
            name="cauchy_dist_cost_fn",
            cost_weight=th.ScaleCostWeight(torch.ones(1).cuda()),
            # autograd_mode="vmap"
            autograd_mode="DENSE"
            # autograd_mode = "LOOP_BATCH"
        )
        objective.add(dist_cost_function)

    #if 0:
    if gt_Rt is not None:
        diff_weight = th.ScaleCostWeight(torch.ones(1).cuda())
        gt_var = th.Vector(tensor= gt_Rt.reshape(batch_size, -1), name='P_gt')
        objective.add(
            th.Difference(
                P_init_var,
                gt_var,
                diff_weight,
                name="camera_diff",
            ))

    step_size = 0.5
    if gt_Rt is None:
        step_size = 0.01

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=40,
        step_size=step_size,
    )
    theseus_optim = th.TheseusLayer(optimizer).cuda()
    theseus_inputs = {
    "P": P_init}

    import warnings
    warnings.simplefilter("ignore")
    start_time = time.time_ns()

    verbose = False
    if gt_Rt is None:
        verbose = True
    _, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":verbose})

    P = info.best_solution['P'].cuda()

    P = init_Rt
    P_init_mat = init_Rt
    if 0:
        pose_solve_dir = '/mnt/e/data/hn/temp/pose_solve/'
        img_opz = view_hd_img(hd, img, P, K, [255,0,0])
        img_gt = view_hd_img(hd, img, gt_Rt, K, [0,255,0])
        img_init = view_hd_img(hd, img, P_init_mat, K,[0,0,255])
        img_mosaic = np.hstack((img_gt,img_init,img_opz))
        cv2.imwrite(os.path.join(pose_solve_dir, 'pose_solve.jpg'), img_mosaic)

        print('///')
        print(P.view(-1,3,4))
        print(P_init_mat)


    reprojection_loss = project_error(hd, dist_img, K, P).mean()
    opz_proj_img = get_view_hd_img(hd, img, P, K)
    return  reprojection_loss, P, opz_proj_img
