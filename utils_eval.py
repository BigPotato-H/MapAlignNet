import torch
from scipy.spatial.transform import Rotation as R
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import LinearSegmentedColormap


def calculate_pix_acc(ground_truth, predictions, num_classes=1):
    """
    Calculates pixel accuracy, given target and output tensors
    and number of classes.
    """
    if 1:
        labeled = (ground_truth > 0) * (ground_truth <= num_classes)
        #_, preds = torch.max(outputs.data, 1)
        correct = (predictions == ground_truth) * labeled
        labeled = labeled.sum()
    else:
        labeled, correct = per_class_accuracy(target, outputs, num_classes)

    if labeled == 0:
        pixacc = 1.0
    else:
        pixacc = correct.sum().item() / labeled
    correct = correct.detach().cpu().numpy()
    errors = np.sum(correct, axis=0)
    #errors = (1 - correct).sum(dim=0)
    return pixacc, errors


def calculate_iou( ground_truth, predictions, num_classes=1):
    iou_per_class = []

    with torch.no_grad():
        for cls in range(1, num_classes + 1):  # Start from class 1 (exclude background, which is assumed as class 0)
            # Create binary masks for the current class in both ground truth and prediction
            pred_mask = (predictions == cls)
            gt_mask = (ground_truth == cls)

            # Intersection: Pixels where both prediction and ground truth are the same for the current class
            intersection = torch.logical_and(pred_mask, gt_mask).sum().float()

            # Union: Pixels that are either in prediction or ground truth for the current class
            union = torch.logical_or(pred_mask, gt_mask).sum().float()

            if union == 0:
                # If there are no pixels for this class in either prediction or ground truth, consider perfect IoU
                iou = torch.tensor(1.0)
            else:
                iou = intersection / union

            iou_per_class.append(iou)

    # Stack IoU for each class into a tensor
    iou_per_class = torch.stack(iou_per_class)

    return iou_per_class


def cull_to_view_frustum(uv,H, W, points_cam):
    is_valid_x = torch.logical_and(0 <= uv[:, 0], uv[:, 0] < W - 1)
    is_valid_y = torch.logical_and(0 <= uv[:, 1], uv[:, 1] < H - 1)
    #is_valid_z = points_cam[:, 2] > 0
    is_valid_z = torch.logical_and(0 < points_cam[:, 2], points_cam[:, 2] < 60)
    is_valid_points = torch.logical_and(torch.logical_and(is_valid_x, is_valid_y), is_valid_z) #bool
    return is_valid_points


def transfoorm_hd_to_image(hd, img, K, Rt):
    N = hd.shape[2]
    B, H, W = img.shape[0], img.shape[2], img.shape[3]

    hd_homo = torch.cat((hd,
                         torch.ones((B, 1, N), dtype=torch.float32, device=hd.device)),
                        dim=1)  # Bx4xN
    P = Rt.reshape(B,3,4)   # B*3*4
    P_hd_homo = torch.bmm(P, hd_homo)  # Bx3xN
    KP_hd_homo = torch.bmm(K, P_hd_homo)  # Bx3xN

    KP_hd_pxpy = KP_hd_homo[:, 0:2, :] / KP_hd_homo[:, 2:3, :]  # Bx2xN
    KP_hd_pxpy = KP_hd_pxpy.to(dtype=torch.long)

    is_valid_points = cull_to_view_frustum(KP_hd_pxpy, H, W, KP_hd_homo)
    is_valid_points = is_valid_points.bool().unsqueeze(1)
    is_valid_points = is_valid_points.repeat(1,2,1)

    mask = torch.zeros_like(KP_hd_pxpy)
    KP_hd_pxpy = torch.where(is_valid_points, KP_hd_pxpy, mask)

    return KP_hd_pxpy


def calculate_reprojection_error(hd, img, K, Rt):
    B, H, W = img.shape[0], img.shape[2], img.shape[3]
    # Project HD map points to image plane
    KP_hd_pxpy = transfoorm_hd_to_image(hd, img, K, Rt)
    u = KP_hd_pxpy[:, 0, :].long().clamp(0, W - 1)  # Shape: (B, N), convert to integers and clamp to image bounds
    v = KP_hd_pxpy[:, 1, :].long().clamp(0, H - 1)  # Shape: (B, N), convert to integers and clamp to image bounds
    # Index into img tensor using advanced indexing to get the float values at (u, v)
    img_values = img[torch.arange(B).unsqueeze(1), :, v, u]  # Shape: (B, N,C)
    # Optionally, you could use the max probability instead of the raw logits
    img_values = img_values.max(dim=2)[0]  # Shape: (B, N)
    # Create a tensor of ones for hd_values (assuming you want to compare with image values)
    hd_values = torch.ones(B, hd.shape[2]).cuda()  # Shape: (B, N)
    # Compute the error between HD values and the float values from the segmentation map
    err = torch.abs(hd_values - img_values)
    # Calculate mean error across all points in each batch
    err = err.mean(dim=1).unsqueeze(1)  # Shape: (B, 1)

    return err


def calculate_pose_accuracy(gt_Rt, pred_Rt):
    # gt_Rt = np.array(gt_Rt.cpu())
    # P = np.array(P.cpu())

    B = gt_Rt.shape[0]

    new_row = torch.tensor([0, 0, 0, 1], device=gt_Rt.device).view(1, 1, 4).repeat(B, 1, 1)  # Shape: [B, 1, 4]

    # Concatenate the original tensor with the new row along the 1st dimension
    gt = torch.cat((gt_Rt, new_row), dim=1)  # Shape: [B, 4, 4]
    pred = torch.cat((pred_Rt.view(-1, 3, 4), new_row), dim=1)  # Shape: [B, 4, 4]

    # Compute the pseudo-inverse of gt for the entire batch
    gt_inv = torch.inverse(gt)  # Shape: [B, 4, 4]

    # Perform matrix multiplication for the entire batch
    relative_pose = torch.matmul(gt_inv, pred)  # Shape: [B, 4, 4]
    rotation_matrix = relative_pose[:, 0:3, 0:3]  # Shape: [B, 3, 3]
    translation_diff = relative_pose[:, 0:3, 3]  # Shape: [B, 3]

    #batch size is 1
    b = 0
    t = translation_diff[b]  # Extract the translation vector
    r = R.from_matrix(rotation_matrix[b].cpu().numpy())
    euler_angles = r.as_euler('zyx', degrees=False)
    #compare with model-based methods, so output radian degree

    return t, torch.tensor(euler_angles)

def calculate_mean_norm_error(errors):
    t_errors = []
    for t in errors:
        t_err = torch.norm(t)
        t_errors.append(t_err)
    t_errors = torch.tensor(t_errors)
    t_mean = torch.mean(t_errors)

    return t_mean


def calculate_registration_recall(translation_errors, rotation_errors, trans_threshold=0.2, rot_threshold=0.2):
    """
    Calculate the registration recall based on reprojection and rotation errors against predefined thresholds.

    Parameters:
        reprojection_errors (torch.Tensor): Tensor containing reprojection errors of each HD map point.
        rotation_errors (torch.Tensor): Tensor containing rotation errors in degrees.
        trans_threshold (float): Threshold for translation errors to consider registration successful.
        rot_threshold (float): Threshold for rotation errors to consider registration successful.

    Returns:
        float: Registration recall rate.
    """
    t_errors = []
    for t in translation_errors:
        t_err = torch.norm(t)
        t_errors.append(t_err)
    t_errors = torch.tensor(t_errors)

    r_errors = []
    # Iterate through each rotation matrix
    for euler_angles in rotation_errors:
        # Compute the norm of the Euler angles and append to the list
        norm = torch.norm(torch.tensor(euler_angles))
        r_errors.append(norm)
    # Convert the list of norms to a torch tensor if needed
    r_errors = torch.tensor(r_errors)

    # Calculate the number of correct predictions
    correct_predictions = ((t_errors <= trans_threshold) & (r_errors <= rot_threshold)).sum()

    # Calculate recall
    total_predictions = len(t_errors)
    recall = correct_predictions / total_predictions

    return recall


def calculate_metrics(ground_truth, outputs,gt_Rt, pred_Rt,hd, img, K,out_dir_metrics):
    # For pixel accuracy
    #_, predictions = torch.max(outputs.data, 1)
    predictions = outputs
    ground_truth[ground_truth > 0] = 1
    predictions[predictions > 0] = 1
    pixacc,errors = calculate_pix_acc(ground_truth, predictions)
    # For IoU calculation
    iou = calculate_iou(ground_truth, predictions)
    #
    #perr = calculate_reprojection_error(hd, outputs, K, pred_Rt)
    #
    err_t, err_r = calculate_pose_accuracy(gt_Rt, pred_Rt)
    #

    return pixacc,errors,iou,err_t, err_r


def save_one_metric_to_csv(metric_values, metric_name,out_dir_metrics):
    """
    Function to save the loss and accuracy data to a CSV file.
    """

    # Path to save the CSV file
    csv_file_path = os.path.join(out_dir_metrics, metric_name +'.csv')
    values = metric_values[metric_name]
    head = ['index']
    if metric_name == 'err_t':
        head = head + ['dx','dy','dz']
    elif metric_name == 'err_r':
        head= head + ['ax', 'ay', 'az']
    else:
        head.append(metric_name)

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        for idx, v in enumerate(values):
            if v.dim() == 0:
                writer.writerow([idx, v.item()])
            else:
                line = [idx]
                for vv in v:
                    if vv.numel() == 1 and vv.is_floating_point():
                        formatted_value = f'{vv.item():.2f}'
                    else:
                        formatted_value = vv.item()
                    line.append(formatted_value)
                writer.writerow(line)


def save_all_metrics_to_csv(metric_values,out_dir_metrics, filename='eval_metrics.csv'):
    csv_file_path = os.path.join(out_dir_metrics, filename)

    # Write data to CSV
    headers = ['img_na','idx','iou','x','y','z','ax','ay','az', 'pixacc']
    metric_names= ['err_t', 'err_r', 'iou','pixacc']
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        # Get the maximum length of the metrics
        max_len = max(len(v) for v in metric_values.values())
        for idx in range(max_len):
            row = [idx]
            for metric_name, values in metric_values.items():
                if idx < len(values):  # Check if index exists for this metric
                    value = values[idx]
                    if value.dim() == 0:  # Single value
                        row.append(f'{value.item():.2f}')
                    else:  # Multiple values (e.g., dx, dy, dz)
                        row += [f'{vv.item():.2f}' for vv in value]
                else:
                    # Fill missing values for shorter metrics
                    if metric_name in ['err_t', 'err_r']:
                        row += [''] * 3
                    else:
                        row.append('')
            writer.writerow(row)


        line = []
        for k, v in metric_values.items():
            if v.numel() == 1 and v.is_floating_point():
                formatted_value = f'{v.item():.2f}'
            else:
                formatted_value = v.item()
            line.append(formatted_value)
        writer.writerow(line)


def generate_error_heatmap(normalized_errors, out_dir_metrics):

    colors = ["black", "red", "yellow", "white"]  # Dark to light
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    #ax = sns.heatmap(normalized_errors, cmap="viridis", linecolor='black', linewidths=0.5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_errors, cmap=cmap, xticklabels=False, yticklabels=False)
    #plt.title('Aggregate Error Heatmap')
    plt.savefig(out_dir_metrics + '/heatmap.jpg')
    plt.close()  # Close the plot to avoid displaying it


def save_pred_pose(pred_poses,csv_file_path):
    head = ['img_na', 'x', 'y', 'z', 'ax', 'ay', 'az']
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(head)

        for pred_pose in pred_poses:
            img_na = pred_pose[0]
            img_na = os.path.splitext(img_na)[0]
            line = [img_na]
            pred_extrinsic = pred_pose[1]

            rotation = pred_extrinsic[0:3, 0:3]
            translation = pred_extrinsic[0:3, 3]

            r = R.from_matrix(rotation)
            euler_angles = r.as_euler('zyx', degrees=False)
            for i in range(0,3):
                line.append(translation[i])
            for i in range(0,3):
                line.append(euler_angles[i])

            writer.writerow(line)


