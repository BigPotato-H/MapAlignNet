import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from deeplab.utils import draw_translucent_seg_maps, view_per_point_label, pix_acc
from pose_solver2 import run, get_view_hd_img
from rt_transform import output_to_transform_matrix,transformation_loss, test,\
transformation_uncertainty_loss


def train(model,
          train_dataset,
          train_dataloader,
          device,
          optimizer,
          criterion,
          classes_to_train,
          opt):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct, train_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(train_dataset) / train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0  # to keep track of batch counter
    num_classes = len(classes_to_train)

    for i, data_item in enumerate(prog_bar):
        counter += 1

        map, img, img_label, gt_Rt, init_Rt, K, upp, index, img_path,unaligned_label_image = data_item
        map, img, img_label, gt_Rt, init_Rt, K, upp,unaligned_label_image = \
            map.to(device), img.to(device), img_label.to(device), \
            gt_Rt.to(device), init_Rt.to(device), K.to(device), upp.to(device), \
            unaligned_label_image.to(device)

        #test(gt_Rt)

        optimizer.zero_grad()
        output_poses, uncertainty = model(img, upp,unaligned_label_image)
        uncertainties = F.softplus(uncertainties)

        pred_P = output_to_transform_matrix(output_poses)
        #loss_pose = transformation_loss(pred_P, gt_Rt)
        loss_pose = transformation_uncertainty_loss(pred_P, gt_Rt, uncertainties)

        loss = loss_pose

        if 1:
            reprojection_loss, enhance_P, proj_image = run(map, img, K, pred_P, gt_Rt)
            pred_P = enhance_P
        else:
            proj_image = get_view_hd_img(map, img.permute(0,2,3,1), pred_P,K)

        #lambda_seg = torch.exp(model.lambda_seg)
        #lambda_pose = torch.exp(model.lambda_pose_proj)
        #loss = lambda_seg * loss_seg + lambda_pose * loss_pose
        train_running_loss += loss
        ###########################

        # For pixel accuracy.
        labeled, correct = pix_acc(img_label, proj_image, num_classes)
        train_running_label += labeled
        train_running_correct += correct
        train_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
        #############################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        prog_bar.set_description(
            desc=f"Loss: {loss.detach().cpu().numpy():.4f} | PixAcc: {train_running_pixacc.cpu().numpy() * 100:.2f}")

    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy
    pixel_acc = ((1.0 * train_running_correct) / (np.spacing(1) + train_running_label)) * 100
    ##############################
    return train_loss, pixel_acc


def validate(model,
             valid_dataset,
             valid_dataloader,
             device,
             criterion,
             classes_to_train,
             label_colors_list,
             epoch,
             all_classes,
             save_dir,
             opt):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct, valid_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(valid_dataset) / valid_dataloader.batch_size)
    num_classes = len(classes_to_train)

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0  # To keep track of batch counter.
        for i, data_item in enumerate(prog_bar):
            counter += 1

            map, img, img_label, gt_Rt, init_Rt, K, upp, index, img_path, unaligned_label_image = data_item
            map, img, img_label, gt_Rt, init_Rt, K, upp, unaligned_label_image = \
                map.to(device), img.to(device), img_label.to(device), \
                gt_Rt.to(device), init_Rt.to(device), K.to(device), upp.to(device), \
                unaligned_label_image.to(device)

            output_poses, uncertainty = model(img, upp, unaligned_label_image)
            uncertainties = F.softplus(uncertainties)

            pred_P = output_to_transform_matrix(output_poses)
            # loss_pose = transformation_loss(pred_P, gt_Rt)
            loss_pose = transformation_uncertainty_loss(pred_P, gt_Rt, uncertainties)
            loss = loss_pose
            proj_image = get_view_hd_img(map, img.permute(0,2,3,1), pred_P, K)

            #lambda_seg = torch.exp(model.lambda_seg)
            #lambda_pose = torch.exp(model.lambda_pose_proj)
            #loss = lambda_seg * loss_seg + lambda_pose * loss_pose
            valid_running_loss += loss

            if i == 0:
                view_per_point_label(
                    img,
                    proj_image,
                    epoch,
                    i,
                    save_dir,
                    label_colors_list,
                )
            if i == num_batches - 1:
                draw_translucent_seg_maps(
                    img,
                    proj_image,
                    epoch,
                    i,
                    save_dir,
                    label_colors_list,
                )

            ###########################

            # For pixel accuracy.
            labeled, correct = pix_acc(img_label, proj_image, num_classes)
            valid_running_label += labeled
            valid_running_correct += correct
            valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled)
            #############################

            prog_bar.set_description(
                desc=f"Loss: {loss.detach().cpu().numpy():.4f} | PixAcc: {valid_running_pixacc.cpu().numpy() * 100:.2f}")

    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy.
    pixel_acc = ((1.0 * valid_running_correct) / (np.spacing(1) + valid_running_label)) * 100.
    ##############################
    return valid_loss, pixel_acc