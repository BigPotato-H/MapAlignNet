import cv2
import os
import sys

root_path=r'F:\hn\code\000-big\VisionHDRegNet\\'
sys.path.append(root_path)
sys.path.append('models')
#absolute_path = os.path.abspath('data/argoverse')
#sys.path.append(absolute_path)
sys.path.append('data/argoverse')
sys.path.append('models/deeplab')
from PIL import Image
from deeplab.utils import ALL_CLASSES
import numpy as np
import options
import torch
import torch.nn as nn
from tqdm import tqdm
from utils_eval import calculate_metrics, calculate_registration_recall,\
    calculate_mean_norm_error, save_one_metric_to_csv, save_all_metrics_to_csv,\
    generate_error_heatmap,save_pred_pose
from pose_solver2 import run, get_view_hd_img
from data.argo_data_loader import argoLoader
from model_da import prepare_model
from utils import image_float_to_int

from rt_transform import output_to_transform_matrix
import pandas as pd
import argparse


def evaluate(model,
             val_dataset,
             val_dataloader):
    print('Evaluating')
    model.eval()

    num_batches = int(len(val_dataset) / val_dataloader.batch_size)

    pandaset_df = pd.DataFrame()
    error_accumulator = 0
    with torch.no_grad():
        prog_bar = tqdm(val_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0

        pred_poses = []
        for i, data_item in enumerate(prog_bar):
            counter += 1
            #if counter > 3:
            #    break
            #if counter < 319:
            #    continue
            hd, img, img_label, gt_Rt, init_Rt, K, upp, index, img_path, unaligned_label_image = data_item
            hd, img, img_label, gt_Rt, init_Rt, K, upp, unaligned_label_image = \
                hd.to(device), img.to(device), img_label.to(device), \
                gt_Rt.to(device), init_Rt.to(device), K.to(device), upp.to(device), \
                unaligned_label_image.to(device)

            img_path = img_path[0]
            img_na = os.path.basename(img_path)

            #### for compasrison experiment, if eor image doesn't exist, skip this image
            if 1:
                eor_dir = os.path.join(opt.output_dir, log_id, 'ply-1eor/')
                if not os.path.exists(eor_dir + img_na):
                    continue
            output_poses = model(img, upp, unaligned_label_image)
            if val_dataloader.batch_size == 1:
                output_poses = output_poses.unsqueeze(0)
            pred_Rt = output_to_transform_matrix(output_poses)

           #reprojection_loss, pred_Rt = run(hd, outputs, K,init_Rt, gt_Rt=None)

            proj_image = get_view_hd_img(hd, img.permute(0,2,3,1), pred_Rt, K)

            #_, pred_pose, proj_image = run(hd, output_segs, K, pred_Rt)

            img = cv2.imread(img_path)
            index = index.item()
            gt_extrinsic = np.vstack([gt_Rt[0,:,:].detach().cpu().numpy(), [0, 0, 0, 1]])
            gt_label_image, gt_overlap_image= val_dataset.get_label_image(index, img, gt_extrinsic,[0, 255, 0])

            init_extrinsic = np.vstack([init_Rt[0, :, :].detach().cpu().numpy(), [0, 0, 0, 1]])
            _, init_overlap_image = val_dataset.get_label_image(index, img, init_extrinsic, [0, 0, 255])

            pred_Rt = pred_Rt.view(-1, 3, 4)
            pred_extrinsic = np.vstack([pred_Rt[0, :, :].detach().cpu().numpy(), [0, 0, 0, 1]])
            pred_label_image, pred_overlap_image = val_dataset.get_label_image(index, img, pred_extrinsic,[255,0, 0])
            mosaic_image = np.hstack((gt_overlap_image, init_overlap_image, pred_overlap_image))

            cv2.imwrite(os.path.join(out_pred_dir, img_na), pred_overlap_image)
            cv2.imwrite(os.path.join(out_img_dir, img_na), mosaic_image)

            pred_poses.append((img_na, pred_extrinsic))
            # # # # #  Calculate each metric
            pred_label_image = val_dataset.get_label_mask(pred_label_image)
            pred_label_image = torch.from_numpy(pred_label_image.astype(np.float32)).unsqueeze(0)
            gt_label_image = val_dataset.get_label_mask(gt_label_image)
            gt_label_image = torch.from_numpy(gt_label_image.astype(np.float32)).unsqueeze(0)

            #pixacc,errors, iou, t_err, r_err =calculate_metrics(img_label, pred_label_image,gt_Rt, pred_Rt,hd, img, K,out_dir_metrics)
            pixacc, errors, iou, t_err, r_err = calculate_metrics(gt_label_image, pred_label_image, gt_Rt, pred_Rt, hd, img,
                                                                  K, out_dir_metrics)
            row = {'img_na':img_na,
                   'idx':index,
                   'iou':iou,
                   'x':t_err[0].item(),
                   'y':t_err[1].item(),
                   'z':t_err[2].item(),
                   'ax':r_err[0].item(),
                   'ay':r_err[1].item(),
                   'az':r_err[2].item(),
                   'pixacc':pixacc.item()}
            pandaset_df = pd.concat([pandaset_df, pd.DataFrame(row)], ignore_index=True)
            error_accumulator += errors
            '''metric_values['pixacc'].append(pixacc)
            metric_values['iou'].append(iou)
            metric_values['err_t'].append(t_err)
            metric_values['err_r'].append(r_err)'''

    error_accumulator = error_accumulator / counter
    #generate_error_heatmap(error_accumulator, out_dir_metrics)

    ##############################
    pandaset_df = pandaset_df.round(4)
    csv_path = os.path.join(out_dir_metrics, "eval_metrics.csv")
    pandaset_df.to_csv(csv_path, index=False)
    #for k, v in metric_values.items():
    #    save_one_metric_to_csv(metric_values, k, out_dir_metrics)

    ####################################
    csv_path = os.path.join(out_dir_metrics, "alignnet_pred_poses.csv")
    save_pred_pose(pred_poses, csv_path)



def find_item_by_log_id(data_series, search_log_id):
    for item in data_series:
        if item["log_id"][0:3] == search_log_id:
            return item
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the program for a specific log.")
    parser.add_argument(
        "--log_id",
        type=str,
        required=True,
        help="logid"
    )
    args = parser.parse_args()
    data_series = find_item_by_log_id(options.data_series, args.log_id)
    log_id = 'arg' + args.log_id
    # Create a directory with the model name for outputs.
    opt = options.Options()
    opt.input_dir = data_series["train_id"]
    opt.log_id = data_series["log_id"]
    dataroot = opt.input_dir

    opt.decalib_file =r'/mnt/e/data/hn/temp/perturb_file-1500/commonnet_perturb_file.csv'
    out_dir = os.path.join(opt.output_dir, log_id, 'AlignNet/')
    os.makedirs(out_dir, exist_ok=True)

    # model_na = 'best_model-new.pth'
    # model_na = 'best_model.pth'
    model_na = 'best_model-pose-1-14.pth'

    out_dir_metrics = os.path.join(out_dir, 'eval_metrics')
    out_img_dir = os.path.join(out_dir, 'eval_image')
    out_pred_dir = os.path.join(out_dir, 'eval_image_single')
    pose_solve_dir = os.path.join(out_dir, 'pose_solve')

    os.makedirs(out_dir_metrics, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)

    os.makedirs(pose_solve_dir, exist_ok=True)

    val_dataset = argoLoader(dataroot, 'val', opt)


    dataset_size = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=0, drop_last=True, pin_memory=True)
    print('#val images = %d' % len(val_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = prepare_model()

    ckpt = torch.load(opt.output_dir + model_na)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval().to(device)

    evaluate(model,
             val_dataset,
             val_dataloader)