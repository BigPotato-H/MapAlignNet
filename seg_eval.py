import cv2
import os
import sys

from deeplab.utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from deeplab.utils import ALL_CLASSES
import numpy as np
import options
import torch
import torch.nn as nn
from tqdm import tqdm
from utils_eval import  calculate_pix_acc,calculate_iou, save_one_metric_to_csv
from pose_solver2 import run, get_view_hd_img
from data.argo_data_loader import argoLoader
from model_da import prepare_model,prepare_model_v3plus_cityscape
from utils import image_float_to_int
from deeplab.utils import draw_translucent_seg_maps,view_per_point_label
from rt_transform import output_to_transform_matrix

def evaluate(model,
             val_dataset,
             val_dataloader):
    print('Evaluating')
    model.eval()

    num_batches = int(len(val_dataset) / val_dataloader.batch_size)

    metric_values = {'pixacc':[],
                     'iou':[]}

    error_accumulator = 0
    with torch.no_grad():
        prog_bar = tqdm(val_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0
        for i, data_item in enumerate(prog_bar):
            counter += 1
            #if counter > 10:
            #    break
            #if counter < 319:
            #    continue
            hd, img, img_label, gt_Rt, init_Rt, K, upp, index, img_path, unaligned_label_image = data_item
            hd, img, img_label, gt_Rt, init_Rt, K, upp, unaligned_label_image = \
                hd.to(device), img.to(device), img_label.to(device), \
                gt_Rt.to(device), init_Rt.to(device), K.to(device), upp.to(device), \
                unaligned_label_image.to(device)

            if opt.AddAttention:
                output_poses, output_segs = model(img, upp, unaligned_label_image)
            else:
                output_segs = model(img)

            proj_image = torch.argmax(output_segs, dim=1)
            #####for view
            index = index.cpu().item()
            seg_map = proj_image[0].detach().cpu().numpy()
            seg_mask = cv2.merge([seg_map, seg_map, seg_map])
            seg_image = image_float_to_int(img[0])
            mask = seg_mask > 0
            seg_image[mask[:, :, 0]] = np.array([255,255,0], dtype=np.uint8)
            cv2.imwrite(os.path.join(seg_img_dir, f'{index}.jpg'), seg_image)

            #pixacc, iou
            pixacc, _ = calculate_pix_acc(img_label, proj_image[0])
            iou = calculate_iou(img_label, proj_image[0])

            metric_values['pixacc'].append(pixacc)
            metric_values['iou'].append(iou)


    ##############################
    for k, v in metric_values.items():
        save_one_metric_to_csv(metric_values, k, out_dir_metrics)


if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    opt = options.Options()
    opt.decalib_file =r'/mnt/e/data/hn/temp/perturb_file/commonnet_perturb_file.csv'
    opt.val_seg = True

    #model_na = 'model-seg-last.pth'
    model_na = 'best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.AddAttention:#deeplab-attention
        model = prepare_model().to(device)
        out_dir = '/mnt/e/data/hn/temp/a1_attention_deeplab/'
    else: #deeplabv3
        model = prepare_model_v3plus_cityscape().to(device)
        out_dir = '/mnt/e/data/hn/temp/a0_deeplab/'

    out_dir_metrics = os.path.join(out_dir, 'eval_metrics')
    out_img_dir = os.path.join(out_dir, 'eval_image')
    seg_img_dir = os.path.join(out_dir, 'seg_image')
    pose_solve_dir = '/mnt/e/data/hn/temp/pose_solve/'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_metrics, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(seg_img_dir, exist_ok=True)
    os.makedirs(pose_solve_dir, exist_ok=True)

    val_dataset = argoLoader(opt.input_dir, 'val', opt)


    dataset_size = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=0, drop_last=True, pin_memory=True)
    print('#val images = %d' % len(val_dataset))

    ckpt = torch.load(out_dir + model_na)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval().to(device)

    evaluate(model,
             val_dataset,
             val_dataloader)