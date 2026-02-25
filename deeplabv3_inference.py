import torch
import argparse
import cv2
import os

from deeplab.utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from deeplab.utils import ALL_CLASSES
from model_da import prepare_model, prepare_model_v3plus_cityscape
import options


if __name__ == '__main__':
    #in_dir = r'E:\data\hn\test\0a132537-3aec-35bb-af13-7faa0811000d\sensors\cameras\ring_front_center\\'
    #out_dir = r'E:\data\hn\temp\deeplabv3\\'

    in_dir = '/mnt/e/data/hn/test/0a132537-3aec-35bb-af13-7faa0811000d/sensors/cameras/ring_front_center/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = options.Options()
    if opt.AddAttention:#deeplab-attention
        model = prepare_model().to(device)
        out_dir = '/mnt/e/data/hn/temp/a1_attention_deeplab/'
    else: #deeplabv3
        model = prepare_model_v3plus_cityscape().to(device)
        out_dir = '/mnt/e/data/hn/temp/a0_deeplabv3/'

    ckpt = torch.load(out_dir + 'best_model.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    opt = options.Options()
    all_image_paths = os.listdir(in_dir)
    for i, image_path in enumerate(all_image_paths):
        print(f"Image {i+1}")
        # Read the image.
        image = Image.open(os.path.join(in_dir, image_path))

        # Resize very large images (if width > 1024.) to avoid OOM on GPUs.
        if image.size[0] > 1024:
            image = image.resize((256, 256))

        # Do forward pass and get the output dictionary.
        outputs = get_segment_labels(image, model, device)
        # Get the data from the `out` key.
        outputs = outputs
        segmented_image = draw_segmentation_map(outputs)

        final_image = image_overlay(image, segmented_image)
        #cv2.imshow('Segmented image', final_image)
        #cv2.waitKey(1)
        cv2.imwrite(os.path.join(out_dir, image_path), final_image)