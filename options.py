import numpy as np
import math
import torch

data_series = [
    {"train_id": "/mnt/f/hn/data/train-001/sensor/train/",
     "log_id": "1bf2bf1c-64d1-308f-afd1-220de9d30290",
     "scenario_type": ["雨", "夜"]},
    {"train_id": "/mnt/f/hn/data/train-001/sensor/train/",
     "log_id": "156a412d-3699-3c1c-9ada-6ab587347996",
     "scenario_type": ["桥下"]},
    {"train_id": "/mnt/f/hn/data/train-002/sensor/train/",
     "log_id": "3a789fb0-5cd2-3710-b8ea-f32fce38e3ca",
     "scenario_type": ["桥下"]},
    {"train_id": "/mnt/f/hn/data/train-tbv/",
     "log_id": "0WHfUzbQi8mopDFgQ7BH7lGjgCEygaoN__Summer_2020",
     "scenario_type": ["桥下"]},
    {"train_id": "/mnt/f/hn/data/train-tbv/",
     "log_id": "0IHD1W1ypZyr2F1I2aA6n5SiRIOZ87RX__Summer_2020",
     "scenario_type": ["无线"]},
    {"train_id": "/mnt/f/hn/data/train-tbv/",
     "log_id": "2OKDOf9DhoxaVC17YM5a4DN78RcUqj8i__Summer_2020",
     "scenario_type": ["无线"]}
]

class Options:
    def __init__(self):
        #self.dataroot = 'D:/0huinian/data/2UH38_20190912_3/003_wuhan3huan_03/'
        #self.dataroot = '/data/nianh/2UH38_20190912_3/003_wuhan3huan_03/'
        #self.dataroot = '/mnt/e/data/hn/'
        self.input_dir = '/mnt/f/hn/data/000-bigdata/input/'
        self.output_dir = '/mnt/f/hn/data/000-bigdata/output/'
        self.log_id = ''
        #self.dataroot = r'E:/data/hn/train-000/sensor/train/'
        #self.pre_ckpt = 'best_model.pth'
        self.pre_ckpt = None
        #self.pre_ckpt = 'model-seg-last.pth'
        self.version = '5.10'
        self.crop_original_top_rows = 500
        self.img_scale = 0.4
        self.img_H = 512
        self.img_W = 512
        self.input_pt_num = 1024

        # CAM coordinate
        self.P_tx_amplitude = 0.5
        self.P_ty_amplitude = 0.5
        self.P_tz_amplitude = 1
        self.P_Rx_amplitude = 1.0 * math.pi / 90.0
        self.P_Ry_amplitude = 1.0 * math.pi / 90.0
        self.P_Rz_amplitude = 1.0 * math.pi / 90.0

        self.dataloader_threads = 1

        self.batch_size = 6
        #self.batch_size = 1
        self.gpu = '0'
        #self.device = torch.device('cuda', int(self.gpu))
        #self.normalization = 'batch'
        #self.norm_momentum = 0.1
        #self.activation = 'relu'
        self.lr = 0.001

        self.epochs = 30
        self.AddAttention = True
        #self.AddAttention = False
        self.decalib_file = None
        #self.train_seg = True
        self.train_seg = False
        self.val_seg = True

