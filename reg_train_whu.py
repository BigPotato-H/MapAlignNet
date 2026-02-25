import torch
import torch.nn as nn
from torch.utils.data import Subset
import os
from reg_engine_whu import train, validate
from model_da import prepare_model
from deeplab.utils import ALL_CLASSES, LABEL_COLORS_LIST, read_class_weights
from deeplab.utils import save_model, SaveBestModel, save_plots,save_metrics_to_csv, plot_metrics_from_csv
from data.whu_data_loader import whuLoader
import options_whu
import numpy as np
import pickle
import sys


if __name__ == '__main__':
      # Create a directory with the model name for outputs.
    opt = options_whu.Options()
    opt.train_seg = False

    out_dir = '/mnt/f/hn/data/000-bigdata/whu/alignnet/'
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)
    out_loss_csv_name = 'loss.csv'
    if 0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = prepare_model().to(device)
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

        # Load the saved model and optimizer state dicts
        start_epoch = 0
        if opt.pre_ckpt is not None:
            checkpoint = torch.load(out_dir + opt.pre_ckpt)  # Replace with the actual path to your checkpoint file
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
            #loss = checkpoint['loss']

        class_weights = read_class_weights()
        criterion = nn.CrossEntropyLoss(class_weights)

        classes_to_train = ALL_CLASSES

        train_dataset = whuLoader(opt.input_dir, 'train', opt)
        val_dataset = whuLoader(opt.input_dir, 'val', opt)

        if 0:
            num_data_points = 100
            subset_indices = range(num_data_points)
            train_dataset = Subset(train_dataset, subset_indices)
            num_data_points = 20
            subset_indices = range(num_data_points)
            val_dataset = Subset(val_dataset, subset_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.dataloader_threads, drop_last=True, pin_memory=True)
        dataset_size = len(train_dataset)
        print('#training images = %d' % len(train_dataset))
        print('#val images = %d' % len(val_dataset))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.dataloader_threads, pin_memory=True)

        # Initialize `SaveBestModel` class.
        save_best_model = SaveBestModel()

        EPOCHS = opt.epochs
        train_loss, train_pix_acc = [], []
        valid_loss, valid_pix_acc = [], []


        for epoch in range (start_epoch, EPOCHS):
            print(f"EPOCH: {epoch + 1}")
            train_epoch_loss, train_epoch_pixacc = train(
                model,
                train_dataset,
                train_loader,
                device,
                optimizer,
                criterion,
                classes_to_train,
                opt
            )
            valid_epoch_loss, valid_epoch_pixacc = validate(
                model,
                val_dataset,
                val_loader,
                device,
                criterion,
                classes_to_train,
                LABEL_COLORS_LIST,
                epoch,
                ALL_CLASSES,
                out_dir_valid_preds,
                opt
            )
            train_loss.append(train_epoch_loss.detach().cpu().numpy())
            train_pix_acc.append(train_epoch_pixacc.cpu().numpy())
            valid_loss.append(valid_epoch_loss.detach().cpu().numpy())
            valid_pix_acc.append(valid_epoch_pixacc.cpu().numpy())

            save_best_model(
                valid_epoch_loss, epoch, model, out_dir
            )

            print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
            print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")
            print('-' * 50)

        save_model(EPOCHS, model, optimizer, criterion, out_dir)
        # Save the loss and accuracy plots.

        save_metrics_to_csv(train_pix_acc, valid_pix_acc, train_loss, valid_loss, out_dir, out_csv_name)

    # draw train curve
    if 1:
        csv_file = out_dir + out_loss_csv_name
        plot_metrics_from_csv(csv_file, out_dir)
        #save_plots(train_pix_acc, valid_pix_acc, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')