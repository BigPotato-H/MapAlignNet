import numpy as np
import cv2
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import pandas as pd

from torchvision import transforms


plt.style.use('ggplot')
plt.rcParams['font.size'] = 14  # Main font size for all text
plt.rcParams['axes.labelsize'] = 14  # Font size for axes labels

ALL_CLASSES = ['background', 'waterbody']

LABEL_COLORS_LIST = [
    (0, 0, 0), # Background.
    (255, 255, 255), # Waterbody.
]

def pix_acc(target, outputs, num_classes):
    target[target > 0] = 1
    outputs[outputs > 0] = 1
    """
    Calculates pixel accuracy, given target and output tensors
    and number of classes.
    """
    if 1:
        labeled = (target > 0) * (target <= num_classes)
        #_, preds = torch.max(outputs.data, 1)
        preds = outputs
        correct = ((preds == target) * labeled).sum().item()
        labeled = labeled.sum()
    else:
        labeled, correct = per_class_accuracy(target, outputs, num_classes)
    return labeled, correct


def per_class_accuracy(target, outputs, num_classes):
    """
    Calculates per-class accuracy.

    Args:
    - target (torch.Tensor): Ground truth labels.
    - outputs (torch.Tensor): Model predictions.
    - num_classes (int): Number of classes.

    Returns:
    - per_class_acc (torch.Tensor): Accuracy for each class.
    """
    _, preds = torch.max(outputs.data, 1)
    correct = (preds == target).float()

    per_class_correct = []
    per_class_total = []

    for cls in range(1, num_classes):
        cls_mask = (target == cls)
        per_class_correct.append((correct * cls_mask).sum().item())
        per_class_total.append(cls_mask.sum().item())

    #per_class_acc = [c / t if t > 0 else 0.0 for c, t in zip(per_class_correct, per_class_total)]

    #return per_class_acc
    return per_class_total.sum(), per_class_correct.sum()


def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g car=0, bus=1.
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

def image_float_to_int(image):
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    mean = np.array([0.45734706, 0.43338275, 0.40058118])
    std = np.array([0.23965294, 0.23532275, 0.2398498])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    idx,
    val_seg_dir, 
    label_colors_list,
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.6 # alpha + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    #seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()
    seg_map = seg_map.detach().cpu().numpy()

    image = image_float_to_int(data[0])

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)


    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 0]
        green_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 1]
        blue_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)

    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{idx}.jpg", image)


def view_per_point_label(data, output, epoch, idx,val_seg_dir,label_colors_list):
    image = data[0]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    mean = np.array([0.45734706, 0.43338275, 0.40058118])
    std = np.array([0.23965294, 0.23532275, 0.2398498])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_label = output[0]  # use only one output from the batch
    img_label = img_label.detach().cpu().numpy()
    #img_label = torch.argmax(img_label.squeeze(), dim=0).detach().cpu().numpy()

    # #########debug
    if 1:
        NCLASSES = len(label_colors_list)
        colors =[[0,0,255], [0,255,0]]
        row_ind, col_ind = np.nonzero(img_label)
        for i in range(0, len(row_ind)):
            cv2.circle(image,
                       (col_ind[i], row_ind[i]),
                       1,
                       colors[1],
                       1)

        cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{idx}_circle.jpg", image)
    return


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, epoch, model, out_dir, name='model'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

def save_model(epochs, model, optimizer, criterion, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, name+'.pth'))

def save_metrics_to_csv(train_acc, valid_acc, train_loss, valid_loss, out_dir, filename='training_metrics.csv'):
    """
    Function to save the loss and accuracy data to a CSV file.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Path to save the CSV file
    csv_file_path = os.path.join(out_dir, filename)

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss'])
        for epoch, (t_acc, v_acc, t_loss, v_loss) in enumerate(zip(train_acc, valid_acc, train_loss, valid_loss), 1):
            writer.writerow([epoch, t_acc, v_acc, t_loss, v_loss])

    print(f"Training metrics saved to {csv_file_path}")

def plot_metrics_from_csv(csv_file, out_dir):
    """
    Function to plot the loss and accuracy from CSV files.
    Args:
    - csv_file_paths (list): List of paths to CSV files containing training metrics.
    - out_dir (str): Directory to save the output plots.
    """
    plt.figure(figsize=(10, 7))

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    label_name = os.path.basename(csv_file).replace('.csv', '')

    # Plot Accuracy
    plt.plot(df['Epoch'], df['Train Accuracy'], linestyle='-', label='Train Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], linestyle='--', label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 7))


    # Plot Loss
    plt.plot(df['Epoch'], df['Train Loss'], linestyle='-', label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], linestyle='--', label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()


def save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

# Define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_segment_labels(image, model, device):
    # transform the image to tensor and load into computation device
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(LABEL_COLORS_LIST)):
        index = labels == label_num
        red_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 0]
        green_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 1]
        blue_map[index] = np.array(LABEL_COLORS_LIST)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 0.5 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def read_class_weights():
    #with open("class_weights.pkl", "rb") as file:  # (needed for python3)
    #    class_weights = np.array(pickle.load(file))
    class_weights = np.array([1,50])
    class_weights = torch.from_numpy(class_weights.astype(np.float32))
    class_weights = class_weights.cuda()
    return class_weights