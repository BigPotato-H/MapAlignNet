import torch
import torch.nn as nn
import torch.nn.functional as F

import deeplabv3plus.network as network
import deeplabv3plus.utils as utils
from deeplabv3plus.network._deeplab import DeepLabHeadV3Plus
from torchvision import models


import os


def prepare_model_v3plus_cityscape(num_classes=2):
    model_name = 'deeplabv3plus_mobilenet'
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    model_path = "deeplabv3plus/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path)["model_state"]
        model_dict = model.state_dict()
        # 模型参数更新
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.classifier.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
    return model


class SpatialEmbedding(nn.Module):
    def __init__(self, embedding_dim, pt_num):
        super(SpatialEmbedding, self).__init__()
        # Define an embedding for each point in the pt_num
        self.spatial_embed = nn.Parameter(torch.randn(pt_num, embedding_dim))

    def forward(self, projected_points):
        batch_size, _, pt_num = projected_points.size()

        # Expand the spatial embedding to match the batch size
        spatial_context = self.spatial_embed.expand(batch_size, -1, -1)  # Shape: (batch_size, pt_num, embedding_dim)

        # Reshape spatial_context to match (batch_size, embedding_dim, pt_num) for concatenation
        spatial_context = spatial_context.permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, pt_num)

        # Concatenate the original projected points with the spatial embeddings along the channel dimension
        # The input projected_points shape is (batch_size, 3, pt_num)
        # The concatenated shape will be (batch_size, 3 + embedding_dim, pt_num)
        return torch.cat([projected_points, spatial_context], dim=1)

class CMSAttention2(nn.Module):
    def __init__(self, img_dim=512, map_dim=512, out_dim=512, value_dim=512):
        super(CMSAttention2, self).__init__()
        # Define the layers for image features
        self.theta_conv = nn.Conv2d(in_channels=img_dim, out_channels=img_dim, kernel_size=1)

        # Define the layers for map features
        self.phi_conv = nn.Conv2d(in_channels=map_dim, out_channels=img_dim, kernel_size=1)

        # Define the layers to process the combined features
        self.value_conv = nn.Conv2d(in_channels=img_dim, out_channels=value_dim, kernel_size=1)
        self.feats_conv = nn.Conv1d(in_channels=1024, out_channels=out_dim, kernel_size=1)
        # Stride of 32 will downsample from 512 to 16
        self.out_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=(2, 2), stride=(2, 2))
        #self.out_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1)

    def forward(self, img_feats, map_feats):
        batch_size, img_dim, height, width = img_feats.size()
        # Process image features
        theta = self.theta_conv(img_feats)  # Shape: [batch_size, img_dim, height, width]
        theta = theta.view(batch_size, img_dim, -1)  # Shape: [batch_size, img_dim, height*width]

        # Process map features
        phi = self.phi_conv(map_feats)  # [batch_size, img_dim, 1, seq_len]
        phi = phi.squeeze(2)  # Remove the singleton dimension after convolution: [batch_size, img_dim, seq_len]

        # Value
        value = self.value_conv(img_feats)  # Apply value_conv to generate V
        value = value.view(batch_size, -1, height * width)

        # Attention mechanism
        attention_map = torch.bmm(theta.permute(0, 2, 1), phi)  # [batch_size, height*width, seq_len]
        attention_map = F.softmax(attention_map, dim=-1)  # Normalize the attention weights

        weighted_value = torch.bmm(value, attention_map)
        weighted_value = weighted_value.view(batch_size, img_dim, height*2, width*2)
        #weighted_value = weighted_value.view(batch_size, img_dim, height, width)
        # Final output processing
        feats = self.out_conv(weighted_value)  # [batch_size, out_dim, height, width]
        # Reshape the feature map to [batch_size, intermediate_dim, height, width]
        feats = feats.view(batch_size, img_dim, height, width)
        feats = feats + img_feats  # Residual connection: adding original image features

        return feats

def remove_layer(model, n):
    modules = list(model.children())[:-n]
    model = nn.Sequential(*modules)
    return model

class PoseSolverHead(nn.Module):
    def __init__(self, init_translation=None, init_euler=None):
        super(PoseSolverHead, self).__init__()
        self.depth_net = remove_layer(models.resnet18(pretrained=False), 2)
        modules = list(self.depth_net.children())
        modules[0] = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.depth_net = nn.Sequential(*modules)
        self.matching = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 6)
        #self.fc2 = nn.Linear(256, 12)

        # Custom initialization for the output layer to start with identity transformation
        if init_translation is not None and init_euler is not None:
            nn.init.constant_(self.fc2.weight, 0)  # Initialize weights to zero

            # Initialize translation part
            self.fc2.bias.data[0] = init_translation[0].item()
            self.fc2.bias.data[1] = init_translation[1].item()
            self.fc2.bias.data[2] = init_translation[2].item()

            # Initialize Euler angles part
            self.fc2.bias.data[3] = init_euler[0].item()
            self.fc2.bias.data[4] = init_euler[1].item()
            self.fc2.bias.data[5] = init_euler[2].item()

    def forward(self, rgb_features, depth_img):
        depth_features = self.depth_net(depth_img)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc1(matching_features)
        x = self.fc2(x)
        return x
        pose = x[:, :6]  # Extract pose components
        uncertainty = x[:, 6:]  # Extract uncertainty components (variances)

        return pose, uncertainty


class MapAttentionSegmentation(nn.Module):
    def __init__(self,model_deeplab, img_channels=256, map_channels=3, pt_num=1024,
                 height=16, width=16, attention_dim=512):
        super(MapAttentionSegmentation, self).__init__()

        # Backbone to extract image features
        self.backbone = model_deeplab.backbone
        self.project = model_deeplab.classifier.project
        self.aspp = model_deeplab.classifier.aspp  # ASPP module
        self.img_conv1x1=nn.Conv2d(in_channels=img_channels, out_channels=attention_dim, kernel_size=1)

        # Linear projection for map features to match spatial dimensions of image features
        self.map_projection = nn.Linear(pt_num, height * width)
        self.spatial_embedding = SpatialEmbedding(embedding_dim=16, pt_num=pt_num)
        self.map_conv1x1 = nn.Conv2d(in_channels=16+3, out_channels=attention_dim, kernel_size=1)

        # Attention mechanism to combine features
        #self.attention = CMSAttention(img_dim=attention_dim, map_dim=attention_dim, out_dim=attention_dim)
        self.attention = CMSAttention2(img_dim=attention_dim, map_dim=attention_dim, out_dim=attention_dim)
        self.attention_img_conv1x1 = nn.Conv2d(in_channels=attention_dim, out_channels=img_channels, kernel_size=1)
        # Final segmentation head
        self.segmentation_head = model_deeplab.classifier.classifier

        #####with pose solver module
        self.lambda_seg = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.lambda_pose_proj = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, image, project_points):
        feats = self.backbone(image)

        low_level_feature = self.project(feats['low_level'])

        # Pass through ASPP
        aspp_out = self.aspp(feats['out'])
        output_feature = self.img_conv1x1(aspp_out)

        # Process map features through spatial embedding
        enriched_map_features = self.spatial_embedding(project_points)  # [batch_size, 3 + embedding_dim, pt_num]
        enriched_map_features = self.map_conv1x1(
            enriched_map_features.unsqueeze(3))  # [batch_size, attention_dim, pt_num, 1]
        enriched_map_features = enriched_map_features.permute(0, 1, 3, 2)

        # Apply CMSAttention to combine the features
        combined_feats = self.attention(output_feature, enriched_map_features)
        combined_feats = self.attention_img_conv1x1(combined_feats)
        output_feature = F.interpolate(combined_feats, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        # Apply the final segmentation head
        output = self.segmentation_head(torch.cat([low_level_feature, output_feature], dim=1 ))# Shape: [batch_size, 1, height, width]
        input_shape = image.shape[-2:]
        # Upsample if needed (e.g., to original image resolution)
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)

        return output

class MapAttentionPoseNet(nn.Module):
    def __init__(self,model_deeplab, img_channels=256, map_channels=3, pt_num=1024,
                 height=16, width=16, attention_dim=512):
        super(MapAttentionPoseNet, self).__init__()

        # Backbone to extract image features
        self.backbone = model_deeplab.backbone
        self.project = model_deeplab.classifier.project
        self.aspp = model_deeplab.classifier.aspp  # ASPP module
        self.img_conv1x1=nn.Conv2d(in_channels=img_channels, out_channels=attention_dim,
                                   kernel_size=(2,2), stride=(2,2))

        # Linear projection for map features to match spatial dimensions of image features
        self.map_projection = nn.Linear(pt_num, height * width)
        self.spatial_embedding = SpatialEmbedding(embedding_dim=16, pt_num=pt_num)
        self.map_conv1x1 = nn.Conv2d(in_channels=16+3, out_channels=attention_dim, kernel_size=1)

        # Attention mechanism to combine features
        #self.attention = CMSAttention(img_dim=attention_dim, map_dim=attention_dim, out_dim=attention_dim)
        self.attention = CMSAttention2(img_dim=attention_dim, map_dim=attention_dim, out_dim=attention_dim)
        self.attention_img_conv1x1 = nn.Conv2d(in_channels=attention_dim, out_channels=img_channels, kernel_size=1)
        # Final segmentation head
        self.segmentation_head = model_deeplab.classifier.classifier

        #####with pose solver module
        self.lambda_seg = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.lambda_pose_proj = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        #####pose solver head
        #whu
        self.pose_solver_head = PoseSolverHead(init_translation=torch.tensor([0,0,0]),
                                               init_euler=torch.tensor([0,-0.07, 0]))

        #argo
        #self.pose_solver_head = PoseSolverHead(init_translation=torch.tensor([0, 0, 0]),
        #                                       init_euler=torch.tensor([0,-1.57, 1.57]))

    def forward(self, image, project_points,unaligned_label_image):
        feats = self.backbone(image)

        low_level_feature = self.project(feats['low_level'])

        # Pass through ASPP
        aspp_out = self.aspp(feats['out'])
        output_feature = self.img_conv1x1(aspp_out)

        # Process map features through spatial embedding
        enriched_map_features = self.spatial_embedding(project_points)  # [batch_size, 3 + embedding_dim, pt_num]
        enriched_map_features = self.map_conv1x1(
            enriched_map_features.unsqueeze(3))  # [batch_size, attention_dim, pt_num, 1]
        enriched_map_features = enriched_map_features.permute(0, 1, 3, 2)

        # Apply CMSAttention to combine the features
        combined_feats = self.attention(output_feature, enriched_map_features)
        output_pose = self.pose_solver_head(combined_feats, unaligned_label_image)

        if 0:
            #####seg head
            combined_feats = self.attention_img_conv1x1(combined_feats)
            output_feature = F.interpolate(combined_feats, size=low_level_feature.shape[2:], mode='bilinear',
                                           align_corners=False)
            # Apply the final segmentation head
            output = self.segmentation_head(torch.cat([low_level_feature, output_feature], dim=1 ))# Shape: [batch_size, 1, height, width]
            input_shape = image.shape[-2:]
            # Upsample if needed (e.g., to original image resolution)
            output_seg = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
            return output_pose, output_seg

        return output_pose

# Instantiate the integrated DeepLabv3+ model with attention
def prepare_model():
    model_deeplab = prepare_model_v3plus_cityscape()

    #model_deeplab = MapAttentionSegmentation(model_deeplab)

    model_deeplab = MapAttentionPoseNet(model_deeplab)
    return model_deeplab
