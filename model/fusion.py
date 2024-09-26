import torch
import torch.nn as nn
import torch.nn.functional as F   

class HierarchicalFusion(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64):
        super(HierarchicalFusion, self).__init__()
        self.high_branch = nn.Sequential(
            nn.Conv2d(in_high_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_low_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Softmax(dim=1)
        )
        
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, xh, xl):
        high_features = self.high_branch(xh)
        low_features = self.low_branch(xl)
        
        channel_weights = self.channel_attention(high_features)
        high_features = high_features * channel_weights
        
        spatial_weights = self.spatial_attention(high_features)
        high_features = high_features * spatial_weights
        
        combined_features = torch.cat([high_features, low_features], dim=1)
        out = self.fusion_bn(self.fusion_conv(combined_features))
        
        return out
    

class AF_Fusion(nn.Module):
    def __init__(self, in_channels_stage, in_channels_vit, out_channels):
        super(AF_Fusion, self).__init__()

        self.dilated_conv_stage = nn.Conv2d(in_channels_stage, in_channels_stage, kernel_size=3, padding=2, dilation=2)
        self.downsample_vit = nn.Conv2d(in_channels_vit, in_channels_vit, kernel_size=3, stride=2, padding=1)
        self.fusion_conv = nn.Conv2d(in_channels_stage + in_channels_vit, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1) * 0.6)
        self.beta = nn.Parameter(torch.ones(1) * 0.4)

    def forward(self, x_stage, x_vit):
        x_stage_enhanced = self.dilated_conv_stage(x_stage)
        x_vit_downsampled = self.downsample_vit(x_vit)
        x_vit_resized = F.interpolate(x_vit_downsampled, size=x_stage_enhanced.shape[2:], mode='bilinear', align_corners=True)
        stage_weighted = self.alpha * x_stage_enhanced
        vit_weighted = self.beta * x_vit_resized
        fused = torch.cat([stage_weighted, vit_weighted], dim=1)
        out = self.fusion_conv(fused)

        return out