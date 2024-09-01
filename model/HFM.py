import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalFusion(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(HierarchicalFusion, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.GlobalContext = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.FineGrained = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.multi_scale_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((3, 3))
        ])

        self.conv_after_pooling = nn.ModuleList([
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        ])

        self.final_conv = nn.Conv2d(self.out_channels * 4, self.out_channels, 1, 1, 0)
        self.final_bn = nn.BatchNorm2d(self.out_channels)
    
    def forward(self, xh, xl):
        
        xh = self.feature_high(xh)
        GlobalContext_wei = self.GlobalContext(xh)
        FineGrained_wei = self.FineGrained(xl)
   
        out = 2 * xl * GlobalContext_wei + 2 * xh * FineGrained_wei

        multi_scale_features = [out]  
        for pool, conv in zip(self.multi_scale_pooling, self.conv_after_pooling):
            pooled = pool(out)
            upsampled = F.interpolate(conv(pooled), size=out.size()[2:], mode='bilinear', align_corners=False)
            multi_scale_features.append(upsampled)

        multi_scale_features = torch.cat(multi_scale_features, dim=1)
        out = self.final_bn(self.final_conv(multi_scale_features))
        
        return out


