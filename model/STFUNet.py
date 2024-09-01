import torch
import torch.nn as nn

from model.Swinv2 import SwinViT,SwinTransformerBlock
from model.ResidualBlock import ResidualBlock,_FCNHead
from model.HFM import HierarchicalFusion


class STFUNet(nn.Module):
    def __init__(self, inchans=1, channels=[32, 64, 128, 256], layer=[2, 2, 2, 2], ds=True, mode='train') -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inchans, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self.make_layer(ResidualBlock, channels[0], channels[0], layer[0], stride=2)
        self.stage2 = self.make_layer(ResidualBlock, channels[0], channels[1], layer[1], stride=2)
        self.stage3 = self.make_layer(ResidualBlock, channels[1], channels[2], layer[2], stride=2)
        self.stage4 = self.make_layer(ResidualBlock, channels[2], channels[3], layer[3], stride=2)

        self.vit1 = SwinViT(img_dim=256, in_channels=channels[0], embedding_dim=channels[2],
                            head_num=4, block_num=1, patch_dim=8)
        self.vit2 = SwinViT(img_dim=128, in_channels=channels[1], embedding_dim=channels[2],
                            head_num=4, block_num=1, patch_dim=4)
        self.vit3 = SwinViT(img_dim=64, in_channels=channels[2], embedding_dim=channels[2],
                            head_num=4, block_num=1, patch_dim=2)
        self.vit4 = SwinTransformerBlock(c1=channels[3], c2=channels[3], num_heads=4, num_layers=1)
        self.botteneck = self.make_layer(ResidualBlock,
                                         3 * channels[2] + channels[3] * 2,
                                         channels[3], 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.HFM1 = HierarchicalFusion(in_high_channels=channels[3], in_low_channels=channels[2],
                                           out_channels=channels[2])
        self.decoder3 = self.make_layer(ResidualBlock, channels[2], channels[2], 2)
        self.HFM2 = HierarchicalFusion(in_high_channels=channels[2], in_low_channels=channels[1],
                                           out_channels=channels[1])
        self.decoder2 = self.make_layer(ResidualBlock, channels[1], channels[1], 2)
        self.HFM3 = HierarchicalFusion(in_high_channels=channels[1], in_low_channels=channels[0],
                                           out_channels=channels[0])
        self.decoder1 = self.make_layer(ResidualBlock, channels[0], channels[0], 2)
        self.HFM4 = HierarchicalFusion(in_high_channels=channels[0], in_low_channels=channels[0],
                                           out_channels=channels[0])
        self.decoder0 = self.make_layer(ResidualBlock, channels[0], channels[0] // 2, 2)
        self.head = _FCNHead(channels[0] // 2, 1)
        self.ds = ds
        self.mode = mode

    def make_layer(self, block, inchans, outchans, layers, stride=1):
        layer = []
        layer.append(block(inchans, outchans, stride))
        for _ in range(layers - 1):
            layer.append(block(outchans, outchans, 1))
        return nn.Sequential(*layer)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        bn = self.botteneck(torch.cat([self.vit1(x1),
                                       self.vit2(x2),
                                       self.vit3(x3),
                                       self.vit4(x4), x4], dim=1))
        dx3 = self.decoder3(self.HFM1(self.up(bn), x3))
        dx2 = self.decoder2(self.HFM2(self.up(dx3), x2))
        dx1 = self.decoder1(self.HFM3(self.up(dx2), x1))
        dx0 = self.decoder0(self.HFM4(self.up(dx1), x0))
        out = self.head(dx0)
        if self.ds:
            if self.mode == 'train':
                return out
            else:
                return out
        else:
            return out
        
# if __name__ == '__main__':
#     model = STFUNet()
#     model = model
#     inputs = torch.rand(2, 1, 256, 256)
#     output = model(inputs)
#     print(output.shape)