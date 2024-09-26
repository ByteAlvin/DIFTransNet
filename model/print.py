import torch
import torch.nn as nn
from swinvit import SwinViT, SwinTransformerBlock
from Conv import ResidualBlock, _FCNHead
from fusion import HierarchicalFusion, AF_Fusion

class STFUNet(nn.Module):
    def __init__(self, inchans=1, channels=[32, 64, 128, 256], layer=[2, 2, 2, 2], ds=True, mode='train') -> None:
        super(STFUNet, self).__init__()
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

        self.vit1 = SwinViT(in_channels=channels[0], embedding_dim=channels[2], head_num=4, block_num=1, patch_dim=8)
        self.vit2 = SwinViT(in_channels=channels[1], embedding_dim=channels[2], head_num=4, block_num=1, patch_dim=4)
        self.vit3 = SwinViT(in_channels=channels[2], embedding_dim=channels[2], head_num=4, block_num=1, patch_dim=2)
        self.vit4 = SwinTransformerBlock(c1=channels[3], c2=channels[3], num_heads=4, num_layers=1)

        self.fusion1 = AF_Fusion(channels[0], channels[2], channels[0])
        self.fusion2 = AF_Fusion(channels[1], channels[2], channels[1])
        self.fusion3 = AF_Fusion(channels[2], channels[2], channels[2])
        self.fusion4 = AF_Fusion(channels[3], channels[3], channels[3])

        self.botteneck = self.make_layer(ResidualBlock, 3 * channels[2] + channels[3] * 2, channels[3], 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.HFM1 = HierarchicalFusion(in_high_channels=channels[3], in_low_channels=channels[2], out_channels=channels[2])
        self.decoder3 = self.make_layer(ResidualBlock, channels[2], channels[2], 2)
        self.HFM2 = HierarchicalFusion(in_high_channels=channels[2], in_low_channels=channels[1], out_channels=channels[1])
        self.decoder2 = self.make_layer(ResidualBlock, channels[1], channels[1], 2)
        self.HFM3 = HierarchicalFusion(in_high_channels=channels[1], in_low_channels=channels[0], out_channels=channels[0])
        self.decoder1 = self.make_layer(ResidualBlock, channels[0], channels[0], 2)
        self.HFM4 = HierarchicalFusion(in_high_channels=channels[0], in_low_channels=channels[0], out_channels=channels[0])
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
        print("Stem output shape:", x0.shape)

        x1 = self.stage1(x0)
        print("Stage 1 output shape:", x1.shape)

        v1 = self.vit1(x1)
        print("ViT 1 output shape:", v1.shape)

        x1_fused = self.fusion1(x1, v1)
        print("Fused Stage 1 output shape:", x1_fused.shape)

        x2 = self.stage2(x1_fused)
        print("Stage 2 output shape:", x2.shape)

        v2 = self.vit2(x2)
        print("ViT 2 output shape:", v2.shape)

        x2_fused = self.fusion2(x2, v2)
        print("Fused Stage 2 output shape:", x2_fused.shape)

        x3 = self.stage3(x2_fused)
        print("Stage 3 output shape:", x3.shape)

        v3 = self.vit3(x3)
        print("ViT 3 output shape:", v3.shape)

        x3_fused = self.fusion3(x3, v3)
        print("Fused Stage 3 output shape:", x3_fused.shape)

        x4 = self.stage4(x3_fused)
        print("Stage 4 output shape:", x4.shape)

        v4 = self.vit4(x4)
        print("ViT 4 output shape:", v4.shape)

        x4_fused = self.fusion4(x4, v4)
        print("Fused Stage 4 output shape:", x4_fused.shape)

        bn = self.botteneck(torch.cat([v1, v2, v3, v4, x4_fused], dim=1))
        print("Bottleneck output shape:", bn.shape)

        dx3 = self.decoder3(self.HFM1(self.up(bn), x3))
        print("Decoder 3 output shape:", dx3.shape)

        dx2 = self.decoder2(self.HFM2(self.up(dx3), x2))
        print("Decoder 2 output shape:", dx2.shape)

        dx1 = self.decoder1(self.HFM3(self.up(dx2), x1))
        print("Decoder 1 output shape:", dx1.shape)

        dx0 = self.decoder0(self.HFM4(self.up(dx1), x0))
        print("Decoder 0 output shape:", dx0.shape)

        out = self.head(dx0)
        print("Final output shape:", out.shape)

        if self.ds:
            if self.mode == 'train':
                return out
            else:
                return out
        else:
            return out


if __name__ == '__main__':
    model = STFUNet()
    model = model
    inputs = torch.rand(2, 1, 288, 224)
    output = model(inputs)
    print(output.shape)

