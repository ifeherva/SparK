import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model


class ResBlock(nn.Module):

    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True))
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)

        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc), nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class Encoder(nn.Module):

    def __init__(self, in_channels, ngf, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.stage1 = ResBlock(in_channels, ngf, norm_layer=norm_layer, scale='down')  # 128
        self.stage2 = ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down')  # 64
        self.stage3 = ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down')  # 32
        self.stage4 = ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 16
        self.stage5 = ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8

        self.feature_map_channels = [ngf, ngf * 2, ngf * 4, ngf * 4, ngf * 4]

    def forward(self, x, hierarchical=False):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return [x1, x2, x3, x4, x5]

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 32

    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.feature_map_channels


@register_model
def tocg_cloth_encoder(pretrained=False, in_channels=4, ngf=96, **kwargs):
    return Encoder(in_channels=in_channels, ngf=ngf)


if __name__ == '__main__':
    from timm.models import create_model


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    cnn = Encoder(4, ngf=96, norm_layer=nn.BatchNorm2d)

    print(count_parameters(cnn))
    print(count_parameters(create_model('resnet50', num_classes=0)))

    # indata = torch.ones(1,4,256,192)
    # outdata = cnn(indata)
    # print(outdata.shape)