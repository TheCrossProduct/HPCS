import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, kernel_size=3, scale=2, bilinear=True):
        """
        Parameters
        ----------
        n_channels: int

        n_classes: int

        n_filters: int

        kernel_size: int or tuple

        scale: int or tuple

        bilinear: bool
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        if isinstance(scale, list):
            self.scale = tuple(scale)
        else:
            self.scale = scale

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(self.n_channels, self.n_filters)
        self.down1 = Down(self.n_filters, 2 * self.n_filters, max_pool=self.scale)
        self.down2 = Down(2 * self.n_filters, 4 * self.n_filters, max_pool=self.scale)
        self.down3 = Down(4 * self.n_filters, 8 * self.n_filters, max_pool=self.scale)
        self.down4 = Down(8 * self.n_filters, 16 * self.n_filters // factor, max_pool=self.scale)

        self.up1 = Up(16 * self.n_filters, (8 * self.n_filters) // factor, bilinear, scale_factor=self.scale)
        self.up2 = Up(8 * self.n_filters, (4 * self.n_filters) // factor, bilinear, scale_factor=self.scale)
        self.up3 = Up(4 * self.n_filters, (2 * self.n_filters) // factor, bilinear, scale_factor=self.scale)
        self.up4 = Up(2 * self.n_filters, self.n_filters, bilinear, scale_factor=self.scale)
        self.outc = OutConv(self.n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLu) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mid_channels is None:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels

        self.kernel_size = kernel_size

        self.double_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=2):
        super(Down, self).__init__()
        self.max_pool = nn.Sequential(nn.MaxPool2d(max_pool),
                                      DoubleConv(in_channels=in_channels, out_channels=out_channels)
                                      )

    def forward(self, x):
        return self.max_pool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=in_channels // 2, kernel_size=2, stride=scale_factor)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

