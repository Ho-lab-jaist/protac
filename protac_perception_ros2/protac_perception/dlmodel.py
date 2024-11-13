import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """Double convolution then downscaling"""

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)    
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
    """UP (upscaling) convolution then double convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2, stride=2):
        super(Decoder, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input shape is B, C, H, W
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]

        x2 = F.max_pool2d(x2, kernel_size=(diffH+1, diffW+1), stride=1, padding=0)

        # x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
        #                 diffH // 2, diffH - diffH // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Map features to the three-component (3-channel) forces"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class TacNet(nn.Module):
    def __init__(self, in_nc=6, n_classes=3, num_of_neurons=2048, bilinear=False):
        super(TacNet, self).__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = 585*3
        self.fc1 = nn.Linear(3*24*28, num_of_neurons, bias=True)
        self.fc2 = nn.Linear(num_of_neurons, num_of_neurons, bias=True)
        self.fc3 = nn.Linear(num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):   # x:   6, 256, 256
        x1 = self.inc(x)    # x1:  8, 256, 256
        # print(x1.shape)
        x2 = self.down1(x1) # x2: (16, 128, 128)
        # print(x2.shape)
        x3 = self.down2(x2) # x3: (32, 64, 64)
        # print(x3.shape)
        x4 = self.down3(x3) # x4: (64, 32, 32)
        # print(x4.shape)
        x5 = self.down4(x4) # x5: (128, 16, 16)
        # print(x5.shape)
        x6 = self.down5(x5) # x6: (256, 8, 8)
        # print(x6.shape)
        x7 = self.down6(x6) # x7: (256, 6, 7)
        # print(x7.shape)
        x = self.up1(x7, x5)# x:  (128, 12, 14) 
        # print(x.shape)
        x = self.up2(x, x4) # x:  (64, 24, 28) 
        # print(x.shape)
        force_map = self.outc(x) # force_map:  (3, 24, 28) 
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        output = force_map.view(-1, self.num_flat_features(force_map))
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ForceNet(nn.Module):
    def __init__(self, in_nc=3, n_classes=2, bilinear=False):
        super(ForceNet, self).__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear, kernel_size=1, stride=2)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):   # x:   6, 256, 256
        x1 = self.inc(x)    # x1:  8, 256, 256
        # print(x1.shape)
        x2 = self.down1(x1) # x2: (16, 128, 128)
        # print(x2.shape)
        x3 = self.down2(x2) # x3: (32, 64, 64)
        # print(x3.shape)
        x4 = self.down3(x3) # x4: (64, 32, 32)
        # print(x4.shape)
        x5 = self.down4(x4) # x5: (128, 16, 16)
        # print(x5.shape)
        x6 = self.down5(x5) # x6: (256, 8, 8)
        # print(x6.shape)
        x7 = self.down6(x6) # x7: (256, 6, 7)
        # print(x7.shape)
        x = self.up1(x7, x5)# x:  (128, 12, 14) 
        # print(x.shape)
        x = self.up2(x, x4) # x:  (64, 23, 27) 
        # print(x.shape)
        force_map = self.outc(x) # force_map:  (3, 23, 27) 
        # print(force_map.shape)
        return force_map