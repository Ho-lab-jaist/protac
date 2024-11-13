"""Module for the TacNet"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, network_input):
        return self.double_convolution(network_input)


class Encoder(nn.Module):
    """Double convolution then downscaling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_convolution = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, network_input):
        return self.maxpool_convolution(network_input)


class Decoder(nn.Module):
    """UP (upscaling) convolution then double convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.upsampling_layer = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.convolutional_layer = DoubleConvolution(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.upsampling_layer = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.convolutional_layer = DoubleConvolution(in_channels, out_channels)

    def forward(self, signal_1, signal_2):
        signal_1 = self.upsampling_layer(signal_1)
        # input shape is B, C, H, W
        height_difference = signal_2.size()[2] - signal_1.size()[2]
        width_difference = signal_2.size()[3] - signal_1.size()[3]

        signal_2 = F.max_pool2d(
            signal_2,
            kernel_size=(height_difference + 1, width_difference + 1),
            stride=1,
            padding=0,
        )

        concatenated_signals = torch.cat([signal_2, signal_1], dim=1)
        return self.convolutional_layer(concatenated_signals)


class OutConv(nn.Module):
    """Map features to the three-component (3-channel) forces"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input_features):
        return self.convolution(input_features)


class TacNet(nn.Module):
    """Class for the TacNet Model"""

    def __init__(self, in_nc = 1, 
                       num_of_features = 621, 
                       num_of_neurons = 2048,
                       n_classes = 3,  
                       bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_of_neurons = num_of_neurons

        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        # formulation of 2d convolutinon with stride=1, padding=0
        # Hout = Hin - kernel_size + 1 (= Hin - 2)
        # Wout = Win - kernel_size + 1 (= Win - 1)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = num_of_features * 3
        self.fc1 = nn.Linear(3 * 24 * 28, self.num_of_neurons, bias=True)
        # self.fc1 = nn.Linear(3 * 52 * 56, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_signal):  # x:   6, 256, 256
        encoded = self.inc(input_signal)  # x1:  8, 256, 256
        # print(x1.shape)
        encoded = self.down1(encoded)  # x2: (16, 128, 128)
        # print(x2.shape)
        encoded = self.down2(encoded)  # x3: (32, 64, 64)
        # print(x3.shape)
        encoded = self.down3(encoded)  # x4: (64, 32, 32)
        fourth_encoded = encoded
        # print(x4.shape)
        encoded = self.down4(encoded)  # x5: (128, 16, 16)
        fifth_encoded = encoded
        # print(x5.shape)
        encoded = self.down5(encoded)  # x6: (256, 8, 8)
        # print(x6.shape)
        seventh_encoded = self.down6(encoded)  # x7: (256, 6, 7)
        # print(x7.shape)
        decoded = self.up1(seventh_encoded, fifth_encoded)  # x:  (128, 12, 14)
        # print(x.shape)
        decoded = self.up2(decoded, fourth_encoded)  # x:  (64, 24, 28)
        # print(x.shape)
        force_map = self.outc(decoded)  # force_map:  (3, 24, 28)
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        num_of_fore_map_features = self.number_flat_features(force_map)
        output = force_map.view(-1, num_of_fore_map_features)
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features

class TacNetMassageArm(nn.Module):
    """Class for the TacNet Model"""

    def __init__(self, in_nc = 2, 
                       num_of_features = 585, 
                       num_of_neurons = 2048,
                       n_classes = 3,  
                       bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_of_neurons = num_of_neurons

        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        # formulation of 2d convolutinon with stride=1, padding=0
        # Hout = Hin - kernel_size + 1 (= Hin - 2)
        # Wout = Win - kernel_size + 1 (= Win - 1)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = num_of_features * 3
        # self.fc1 = nn.Linear(3 * 24 * 28, self.num_of_neurons, bias=True)
        self.fc1 = nn.Linear(3 * 48 * 52, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_signal):  # x:   6, 256, 256
        encoded = self.inc(input_signal)  # x1:  8, 256, 256
        # print(x1.shape)
        encoded = self.down1(encoded)  # x2: (16, 128, 128)
        # print(x2.shape)
        encoded = self.down2(encoded)  # x3: (32, 64, 64)
        # print(x3.shape)
        encoded = self.down3(encoded)  # x4: (64, 32, 32)
        fourth_encoded = encoded
        # print(x4.shape)
        encoded = self.down4(encoded)  # x5: (128, 16, 16)
        fifth_encoded = encoded
        # print(x5.shape)
        encoded = self.down5(encoded)  # x6: (256, 8, 8)
        # print(x6.shape)
        seventh_encoded = self.down6(encoded)  # x7: (256, 6, 7)
        # print(x7.shape)
        decoded = self.up1(seventh_encoded, fifth_encoded)  # x:  (128, 12, 14)
        # print(x.shape)
        decoded = self.up2(decoded, fourth_encoded)  # x:  (64, 24, 28)
        # print(x.shape)
        force_map = self.outc(decoded)  # force_map:  (3, 24, 28)
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        num_of_fore_map_features = self.number_flat_features(force_map)
        output = force_map.view(-1, num_of_fore_map_features)
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features
    
class TacNetProTac(nn.Module):
    """Class for the TacNet Model"""

    def __init__(self, in_nc = 1, 
                       num_of_features = 621, 
                       num_of_neurons = 2048,
                       n_classes = 3,  
                       bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_of_neurons = num_of_neurons

        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        # formulation of 2d convolutinon with stride=1, padding=0
        # Hout = Hin - kernel_size + 1 (= Hin - 2)
        # Wout = Win - kernel_size + 1 (= Win - 1)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = num_of_features * 3
        # self.fc1 = nn.Linear(3 * 24 * 28, self.num_of_neurons, bias=True)
        self.fc1 = nn.Linear(3 * 48 * 52, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_signal):  # x:   1, 480, 480
        encoded = self.inc(input_signal)  # x1:  8, 480, 480
        # print(x1.shape)
        encoded = self.down1(encoded)  # x2: (16, 240, 240)
        # print(x2.shape)
        encoded = self.down2(encoded)  # x3: (32, 120, 120)
        # print(x3.shape)
        encoded = self.down3(encoded)  # x4: (64, 60, 60)
        fourth_encoded = encoded
        # print(x4.shape)
        encoded = self.down4(encoded)  # x5: (128, 30, 30)
        fifth_encoded = encoded
        # print(x5.shape)
        encoded = self.down5(encoded)  # x6: (256, 15, 15)
        # print(x6.shape)
        seventh_encoded = self.down6(encoded)  # x7: (256, 12, 13)
        # print(x7.shape)
        decoded = self.up1(seventh_encoded, fifth_encoded)  # x:  (128, 24, 26)
        # print(decoded.shape)
        decoded = self.up2(decoded, fourth_encoded)  # x:  (64, 48, 52)
        # print(decoded.shape)
        force_map = self.outc(decoded)  # force_map:  (3, 48, 52)
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        num_of_fore_map_features = self.number_flat_features(force_map)
        output = force_map.view(-1, num_of_fore_map_features)
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features
    
# Define the STN module
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # Adjusted output size for 2D affine transformation
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 60)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)

        return x_transformed
    
class TacNetSTN(nn.Module):
    """Class for the TacNet Model"""

    def __init__(self, in_nc = 1, 
                       num_of_features = 621, 
                       num_of_neurons = 2048,
                       n_classes = 3,  
                       bilinear=False):
        super().__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_of_neurons = num_of_neurons

        self.inc = DoubleConvolution(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        # formulation of 2d convolutinon with stride=1, padding=0
        # Hout = Hin - kernel_size + 1 (= Hin - 2)
        # Wout = Win - kernel_size + 1 (= Win - 1)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = num_of_features * 3
        self.fc1 = nn.Linear(3 * 24 * 28, self.num_of_neurons, bias=True)
        self.fc2 = nn.Linear(self.num_of_neurons, self.num_of_neurons, bias=True)
        self.fc3 = nn.Linear(self.num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # STN module
        self.stn = STN()

    def forward(self, input_signal):  # x:   6, 256, 256
        # Apply STN module to input image
        tf_input_signal = self.stn(input_signal)
        encoded = self.inc(tf_input_signal)  # x1:  8, 256, 256
        # print(x1.shape)
        encoded = self.down1(encoded)  # x2: (16, 128, 128)
        # print(x2.shape)
        encoded = self.down2(encoded)  # x3: (32, 64, 64)
        # print(x3.shape)
        encoded = self.down3(encoded)  # x4: (64, 32, 32)
        fourth_encoded = encoded
        # print(x4.shape)
        encoded = self.down4(encoded)  # x5: (128, 16, 16)
        fifth_encoded = encoded
        # print(x5.shape)
        encoded = self.down5(encoded)  # x6: (256, 8, 8)
        # print(x6.shape)
        seventh_encoded = self.down6(encoded)  # x7: (256, 6, 7)
        # print(x7.shape)
        decoded = self.up1(seventh_encoded, fifth_encoded)  # x:  (128, 12, 14)
        # print(x.shape)
        decoded = self.up2(decoded, fourth_encoded)  # x:  (64, 24, 28)
        # print(x.shape)
        force_map = self.outc(decoded)  # force_map:  (3, 24, 28)
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        num_of_fore_map_features = self.number_flat_features(force_map)
        output = force_map.view(-1, num_of_fore_map_features)
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)
        return output

    def number_flat_features(self, features):
        feature_size = features.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for size in feature_size:
            num_features *= size
        return num_features