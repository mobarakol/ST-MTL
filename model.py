'''
@author: Mobarakol Islam (mobarakol@u.nus.edu)
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
import torch.nn.functional as F

# Source: https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/models/common.py
class ActivatedBatchNorm(nn.Module):
    def __init__(self, num_features, activation='relu', slope=0.01, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        net_se_ratio = 16
        cmpe_se_ratio = 16
        self.reimage_k = 16

        self.bn1 = nn.BatchNorm2d(num_features=channel)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.Multi_Map = nn.Conv2d(1, int(channel / cmpe_se_ratio), kernel_size=3,
                                   stride=1, padding=0, bias=False)
        self.inn = (2 * self.reimage_k - 2) * (int(channel / cmpe_se_ratio) - 2)
        self.net_SE = nn.Sequential(
            nn.Linear((2 * self.reimage_k - 2) * (int(channel / cmpe_se_ratio) - 2), int(channel / net_se_ratio),
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / net_se_ratio), channel, bias=False),
            nn.Sigmoid())

        self.bn1_s = nn.BatchNorm2d(num_features=channel)
        self.conv1_s = nn.Conv2d(channel, channel, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn2_s = nn.BatchNorm2d(num_features=channel)
        self.conv2_s = nn.Conv2d(channel, channel, kernel_size=3,
                                 stride=1, padding=1, bias=False)

        self.spatial_se = nn.Conv2d(2 * channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        residual = x
        x_s = x

        x = self.bn1(x)
        x = F.relu(x)
        conv_out = self.conv1(x)

        conv_out = self.bn2(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2(conv_out)

        se_ip_conv = self.avg_pool(conv_out)
        se_ip_skip = self.avg_pool(residual)

        conv_x_concat = torch.cat([se_ip_conv, se_ip_skip], dim=-1).transpose(1, 2)  # .transpose(0, 3)
        conv_copy = conv_x_concat.clone()
        conv_x_concat_out = conv_x_concat.view(conv_copy.size()[0], conv_copy.size()[1],
                                               int(conv_copy.size()[3] * self.reimage_k),
                                               int(conv_copy.size()[2] / self.reimage_k))
        multi_map = self.Multi_Map(conv_x_concat_out)
        out_map = torch.mean(multi_map, dim=1).unsqueeze(1)
        out_map = out_map.view(out_map.size()[0], -1)
        se_out = self.net_SE(out_map)
        se_out = se_out.view(se_out.size()[0], se_out.size()[1], 1, 1)
        comp_chn_se = torch.mul(conv_out, se_out)

        x = self.bn1_s(x_s)
        x = F.relu(x)
        conv_out = self.conv1_s(x)

        conv_out = self.bn2_s(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2_s(conv_out)

        conv_s_concat = torch.cat([conv_out, residual], dim=1)
        spa_se = torch.sigmoid(self.spatial_se(conv_s_concat))
        comp_spa_se = torch.mul(conv_out, spa_se)

        net_comp = residual + comp_spa_se + comp_chn_se

        return net_comp


class DecoderSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.ABN = ActivatedBatchNorm(middle_channels)
        self.SCSEBlock = SCSEBlock(middle_channels)
        self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, *args):
        x = torch.cat(args, 1)
        x = self.conv1(x)
        x = self.ABN(x)
        x = self.SCSEBlock(x)
        x = self.deconv(x)
        return x



class Shared_Encoder(nn.Module):
    def __init__(self):
        super(Shared_Encoder, self).__init__()

        base = resnet.resnet18(pretrained=True)
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

    def forward(self, x):
        # Initial block
        x = self.in_block(x)  # 64

        # Encoder blocks
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        return (x, e1, e2, e3, e4)


class Segmentation_Decoder(nn.Module):
    def __init__(self, num_classes=8):

        super(Segmentation_Decoder, self).__init__()
	
	# Segmentation Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 3, 1, 1)
        self.lsm = nn.LogSoftmax(dim=1)

        # Segmentation
        self.pool = nn.MaxPool2d(2, 2)
        self.center = DecoderSCSE(512, 256, 256)
        self.decoder5 = DecoderSCSE(768, 512, 256)
        self.decoder4 = DecoderSCSE(512, 256, 128)
        self.decoder3 = DecoderSCSE(256, 128, 64)
        self.decoder2 = DecoderSCSE(128, 64, 64)
        self.decoder1 = DecoderSCSE(128, 64, 64)

    def forward(self, encoder):
        x, e1, e2, e3, e4 = encoder[0], encoder[1], encoder[2], encoder[3], encoder[4]

        # Seg Decoder blocks
        c = self.center(self.pool(e4))
        c = F.upsample(c, e4.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.decoder5(c, e4)
        d3 = self.decoder4(d4, e3)        
        d2 = self.decoder3(d3, e2)
        d2 = F.upsample(d2, e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder2(d2, e1)
        x = F.upsample(x, d1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1, x)

        # Seg Classifier
        y = self.conv2(d1)
        y_seg = self.tp_conv2(y)
        #y_seg = self.lsm(y)
        return y_seg




class ST_MTL_SEG(nn.Module):
    def __init__(self, num_classes=8):
        super(ST_MTL_SEG, self).__init__()
        self.shared_encoder = Shared_Encoder()
        # segmentation decoder
        self.seg_decoder = Segmentation_Decoder(num_classes=num_classes)

    def forward(self, x):
        x, e1, e2, e3, e4 = self.shared_encoder(x)
        y_seg = self.seg_decoder([x, e1, e2, e3, e4])

        return y_seg











