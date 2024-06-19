# hard-coded implementation from BTE ICCV2021 paper
# for the oracle model


import torch
import torch.nn as nn


class Resnet50_128(nn.Module):
    def __init__(self):
        super(Resnet50_128, self).__init__()
        self.meta = {
            "mean": [131.0912, 103.8827, 91.4953],
            "std": [1, 1, 1],
            "imageSize": [224, 224, 3],
        }
        self.conv1_7x7_s2 = nn.Conv2d(
            3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False
        )
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv1_relu_7x7_s2 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=(0, 0),
            dilation=1,
            ceil_mode=True,
        )
        self.conv2_1_1x1_reduce = nn.Conv2d(
            64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_1_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv2_1_3x3 = nn.Conv2d(
            64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2_1_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_1_3x3_relu = nn.ReLU(inplace=True)
        self.conv2_1_1x1_increase = nn.Conv2d(
            64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_1_1x1_proj = nn.Conv2d(
            64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_1_relu = nn.ReLU(inplace=True)
        self.conv2_2_1x1_reduce = nn.Conv2d(
            256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_2_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv2_2_3x3 = nn.Conv2d(
            64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2_2_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_2_3x3_relu = nn.ReLU(inplace=True)
        self.conv2_2_1x1_increase = nn.Conv2d(
            64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_2_relu = nn.ReLU(inplace=True)
        self.conv2_3_1x1_reduce = nn.Conv2d(
            256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_3_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv2_3_3x3 = nn.Conv2d(
            64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2_3_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_3_3x3_relu = nn.ReLU(inplace=True)
        self.conv2_3_1x1_increase = nn.Conv2d(
            64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv2_3_relu = nn.ReLU(inplace=True)
        self.conv3_1_1x1_reduce = nn.Conv2d(
            256, 128, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_1_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv3_1_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_1_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_1_3x3_relu = nn.ReLU(inplace=True)
        self.conv3_1_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_1_1x1_proj = nn.Conv2d(
            256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_1_relu = nn.ReLU(inplace=True)
        self.conv3_2_1x1_reduce = nn.Conv2d(
            512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_2_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv3_2_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_2_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_2_3x3_relu = nn.ReLU(inplace=True)
        self.conv3_2_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_2_relu = nn.ReLU(inplace=True)
        self.conv3_3_1x1_reduce = nn.Conv2d(
            512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv3_3_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_3_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_3x3_relu = nn.ReLU(inplace=True)
        self.conv3_3_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_3_relu = nn.ReLU(inplace=True)
        self.conv3_4_1x1_reduce = nn.Conv2d(
            512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv3_4_3x3 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv3_4_3x3_bn = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_3x3_relu = nn.ReLU(inplace=True)
        self.conv3_4_1x1_increase = nn.Conv2d(
            128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv3_4_relu = nn.ReLU(inplace=True)
        self.conv4_1_1x1_reduce = nn.Conv2d(
            512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_1_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_1_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_1_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_1x1_proj = nn.Conv2d(
            512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_1_relu = nn.ReLU(inplace=True)
        self.conv4_2_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_2_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_2_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_2_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_2_relu = nn.ReLU(inplace=True)
        self.conv4_3_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_3_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_3_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_3_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_3_relu = nn.ReLU(inplace=True)
        self.conv4_4_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_4_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_4_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_4_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_4_relu = nn.ReLU(inplace=True)
        self.conv4_5_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_5_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_5_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_5_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_5_relu = nn.ReLU(inplace=True)
        self.conv4_6_1x1_reduce = nn.Conv2d(
            1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv4_6_3x3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv4_6_3x3_bn = nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_3x3_relu = nn.ReLU(inplace=True)
        self.conv4_6_1x1_increase = nn.Conv2d(
            256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv4_6_relu = nn.ReLU(inplace=True)
        self.conv5_1_1x1_reduce = nn.Conv2d(
            1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv5_1_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_1_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_3x3_relu = nn.ReLU(inplace=True)
        self.conv5_1_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_1x1_proj = nn.Conv2d(
            1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False
        )
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_1_relu = nn.ReLU(inplace=True)
        self.conv5_2_1x1_reduce = nn.Conv2d(
            2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv5_2_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_2_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_3x3_relu = nn.ReLU(inplace=True)
        self.conv5_2_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_2_relu = nn.ReLU(inplace=True)
        self.conv5_3_1x1_reduce = nn.Conv2d(
            2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_1x1_reduce_relu = nn.ReLU(inplace=True)
        self.conv5_3_3x3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv5_3_3x3_bn = nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_3x3_relu = nn.ReLU(inplace=True)
        self.conv5_3_1x1_increase = nn.Conv2d(
            512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(
            2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.conv5_3_relu = nn.ReLU(inplace=True)
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.feat_extract = nn.Conv2d(
            2048, 128, kernel_size=[1, 1], stride=(1, 1), bias=False
        )
        self.output_size = 128

    def forward(self, data, return_preflatten=False):
        data = (data + 1) / 2
        data = data[:, [2, 1, 0], ...]
        data -= (
            torch.FloatTensor([131.0912, 103.8827, 91.4953])
            .to(data.device)
            .view(1, -1, 1, 1)
            / 255.0
        )
        data = torch.nn.functional.interpolate(
            data,
            size=(224, 224),
            scale_factor=None,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=None,
        )
        conv1_7x7_s2 = self.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.pool5_7x7_s1(conv5_3x)
        feat_extract_preflatten = self.feat_extract(pool5_7x7_s1)
        feat_extract = feat_extract_preflatten.view(feat_extract_preflatten.size(0), -1)
        if return_preflatten:
            return feat_extract, pool5_7x7_s1
        else:
            return feat_extract
        # return self.classifier(feat_extract)


class Oracle(nn.Module):
    def __init__(self, feat_extract, classifier):
        super().__init__()
        self.feat_extract = feat_extract
        self.classifier = classifier

    def forward(self, x):
        """
        Uses as input images on the range [-1, 1]
        """
        f = self.feat_extract(x)
        return f, self.classifier(f)


# ===============================================================
# class for computing the oracle FVA and MNAC
# ===============================================================


class OracleMetrics(nn.Module):
    def __init__(self, weights_path, device):
        super().__init__()

        feat_extract = Resnet50_128()
        classifier = nn.Linear(feat_extract.output_size, 40)

        sd = torch.load(weights_path, map_location="cpu")
        feat_extract.load_state_dict(sd["feat_extract"])
        classifier.load_state_dict(sd["classifier"])

        self.device = device
        self.oracle = Oracle(feat_extract, classifier).to(device)
        self.oracle.eval()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, im, cf):
        # renormalize from [-1, 1] to [0, 1]
        im = (im + 1) / 2
        cf = (cf + 1) / 2

        f_i, p_i = self.oracle(im)
        f_c, p_c = self.oracle(cf)

        p_i = torch.round(torch.sigmoid(p_i))
        p_c = torch.round(torch.sigmoid(p_c))

        FVA = 1 - self.cosine_similarity(f_i, f_c)
        FVA = FVA < 0.5
        MNAC = (p_i != p_c).float().sum(dim=1)
        return FVA, MNAC, p_i, p_c

    @torch.no_grad()
    def compute_metrics(self, im, cf):
        return self(im, cf)
