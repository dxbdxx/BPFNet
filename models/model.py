import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
import kornia
import math

from collections import OrderedDict
import models.utils as utils
from models.metrics import ArcMarginProduct
from models.crop_and_resize import CropAndResize


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CenterNet(nn.Module):
    def __init__(self, block, layers, head_conv):
        super(CenterNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(2, [256, 128], [4, 4])

        self.fc1 = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))

        self.fc2 = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.fc3 = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.fc4 = nn.Sequential(nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True))

        self.init_weights()
        self.fc1[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.fc2)
        self.fill_fc_weights(self.fc3)
        self.fc4[-1].bias.data.fill_(-2.19)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = nn.Conv2d(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1)

            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            self.fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        fc3 = self.fc3(x)
        fc4 = self.fc4(x)
        return utils._sigmoid(fc1), fc2, fc3, fc4, x

    def init_weights(self):
        state_dict = torch.load('./models/pretrain_models/resnet18-5c106cde.pth', map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_model(self, model):
        state_dict = torch.load(model, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name_key = k[7:]  # remove `module.`
            new_state_dict[name_key] = v

        state_dict = {k: v for k, v in new_state_dict.items() if k in self.state_dict()}
        model_state_dict = self.state_dict()
        model_state_dict.update(state_dict)

        self.load_state_dict(model_state_dict)


class ResNet18_pre(nn.Module):
    def __init__(self, pretrain=False):
        super(ResNet18_pre, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain:
            state_dict = torch.load('./models/pretrain_models/resnet18-5c106cde.pth', map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class Resnet_palm(nn.Module):
    def __init__(self, num_classes, pretrain, arc, m, s):
        super(Resnet_palm, self).__init__()
        self.arc = arc
        self.m = m
        self.s = s
        self.inplanes = 128
        self._norm_layer = nn.BatchNorm2d
        
        self.model_IR = ResNet18_pre(pretrain=pretrain)
        self.model_RGB = self._make_layer(BasicBlock, 256, 2, stride=2)

        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if arc:
            self.fc = ArcMarginProduct(512, num_classes, s, m)
        else:
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_attention(self, x1, x2):
        bs, c1, h1, w1 = x1.shape
        _, c2, h2, w2 = x2.shape

        x1_a = x1.view(bs, c1, -1)
        x2_a = x2.view(bs, c2, -1)
        x1_n = F.normalize(x1_a, dim=2, p=2)
        x2_n = F.normalize(x2_a, dim=2, p=2)

        mat = torch.matmul(x1_n, x2_n.permute(0,2,1))
        ind_mat = mat.argmax(2)
        x1_att = x2_a.gather(1, ind_mat.view(bs, c1, 1).expand(bs, c1, h1*w1))
        
        x = x1_a + x1_att
        x = x.view(bs, c1, h1, w1)

        return x

    def forward(self, x1, x2, y):
        N = int(x1.size(0)/2)
        x1 = self.model_RGB(x1)
        x2 = self.model_IR(x2)

        x = self.get_attention(x1, x2)
        x = self.layer4(x)
        x = self.avgpool(x)
        self.ft = torch.flatten(x, 1)

        if self.arc:
            self.logits = self.fc(self.ft, y)
        else:
            self.logits = self.fc(self.ft)

        return self.logits, self.ft


class PalmAlignNet(nn.Module):
    def __init__(self, config, num_classes):
        super(PalmAlignNet, self).__init__()

        self.N = config.batch_size
        ## ------------- detection ----------------
        self.model_det = CenterNet(BasicBlock, [2, 2, 2, 2], 256)
        self.crop_resize1 = CropAndResize(32, 32)
        self.crop_resize2 = CropAndResize(128, 128)

        self.ind_boxes = torch.arange(self.N).int().cuda()

        ## ------------- recognition ----------------
        self.model_cls = Resnet_palm(num_classes, True, config.arc, config.m, config.s)

    def load_model(self, model):
        state_dict = torch.load(model, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[7:] == 'module.':
                k = k[7:]  # remove `module.`
            new_state_dict[k] = v

        state_dict = {k: v for k, v in new_state_dict.items() if k in self.state_dict()}
        model_state_dict = self.state_dict()
        model_state_dict.update(state_dict)

        self.load_state_dict(model_state_dict)

    def pred_box(self, hd1, hd2, hd3, hd4, w, h):
        hd1 = utils.nms(hd1)
        inds = hd1.view(self.N, -1).max(1)[1].view(self.N, -1)
        ind_h = inds // hd1.size(3)
        ind_w = inds - ind_h * hd1.size(3)

        wh_pred = hd2.view(self.N, 2, -1).gather(2, inds.view(self.N, 1, 1).expand(self.N, 2, 1))

        disp_pred = hd3.view(self.N, 2, -1).gather(2, inds.view(self.N, 1, 1).expand(self.N, 2, 1))

        bin_pred = hd4.view(self.N, 4, -1).gather(2, inds.view(self.N, 1, 1).expand(self.N, 4, 1)).squeeze()
        class_pred = (bin_pred[:, 1] > bin_pred[:, 0]).long()
        theta_pred = bin_pred[:, 2:]
        theta_pred = theta_pred.gather(1, class_pred.view(-1, 1)).view(-1)

        bbox_pred = torch.cat((ind_w - wh_pred[:, 0] / 2, ind_h - wh_pred[:, 1] / 2, ind_w + wh_pred[:, 0] / 2,
                               ind_h + wh_pred[:, 1] / 2), dim=1)

        bbox_pred[:, [0, 2]] = torch.clamp(bbox_pred[:, [0, 2]], min=0, max=w-1)
        bbox_pred[:, [1, 3]] = torch.clamp(bbox_pred[:, [1, 3]], min=0, max=h-1)

        return wh_pred.squeeze(), disp_pred.squeeze(), bbox_pred, theta_pred, bin_pred

    def extract_ROI(self, X, img_IR, bbox_pred, theta_pred, disp_pred):
        x1, y1, x3, y3 = bbox_pred[:, 0], bbox_pred[:, 1], bbox_pred[:, 2], bbox_pred[:, 3]
        center_pred_x, center_pred_y = (x1 + x3) / 2, (y1 + y3) / 2
        center_pred = torch.cat([center_pred_x, center_pred_y]).view(self.N, 2)
        ratio_x, ratio_y = bbox_pred[:, [0, 2]] / X.size(3), bbox_pred[:, [1, 3]] / X.size(2)

        X1_r = kornia.geometry.rotate(X, theta_pred, center_pred, 'nearest')
        X2 = kornia.geometry.translate(img_IR, -4*disp_pred)
        X2_r = kornia.geometry.rotate(X2, theta_pred, center_pred, 'nearest')

        ROI1 = self.crop_resize1(X1_r, torch.stack([ratio_y[:, 0], ratio_x[:, 0], ratio_y[:, 1], ratio_x[:, 1]], dim=1), self.ind_boxes)
        ROI2 = self.crop_resize2(X2_r, torch.stack([ratio_y[:, 0], ratio_x[:, 0], ratio_y[:, 1], ratio_x[:, 1]], dim=1), self.ind_boxes)

        return ROI1, ROI2

    def forward(self, img, y):
        N = img.size(0)
        if N != self.N:
            self.N = N
            self.ind_boxes = torch.arange(self.N*2).int().cuda()

        img_RGB = img[:,0,:,:,:]
        img_IR = img[:,1,:,:,:]
        h, w = img_RGB.shape[2:4]

        hd1, hd2, hd3, hd4, X = self.model_det(img_RGB)
        wh_pred, disp_pred, bbox_pred, theta_pred, bin_pred = self.pred_box(hd1, hd2, hd3, hd4, w, h)

        ROI1, ROI2 = self.extract_ROI(X, img_IR, bbox_pred, theta_pred, disp_pred)

        if self.training:
            logits, ft = self.model_cls(ROI1, ROI2, y)
            return hd1, wh_pred, disp_pred, bin_pred, logits, ft
        else:
            logits, ft = self.model_cls(ROI1, ROI2, y)
            return bbox_pred, theta_pred, logits, ft