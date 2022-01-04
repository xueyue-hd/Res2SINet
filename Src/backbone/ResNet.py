import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    """
    Bottleneck 有两种模块，记为：BTNK1 和 BTNK2.
    两种模块都包含：CONV1+BN+ReLU、CONV3+BN+ReLU、CONV1+BN 和 ReLU.
    区别在于：
        BTNK1：inplanes != planes, 因此 它多了一个CONV1+BN，即downsample模块.
        BTNK2: inplanes = planes, 没有downsample模块.
    downsample的作用：将x的维度修改成与经过公有阶段后的out的维度一样。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 自定义卷积层
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果 输入和输出通道数不同，residual(即x)无法与out直接相加，需要先做下采样，修改维度至两者相同。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_2Branch(nn.Module):
    # ResNet50 with two branches (modified from torchvision.models.resnet (pytorch==0.4.1))
    """
    构建ResNet50模型的5个 stage 从0到4.
    除了 stage0 以外，其余 stage 的模块都由 Bottleneck 搭建。
    其余 stage 的1号模块都为 Bottleneck 的 BTNK1 类型，其余模块为 BTNK2.
    """
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64
        # stage0 inplanes=64；stage1 inplanes=64-->256;stage2 inplanes=256-->512;stage3_1 inplanes=512-->1024
        super(ResNet_2Branch, self).__init__()
        # 定义 ResNet Stage0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 定义 Stage0 完毕

        # 开始定义 stage1-4
        self.layer1 = self._make_layer(Bottleneck, 64, 3)   # 定义 Stage1
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 回归stage2刚结束时的inplanes.
        self.inplanes = 512
        self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)
        # 定义 stage1-4 完毕

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: Bottleneck
        :param planes: 通道数
        :param blocks: Bottleneck 的数量
        :param stride: 默认为1
        :return:
        """
        downsample = None
        # 当输入和输出通道数相同或者不同时，Bottleneck 结构分别不同。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # block = Bottleneck，Bottleneck(self.inplanes, planes, stride, downsample)
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 设置输出通道数=输入通道数 用于之后构建 BTNK2 模块.
        self.inplanes = planes * block.expansion
        # blocks 为  Bottleneck个数，循环用于构造每个 stage 的除1号模块(即BTNK1)以外的其他模块.
        # i = 1 ... blocks-1.
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # stage0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # stage1
        x = self.layer1(x)
        # stage2
        x = self.layer2(x)
        # stage3
        x1 = self.layer3_1(x)
        # stage4
        x1 = self.layer4_1(x1)

        x2 = self.layer3_2(x)
        x2 = self.layer4_2(x2)

        return x1, x2
