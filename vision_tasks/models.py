import copy, math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class CenteredModel(nn.Module):
    def __init__(self, model_class, *args, **kwargs):
        super(CenteredModel, self).__init__()
        self.model1 = model_class(*args, **kwargs)
        self.model2 = copy.deepcopy(self.model1)
        
        # Freeze all parameters in model2
        for param in self.model2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model1(x) - self.model2(x)

class FinalLinear(nn.Module):
    ''' The final layer readout. 
    This, together with upscaling the learning rate with N,
    is the key difference between muP and SP. This should always be
    the last layer in any architecture that uses muP.'''
    def __init__(self, in_features, out_features, bias=True):
        super(FinalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)/x.shape[-1]

class StdLinear(nn.Module):
    '''Same as a normal linear layer but with weights and biases both initialized as N(0, 1),
    with an explicity 1/sqrt(C) scaling factor in the forward pass'''
    def __init__(self, in_features, out_features, bias=True):
        super(StdLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)/math.sqrt(x.shape[-1])

class StdConv2d(nn.Module):
    '''Same as a normal conv layer but with weights and biases both initialized as N(0, 1),
    with an explicity 1/sqrt(C) scaling factor in the forward pass'''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(StdConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init.normal_(self.conv.weight, 0, 1)
        if bias:
            init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)/math.sqrt(x.shape[-3])

class MLP(nn.Module):
    def __init__(self, D, width=32, depth=3, gamma0=1.0, num_classes=10, bias=True):
        super(MLP, self).__init__()
        self.gamma0 = gamma0
        layers = [StdLinear(D, width, bias=bias), nn.ReLU()]
        for _ in range(depth-2):
            layers.extend([StdLinear(width, width, bias=bias), nn.ReLU()])
        layers.append(FinalLinear(width, num_classes, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.net(x)
        output = x/self.gamma0 
        return output

class CNN(nn.Module):
    def __init__(self, H, W, C,  width=32, gamma0=1.0, num_classes=10, bias=True):
        super(CNN, self).__init__()
        self.gamma0 = gamma0
        self.conv1 = StdConv2d(C, width, 3, 1, bias=bias)
        self.conv2 = StdConv2d(width, 2*width, 3, 1, bias=bias)
        self.fc1 = StdLinear(((H)//4-2)*((W)//4-2)*2*width, 8*width, bias=bias)
        self.fc2 = FinalLinear(8*width, num_classes, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x/self.gamma0
        return output

# ResNet, adapted from 
# github.com/microsoft/mup/
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = StdConv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StdConv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StdConv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = StdConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StdConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = StdConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StdConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, gamma0=1.0, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()
        self.gamma0 = gamma0
        base_widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in base_widths]

        self.in_planes = widths[0]
        self.conv1 = StdConv2d(3, self.in_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = FinalLinear(feat_scale*widths[3]*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), -1)
        output = self.linear(x)/self.gamma0

        return output
    
def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

