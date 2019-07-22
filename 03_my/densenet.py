#%%
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from torchsummary import summary

#%%
class BaseModule(nn.Module):
    def __init__(self, filter_size, inf, outf, padding, bias = False, stride = 1):
        super(BaseModule, self).__init__()
        self.bn = nn.BatchNorm2d(inf)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels = inf,
            out_channels = outf,
            kernel_size = filter_size,
            stride = stride,
            padding = padding,
            bias = bias
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        out = self.conv(x)

        return out

class TransitionLayer(nn.Module):
    def __init__(self, inf, outf, padding, filter_size = 1, bias = False, stride = 1):
        super(TransitionLayer, self).__init__()
        self.conv = BaseModule(1, inf, outf, padding, bias = bias, stride = 1)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv(x)
        out = self.pool(x)

        return out

class BottleNeckLayer(nn.Module):
    def __init__(self, inf, growth_rate):
        super(BottleNeckLayer, self).__init__()
        self.base1x1 = BaseModule(1, inf, growth_rate*4, padding = 0, bias = False, stride = 1)
        self.base3x3 = BaseModule(3, growth_rate*4, growth_rate, padding = 1, bias = False, stride = 1)
    
    def forward(self, x):
        out = self.base1x1(x)
        out = self.base3x3(out)
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Sequential):
    def __init__(self, inf, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        for i in range(n_layers):
            nin_per_layer = inf + growth_rate*i
            self.add_module('bottleneck_{}'.format(i+1), BottleNeckLayer(nin_per_layer, growth_rate))
        
        def forward(self, x):
            out = super(DenseBlock, self).forward(x)
            return out

class DenseNet(nn.Module):
    def __init__(self, init_channel, growth_rate, layer_sequence):
        super(DenseNet, self).__init__()
        self.init_conv = nn.Conv2d(in_channels = init_channel, out_channels = growth_rate*2, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.init_bn = nn.BatchNorm2d(growth_rate*2)
        self.relu = nn.ReLU()
        self.init_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.growth_rate = growth_rate
        self.n_blocks = len(layer_sequence)
        
        dense_sequence = []
        nin_block = growth_rate*2

        for i, n_l in enumerate(layer_sequence):
            dense_sequence.append(DenseBlock(inf = nin_block, growth_rate = self.growth_rate, n_layers = n_l))
            nin_block = nin_block + self.growth_rate * n_l
            if i + 1 != self.n_blocks:
                dense_sequence.append(TransitionLayer(inf = nin_block, outf = nin_block // 2, padding = 0, filter_size = 1, bias = False, stride = 1))
                nin_block = nin_block // 2
            #print(nin_block)
        
        self.blocks = nn.Sequential(*dense_sequence)
        self.avgpool = nn.AvgPool2d(kernel_size = 7)
        self.fc = nn.Linear(nin_block, 1000)

    def forward(self, x):
        print(x.shape)
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.relu(out)
        out = self.init_pool(out)
        out = self.blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

#%%
mymodel= DenseNet(3, 32, [6, 12, 24, 16])
mymodel.cuda()
summary(mymodel, (3, 224, 224), batch_size = 1)
#%%
nin_block = 64
growth_rate = 32
for i, n_l in enumerate([6, 12, 24, 16]):
    nin_block = nin_block + growth_rate * n_l
    print(nin_block)
    if i < 3:
        nin_block = nin_block // 2
        print(nin_block)
#%%
summary(torchvision.models.densenet121().cuda(), (3, 224, 224))