import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ResidualBlock(nn.Module):
    #这段代码定义了一个残差块（ResidualBlock），用于神经网络中的残差学习。它包含两个卷积层（conv1 和 conv2），
    # 每个卷积层后面跟着一个批量归一化层（bn1 和 bn2），以及一个 PReLU 激活函数。该结构常用于深度卷积网络中，以帮助缓解训练中的梯度消失问题
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)#卷积层
        self.bn1 = nn.BatchNorm2d(channels)#批量归一化层
        self.prelu = nn.PReLU()#PReLU 激活函数
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x#将输入保存下来，以便在后续步骤中与卷积后的输出相加。
        out = self.prelu(self.bn1(self.conv1(x)))#对输入 x 进行卷积、批量归一化和 PReLU 激活
        out = self.bn2(self.conv2(out))# 继续对激活后的输出进行卷积和批量归一化。
        out += residual#将输入 x（残差）加到卷积后的输出上。
        return out# 返回经过处理的输出。



class ChannelAttention(nn.Module):#ChannelAttention 类的主要作用是应用通道注意力机制来加权输入特征图。
    def __init__(self, channels, reduction=16):# 初始化 (__init__ 方法)
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()# 获取输入特征图的形状，b是批量大小，c是通道数
        y = self.avg_pool(x).view(b, c) # 对输入特征图应用全局平均池化，将其压缩为形状为 (b, c) 的张量
        y = self.fc(y).view(b, c, 1, 1) # 通过全连接层计算每个通道的注意力权重，并调整形状为 (b, c, 1, 1)
        return x * y.expand_as(x)  # 使用注意力权重对输入特征图进行加权，使每个通道的特征图得到调整，应用注意力权重：


class Generator1(nn.Module):#继承自 PyTorch 的 nn.Module
    def __init__(self):#初始化网络结构
        super(Generator1, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),  # Assuming input has a single channel
            nn.PReLU()#第一个卷积块，将输入的单通道图像通过一个 9x9 的卷积核和 PReLU 激活函数处理，输出 64 个特征图。
        )
        self.block2 = ResidualBlock(64)#残差快学习，五个残差块（ResidualBlock），每个输入和输出的通道数为 64。残差块有助于缓解深层网络的训练困难。
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )#一个卷积层后接批归一化，处理通道数为 64 的特征图，卷积核为 3x3。
        self.attention = ChannelAttention(64)#通道注意力机制，用于加权通道特征。
        self.final = nn.Conv2d(64, 1, kernel_size=9, padding=4)  # Output also has a single channel
#最后的卷积层，将处理后的特征图映射回单通道输出。
    def forward(self, x):#定义了 Generator1 类的前向传播过程
        block1 = self.block1(x)#输入 x 通过第一个卷积层 block1，该层包括一个 9x9 卷积和 PReLU 激活函数。输出是经过处理的特征图。
        block2 = self.block2(block1)#第一个残差块 block2 处理 block1 的输出，进行残差学习以捕捉更深层次的特征。
        block3 = self.block3(block2)#处理前一个块的输出，进一步提取特征。
        block4 = self.block4(block3)#继续提取和调整特征。
        block5 = self.block5(block4)#继续对特征进行处理。
        block6 = self.block6(block5)#处理到目前为止的特征，生成最终的特征图。
        block7 = self.block7(block6)#通过最后的卷积层 block7 处理残差块的输出，输出特征图经过卷积和批归一化。

        attention = self.attention(block7)#通过通道注意力机制 self.attention 处理 block7 的输出，生成注意力权重。
        out = self.final(block1 + attention)  # Skip connection from block1 to final
#使用 block1 的输出和注意力机制的加权结果进行跳跃连接（skip connection），并通过最终的卷积层 self.final 生成最终的输出。
        return (torch.tanh(out) + 1) / 2  # Ensuring output is between 0 and 1对最终输出 out 应用 tanh 激活函数，将其范围从 [-1, 1] 转换到 [0, 1]。


class UnetSkipConnectionBlock(nn.Module):# U-Net 结构中的一个子模块，实现了跳跃连接（skip connection）。
    # U-Net 是一种用于图像分割的经典神经网络结构，其中跳跃连接用于在编码器和解码器之间传递特征图，以便更好地恢复细节。
    """Defines the Unet submodule with skip connection."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(input_nc or outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True) if not innermost else nn.ReLU(True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)

# class Generator2(nn.Module):
#
#     """Create a Unet-based generator"""
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet generator."""
#         super(Generator2, self).__init__()
#         # Build Unet network from the innermost to outermost layer
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
#         for _ in range(num_downs - 5):
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
#
#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)


