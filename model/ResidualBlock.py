import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c1, c2, c3 , num_heads):
        super().__init__()
        self.ln = nn.LayerNorm(c1)
        # self.q = nn.Conv1d(c1, c1, 1, bias=False)
        self.q = nn.Linear(c1, c1, bias=False)
        # self.k = nn.Conv1d(c1, c1, 1, bias=False)
        self.k = nn.Linear(c1, c1, bias=False)
        # self.v = nn.Conv1d(c1, c1, 1, bias=False)
        self.v = nn.Linear(c1, c1, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads)
        # self.fc1 = nn.Conv1d(c1, c2, 1, bias=False)
        self.fc1 = nn.Linear(c1, c2, bias=False)
        # self.fc2 = nn.Conv1d(c2, c1, 1, bias=False)
        self.fc2 = nn.Linear(c2, c1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(c3,c3,3,1,1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        # self.conv = nn.Sequential()
        # if c3 < 576:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(c3,c3,3,1,1),
        #         nn.BatchNorm2d(c3),
        #         nn.ReLU(inplace=True)
        #     )
        

    def forward(self, x):
        x = self.ln(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.ln(x)
        x = self.fc2(self.fc1(x)) + x
        # residual = x
        # a,b,_=x.shape
        # x = self.fc1(x).view(a,b,56,-1).permute(1,0,2,3)
        # x = self.conv(x).permute(1,0,2,3).view(a,b,-1)
        # x = self.fc2(x) + residual  
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, c3, c4, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = nn.Sequential(
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU()
            )
        # self.linear = nn.Conv1d(c2, c2, 1)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, c3, c4, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, c, w, h = x.shape
        # p = x.view(b, c, h*w)
        # e = self.linear(p)
        # x = (p + e)
        # x = self.tr(x).view(b, -1, w, h)
        p = x.view(b, c, h*w).permute(2, 0, 1)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=True,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)  #[bs, w*h, c]

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  #[bs, w*h, c]

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  #[bs, c, w*h]
        
        f = torch.matmul(theta_x, phi_x)  #[bs, w*h, w*h]

        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)#[bs, w*h, c]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class ASPPConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,dilation):
        modules=[
            nn.Conv2d(in_channels,out_channels,3,padding=dilation,dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv,self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(ASPPPooling,self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        size=x.shape[-2:]
        x=super(ASPPPooling,self).forward(x)
        return F.interpolate(x,size=size,mode='bilinear')

class ASPP(nn.Module):
    def __init__(self,in_channels=256,out_channels=256,atrous_rates=[3,5,7]) -> None:
        super(ASPP,self).__init__()
        modules=[]
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        rate1,rate2,rate3=tuple(atrous_rates)
        modules.append(ASPPConv(in_channels,out_channels,rate1))
        modules.append(ASPPConv(in_channels,out_channels,rate2))
        modules.append(ASPPConv(in_channels,out_channels,rate3))
        modules.append(ASPPPooling(in_channels,out_channels))
        self.convs=nn.ModuleList(modules)
        self.project=nn.Sequential(
            nn.Conv2d(5*out_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self,x):
        res=[]
        for conv in self.convs:
            res.append(conv(x))
        res=torch.cat(res,dim=1)
        out=self.project(res)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),#(b,c,h/stride,w/stride)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),#(b,c,h/stride,w/stride)
            nn.BatchNorm2d(out_channels),
        )
        self.downsample=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),#(b,c,h/stride,w/stride)
                nn.BatchNorm2d(out_channels),
            )
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        out=self.body(x)
        out=out+self.downsample(x)
        out=self.relu(out)
        return out

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)