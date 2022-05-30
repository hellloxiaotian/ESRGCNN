import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        
class  MFCModule(nn.Module):
    def __init__(self,in_channels,out_channels,gropus=1):
        super(MFCModule,self).__init__()
        kernel_size =3
        padding = 1
        features = 64
        features1 = 48
        distill_rate = 0.25
        self.distilled_channels = int(features*distill_rate)
        self.remaining_channels = int(features-self.distilled_channels)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features1,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self,input):
        out1_c = self.conv1_1(input)
        dit1,remain1 = torch.split(out1_c,(self.distilled_channels,self.remaining_channels),dim=1)
        out1_r = self.ReLU(remain1)
        out2_c = self.conv1_2(out1_r)
        #out2_c = out2_c + out1_c
        dit2,remain2 = torch.split(out2_c,(self.distilled_channels,self.remaining_channels),dim=1)
        remain2 = remain2+remain1
        out2_r = self.ReLU(remain2)
        out3_c = self.conv1_3(out2_r)
        #out3_c = out3_c + out2_c
        dit3,remain3 = torch.split(out3_c,(self.distilled_channels,self.remaining_channels),dim=1)
        remain3 = remain3+remain2
        out3_r = self.ReLU(remain3)
        out4_c  = self.conv1_4(out3_r)
        #out4_c = out4_c + out3_c
        dit4,remain4 = torch.split(out4_c,(self.distilled_channels,self.remaining_channels),dim=1)
        remain4 = remain4+remain3
        dit = dit1+dit2+dit3+dit4
        out_t =  torch.cat([dit,remain4],dim=1)
        #out_t =  out1_c+out2_c+out3_c+out4_c+out_t
        out_r = self.ReLU(out_t)
        out5_r = self.conv1_5(out_r)
        out6_r = self.conv1_6(out5_r)
        out6_r = input +out6_r
        return out6_r


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.b1 = MFCModule(features,features)
        self.b2 = MFCModule(features,features)
        self.b3 = MFCModule(features,features)
        self.b4 = MFCModule(features,features)
        self.b5 = MFCModule(features,features)
        self.b6 = MFCModule(features,features)
        self.ReLU=nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=    padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x1 = self.conv1_1(x)
        b1 = self.b1(x1)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        #b6 = x1+b1+b2+b3+b4+b5+b6
        #x2 = x1+b1+b2+b3+b4+b5+b6
        x2 = self.conv2(b6)
        temp = self.upsample(x2, scale=scale)
        #temp = self.upsample(b6, scale=scale)
        temp2 = self.ReLU(temp)
        out = self.conv3(temp2)
        out = self.add_mean(out)
        return out
