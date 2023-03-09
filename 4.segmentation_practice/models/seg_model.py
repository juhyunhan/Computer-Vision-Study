import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from .modules import ConvBlock

class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [64,128,256,512,512] #1/4 16 32 64
        self.num_class = 31
        
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.avgpool = nn.Identity() #똑같이 나옴, weight가 없음
        self.resnet18.fc = nn.Identity()
        self.conv32_64 = ConvBlock(in_channels=self.channels[3], out_channels=self.channels[4],kernel_size=3,stride=2,padding=1) #resnet 은 64까지 안해줘서 직접 해주는 것
        #3 2 1 하면 반으로 줄어든다 
        self.conv64_1 = ConvBlock(in_channels=self.channels[4], out_channels=self.channels[4],kernel_size=3,stride=1,padding=1)
        self.conv_cat_32 = ConvBlock(in_channels=self.channels[3]*2, out_channels=self.channels[3], kernel_size=1, stride=1)
        self.conv_cat_16 = ConvBlock(in_channels=self.channels[2]*2, out_channels=self.channels[2], kernel_size=1, stride=1)
        self.conv_cat_8 = ConvBlock(in_channels=self.channels[1]*2, out_channels=self.channels[1], kernel_size=1, stride=1)
        self.conv_cat_4 = ConvBlock(in_channels=self.channels[0]*2, out_channels=self.channels[0], kernel_size=1, stride=1)
        
        self.conv32_1 = ConvBlock(in_channels=self.channels[3], out_channels=self.channels[2], kernel_size=3, stride=1,padding=1)
        self.conv16_1 = ConvBlock(in_channels=self.channels[2], out_channels=self.channels[1], kernel_size=3, stride=1,padding=1)
        self.conv8_1 = ConvBlock(in_channels=self.channels[1], out_channels=self.channels[0], kernel_size=3, stride=1,padding=1)
        
        self.conv4_1 = ConvBlock(in_channels=self.channels[0], out_channels=self.channels[0] // 2, kernel_size=3, stride=1,padding=1)
        self.conv4_2 = ConvBlock(in_channels=self.channels[0] //2, out_channels=self.channels[0] // 2, kernel_size=3, stride=1,padding=1)
        
        self.pred = nn.Conv2d(in_channels=self.channels[0] //2, out_channels=self.channels[0] // 2, kernel_size=3, stride=1,padding=1)
        
        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        
        x = self.resnet18.maxpool(x)
        
        feats4 = self.resnet18.layer1(x)
        feats8 = self.resnet18.layer2(feats4)
        feats16 = self.resnet18.layer3(feats8)
        feats32 = self.resnet18.layer4(feats16)
        feats64 = self.conv32_64(feats32)
        
        
        #! --- 64 --- #!
        feats64 = self.conv64_1(feats64)
        feat64_32 = F.interpolate(feats64, feats32.shape[2:], mode='bilinear', align_corners = True) #사이즈를 바꿔주는 것 #bchw / #b 512 h/32 w/32
        
        #! --- 32 --- #!
        cat_64_32 = torch.cat([feats32, feat64_32], dim = 1 ) #b 1024, h/32, w/32
        cat_32 = self.conv_cat_32(cat_64_32) #b, 512, h/32, w/32
        feat_32 =  self.conv32_1(cat_32) #256 h/32 w/32
        feat_32_16 = F.interpolate(feat_32, feats16.shape[2:], mode='bilinear', align_corners = True) #b 256 h/16 w/16
        
        #! --- 16 --- #!
        cat_32_16 = torch.cat([feats16, feat_32_16], dim = 1 ) # 512 h/16 w/16
        cat_16 = self.conv_cat_16(cat_32_16) #256 h/16 w/16
        feat_16 = self.conv16_1(cat_16) #128 h/16
        feat_16_8 = F.interpolate(feat_16, feats8.shape[2:], mode='bilinear', align_corners = True) #128
        
        #! --- 8 --- #!
        cat_16_8 = torch.cat([feats8, feat_16_8], dim = 1 ) #256
        cat_8 = self.conv_cat_8(cat_16_8) #128
        feat_8 = self.conv8_1(cat_8) #64
        feat_8_4 = F.interpolate(feat_8, feats4.shape[2:], mode='bilinear', align_corners = True) #64
        
        #! --- 4 --- #!
        cat_8_4 = torch.cat([feats4, feat_8_4], dim = 1 ) #128
        cat_4 = self.conv_cat_4(cat_8_4)
        feat_4 = self.conv4_1(cat_4)
        feat_4 = self.conv4_2(feat_4)
        
        #! --- pred --- #!
        logit = self.pred(feat_4) #1/4 pred
        
        return logit
        