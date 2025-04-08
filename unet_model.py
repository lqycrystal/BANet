
import torch.nn.functional as F

from unet_parts import *

class Pred_Layer(nn.Module):
    def __init__(self, in_c=32):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x

class BAM(nn.Module):
    def __init__(self, in_c):
        super(BAM, self).__init__()
        self.reducea = nn.Conv2d(in_c*2, in_c, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.bf_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(in_c)

    def forward(self, feat, pred):
        
        #feat = self.reducea(feat)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_feat = torch.cat((ff_feat, bf_feat), 1)
        new_feat = self.reducea(new_feat)
        new_pred = self.rgbd_pred_layer(new_feat)
        return new_feat, new_pred




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.rgbd_global = Pred_Layer(512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.bams = nn.ModuleList([
            BAM(64),
            BAM(128),
            BAM(256),
            BAM(512),
        ])
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid=nn.Sigmoid()
    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y
    def forward(self, x):
        [_, _, H, W] = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        rgbd_pred = self.rgbd_global(x5)
        preds=[]
        preds.append(torch.sigmoid(F.interpolate(rgbd_pred,size=(H, W),mode='bilinear',align_corners=True)))
        p = rgbd_pred
        x4, p = self.bams[3](x4, p)
        preds.append(torch.sigmoid(F.interpolate(p,size=(H, W),mode='bilinear',align_corners=True)))
        x = self.up1(x5, x4)
        x3,p = self.bams[2](x3, p)
        preds.append(torch.sigmoid(F.interpolate(p,size=(H, W),mode='bilinear',align_corners=True)))        
        x = self.up2(x, x3)
        x2,p = self.bams[1](x2, p)
        preds.append(torch.sigmoid(F.interpolate(p,size=(H, W),mode='bilinear',align_corners=True)))
        x = self.up3(x, x2)
        x1,p = self.bams[0](x1, p)
        preds.append(torch.sigmoid(F.interpolate(p,size=(H, W),mode='bilinear',align_corners=True)))
        x= self.up4(x, x1)
        x = self.outc(x)
        x = self.softmax(x)
        #x=F.relu(x)
        #x=self.sigmoid(x)
        preds.append(x)
        return preds