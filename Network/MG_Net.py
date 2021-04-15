import torch
import torch.nn as nn


class VarAttention(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        if mode == 'all':
            self.conv1 = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        elif mode == 'var':
            self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in')

    def forward(self, x, group):
        """
        x: (batch, n*c, *)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, *)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        shape = list(x.shape)
        x = x.reshape([shape[0], group, shape[1] // group] + shape[2:])  # (batch, n, c, *)

        var_out = compute_var_map(x)  # (batch, 1, *)

        if self.mode == 'all':
            x = torch.cat([avg_out, max_out, var_out], dim=1)
        else:
            x = var_out

        x = self.conv1(x)

        return self.sigmoid(x)


def compute_var_map(x):
    """
    x: (batch, n, c, *)
    """
    x = x.var(dim=1)  # (batch, c, *)
    x = x.mean(dim=1, keepdim=True)  # (batch, 1, *)
    return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, group, mode='3d', two_conv=False, attention=False):
        super().__init__()
        self.attention = attention
        self.group = group

        if mode=='3d':
            maxpool = nn.MaxPool3d
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            maxpool = nn.MaxPool2d
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.maxpool = maxpool(kernel_size=2, stride=2)
        if two_conv:
            self.conv = nn.Sequential(conv(in_channel, out_channel, kernel_size=3, padding=1, groups=group),
                                      conv(out_channel, out_channel, kernel_size=3, padding=1, groups=group))
        else:
            self.conv = conv(in_channel, out_channel, kernel_size=3, padding=1, groups=group)
        self.bn = bn(out_channel)
        self.prelu = nn.PReLU()

        if attention:
            self.varattention = VarAttention(attention)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)

        if self.attention:
            map = self.varattention(x, self.group)
            x = map * x

        x = self.bn(x)
        x = self.prelu(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, group, mode='3d', two_conv=False, attention=False):
        super().__init__()
        self.group = group
        self.attention = attention

        if mode=='3d':
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        if two_conv:
            self.conv = nn.Sequential(conv(in_channel, out_channel, kernel_size=3, padding=1, groups=group),
                                      conv(out_channel, out_channel, kernel_size=3, padding=1, groups=group))
        else:
            self.conv = conv(in_channel, out_channel, kernel_size=3, padding=1, groups=group)
        self.bn = bn(out_channel)
        self.prelu = nn.PReLU()

        if attention:
            self.varattention = VarAttention(attention)

    def forward(self, x, y):
        x = self.upsample(x)

        x_shape = list(x.shape)  # [batch, x*cx, *]
        x = x.reshape([x_shape[0], self.group, x_shape[1]//self.group]+x_shape[2:])  # (batch, n, c, h, w, d)
        y = y.reshape([x_shape[0], self.group, y.shape[1]//self.group]+x_shape[2:])
        x = torch.cat([x, y], dim=2)  # (batch, n, cx+cy, h, w, d)
        x = x.reshape([x_shape[0], -1]+x_shape[2:])  # (batch, n*(cx+cy), h, w, d)

        x = self.conv(x)

        if self.attention:
            map = self.varattention(x, self.group)
            x = (map+1) * x

        x = self.bn(x)
        x = self.prelu(x)
        return x


class MGNet(nn.Module):
    def __init__(self, in_channel, filters, group, mode='3d', two_conv=False, attention=False):
        super().__init__()
        if mode=='3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.group = group
        if two_conv:
            self.inc = nn.Sequential(conv(in_channel, filters[0], kernel_size=3, padding=1, groups=group),
                                     bn(filters[0]),
                                     nn.PReLU(),
                                     conv(filters[0], filters[0], kernel_size=3, padding=1, groups=group),
                                     bn(filters[0]),
                                     nn.PReLU())
        else:
            self.inc = nn.Sequential(conv(in_channel, filters[0], kernel_size=3, padding=1, groups=group),
                                     bn(filters[0]),
                                     nn.PReLU())

        self.down1 = Down(filters[0], filters[1], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.down2 = Down(filters[1], filters[2], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.down3 = Down(filters[2], filters[3], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.down4 = Down(filters[3], filters[4], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.up1 = Up(filters[4]+filters[3], filters[3], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.up2 = Up(filters[3]+filters[2], filters[2], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.up3 = Up(filters[2]+filters[1], filters[1], group=group, mode=mode, two_conv=two_conv, attention=attention)
        self.up4 = Up(filters[1]+filters[0], filters[0], group=group, mode=mode, two_conv=two_conv, attention=attention)

        if two_conv:
            self.class_conv = nn.Sequential(conv(filters[0], filters[0], kernel_size=3, padding=1, groups=group),
                                            bn(filters[0]),
                                            nn.PReLU(),
                                            conv(filters[0], group, kernel_size=3, padding=1, groups=group))
        else:
            self.class_conv = conv(filters[0], group, kernel_size=3, padding=1, groups=group)

    def forward(self, x):
        """
        x: (batch, 1, h, w, d) or (batch, n*1, h, w, d) if group_in.
        """
        # TODO: check
        if x.shape[1] < self.group:
            dim = len(x.shape[2:])
            x = x.repeat([1,self.group] + [1,]*dim)

        x1 = self.inc(x)  # [batch, n*c1, h, w, d]
        x2 = self.down1(x1)  # [batch, n*c2, h, w, d]
        x3 = self.down2(x2)  # [batch, n*c3, h, w, d]
        x4 = self.down3(x3)  # [batch, n*c4, h, w, d]
        x5 = self.down4(x4)  # [batch, c5, h, w, d]
        x = self.up1(x5, x4)  # [batch, n*c4, h, w, d]
        x = self.up2(x, x3)  # [batch, n*c3, h, w, d]
        x = self.up3(x, x2)  # [batch, n*c2, h, w, d]
        x = self.up4(x, x1)  # [batch, n*c1, h, w, d]
        x = self.class_conv(x)  # [batch, n, h, w, d] before sigmoid
        x = torch.sigmoid(x)

        return x
