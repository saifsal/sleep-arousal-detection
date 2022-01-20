from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=13, n_classes=1):
        super().__init__()

        self.inc = DoubleConv(n_channels, 15)
        self.down1 = Down(15, 18, size=2)
        self.down2 = Down(18, 21)
        self.down3 = Down(21, 25)
        self.down4 = Down(25, 30)
        self.down5 = Down(30, 60)
        self.down6 = Down(60, 120)
        self.down7 = Down(120, 240)
        self.down8 = Down(240, 480)
        self.up1 = Up(480, 240)
        self.up2 = Up(240, 120)
        self.up3 = Up(120, 60)
        self.up4 = Up(60, 30)
        self.up5 = Up(30, 25)
        self.up6 = Up(25, 21)
        self.up7 = Up(21, 18)
        self.up8 = Up(18, 15, setting=2)
        self.outc = OutConv(15, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        output = self.outc(x)
        return output
