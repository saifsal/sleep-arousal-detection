import torch
import numpy as np

from unet.unet0 import get_unet
from unet.unet_model import UNet

kmodel = get_unet()
kmodel.load_weights("weights_01.h5")

pmodel = UNet(5, 1)


def fill_sconv(sconv, klayers):
    if len(klayers) == 2:
        # Conv1d
        sconv[0].weight.data = torch.from_numpy(
            np.transpose(klayers[0].get_weights()[0], (2, 1, 0))
        )
        sconv[0].bias.data = torch.from_numpy(klayers[0].get_weights()[1])
        # Batchnorm1d
        sconv[1].weight.data = torch.from_numpy(klayers[1].get_weights()[0])
        sconv[1].bias.data = torch.from_numpy(klayers[1].get_weights()[1])
        sconv[1].running_mean.data = torch.from_numpy(
            klayers[1].get_weights()[2])
        sconv[1].running_var.data = torch.from_numpy(
            klayers[1].get_weights()[3])
    else:
        raise ValueError("Wrong size klayers")


def fill_dconv(dconv, klayers):
    if len(klayers) == 4:
        fill_sconv(dconv[0:3:2], klayers[:2])
        fill_sconv(dconv[3:6:2], klayers[2:4])
    else:
        raise ValueError("Wrong size klayers")


def fill_down(down, klayers):
    if len(klayers) == 4:
        fill_dconv(down.maxpool_conv[1].double_conv, klayers)
    else:
        raise ValueError("Wrong size klayers")


def fill_up(up, klayers):
    if len(klayers) == 8:
        up.up.weight.data = torch.from_numpy(
            np.transpose(klayers[1].get_weights()[0].squeeze(), (2, 1, 0))
        )
        up.up.bias.data = torch.from_numpy(klayers[1].get_weights()[1])
        fill_dconv(up.conv.double_conv, klayers[4:])
    else:
        raise ValueError("Wrong size klayers")


def fill_outconv(outconv, klayer):
    outconv.conv_sigmoid[0].weight.data = torch.from_numpy(
        klayer.get_weights()[0])
    outconv.conv_sigmoid[0].bias.data = torch.from_numpy(
        klayer.get_weights()[1])


with torch.no_grad():
    fill_dconv(pmodel.inc.double_conv, kmodel.layers[1:5])
    fill_down(pmodel.down1, kmodel.layers[6:10])
    fill_down(pmodel.down2, kmodel.layers[11:15])
    fill_down(pmodel.down3, kmodel.layers[16:20])
    fill_down(pmodel.down4, kmodel.layers[21:25])
    fill_down(pmodel.down5, kmodel.layers[26:30])
    fill_down(pmodel.down6, kmodel.layers[31:35])
    fill_down(pmodel.down7, kmodel.layers[36:40])
    fill_down(pmodel.down8, kmodel.layers[41:45])
    fill_up(pmodel.up1, kmodel.layers[45:53])
    fill_up(pmodel.up2, kmodel.layers[53:61])
    fill_up(pmodel.up3, kmodel.layers[61:69])
    fill_up(pmodel.up4, kmodel.layers[69:77])
    fill_up(pmodel.up5, kmodel.layers[77:85])
    fill_up(pmodel.up6, kmodel.layers[85:93])
    fill_up(pmodel.up7, kmodel.layers[93:101])
    fill_up(pmodel.up8, kmodel.layers[101:109])
    fill_outconv(pmodel.outc, kmodel.layers[109])
    torch.save(pmodel.state_dict(), "li/weights_01")
