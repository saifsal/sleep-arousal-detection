import torch
import numpy as np

from unet.unet0 import get_unet
from unet.unet_model import UNet

kmodel = get_unet()
kmodel.load_weights("li/weights_01.h5")

pmodel = UNet(5, 1)
pmodel.load_state_dict(
    torch.load(
        "li/weights_01",
        map_location=torch.device("cpu")))
pmodel.eval()


def check_tensor(t1, t2):
    if not (
        np.array_equal(
            t1, t2) or np.array_equal(
            t1, t2.transpose(
                (2, 1, 0)))):
        print(t1.shape)
        print(t2.shape)
        print(
            "{:e}  {:e} | {:e}   {:e}".format(
                np.mean(t1), np.var(t1), np.mean(t2), np.var(t2)
            )
        )


def check_sconv(sconv, klayers):
    if len(klayers) == 2:
        # Conv1d
        check_tensor(
            sconv[0].weight.data.detach().numpy(),
            klayers[0].get_weights()[0])
        check_tensor(
            sconv[0].bias.data.detach().numpy(),
            klayers[0].get_weights()[1])
        # Batchnorm1d
        check_tensor(
            sconv[1].weight.data.detach().numpy(),
            klayers[1].get_weights()[0])
        check_tensor(
            sconv[1].bias.data.detach().numpy(),
            klayers[1].get_weights()[1])
        check_tensor(
            sconv[1].running_mean.data.detach().numpy(),
            klayers[1].get_weights()[2])
        check_tensor(
            sconv[1].running_var.data.detach().numpy(),
            klayers[1].get_weights()[3])
    else:
        raise ValueError("Wrong size klayers")


def check_dconv(dconv, klayers):
    if len(klayers) == 4:
        check_sconv(dconv[0:3:2], klayers[:2])
        check_sconv(dconv[3:6:2], klayers[2:4])
    else:
        raise ValueError("Wrong size klayers")


def check_down(down, klayers):
    if len(klayers) == 4:
        check_dconv(down.maxpool_conv[1].double_conv, klayers)
    else:
        raise ValueError("Wrong size klayers")


def check_up(up, klayers):
    if len(klayers) == 8:
        check_tensor(
            up.up.weight.data.detach().numpy(),
            klayers[1].get_weights()[0].squeeze())
        check_tensor(
            up.up.bias.data.detach().numpy(),
            klayers[1].get_weights()[1])
        check_dconv(up.conv.double_conv, klayers[4:])
    else:
        raise ValueError("Wrong size klayers")


def check_outconv(outconv, klayer):
    check_tensor(
        outconv.conv_sigmoid[0].weight.data.detach().numpy(),
        klayer.get_weights()[0])
    check_tensor(
        outconv.conv_sigmoid[0].bias.data.detach().numpy(),
        klayer.get_weights()[1])


with torch.no_grad():
    check_dconv(pmodel.inc.double_conv, kmodel.layers[1:5])
    check_down(pmodel.down1, kmodel.layers[6:10])
    check_down(pmodel.down2, kmodel.layers[11:15])
    check_down(pmodel.down3, kmodel.layers[16:20])
    check_down(pmodel.down4, kmodel.layers[21:25])
    check_down(pmodel.down5, kmodel.layers[26:30])
    check_down(pmodel.down6, kmodel.layers[31:35])
    check_down(pmodel.down7, kmodel.layers[36:40])
    check_down(pmodel.down8, kmodel.layers[41:45])
    check_up(pmodel.up1, kmodel.layers[45:53])
    check_up(pmodel.up2, kmodel.layers[53:61])
    check_up(pmodel.up3, kmodel.layers[61:69])
    check_up(pmodel.up4, kmodel.layers[69:77])
    check_up(pmodel.up5, kmodel.layers[77:85])
    check_up(pmodel.up6, kmodel.layers[85:93])
    check_up(pmodel.up7, kmodel.layers[93:101])
    check_up(pmodel.up8, kmodel.layers[101:109])
    check_outconv(pmodel.outc, kmodel.layers[109])
