import h5py
import numpy as np
import scipy.io
import torch

from os import listdir
from dataset.utils import Mode
from utils.utils import ensure_path


def physio_channels():
    return {
        "eeg": (0, [0, 1, 2, 3, 4, 5]),
        "eog": (6, [6]),
        "chin": (7, [7]),
        "emg": (8, [8, 9]),
        "airflow": (10, [10]),
        "ecg": (12, [12]),
    }


class PhysioDataset(torch.utils.data.Dataset):
    def __init__(self, directory, selector, rng, indices=None):
        self.dir = ensure_path(directory)

        self.ref = np.load("../li/ref555.npy")

        self.data = listdir(self.dir)

        if indices:
            self.data = [self.data[i] for i in indices]

        self.rng = rng
        self.selector = selector

        self.mode = Mode.valid

    def anchor(self, ori):  # input m*n np array
        d0 = self.ref.shape[0]
        s1 = float(self.ref.shape[1])  # size in
        s2 = float(ori.shape[1])  # size out
        ori_new = ori.copy()
        for i in range(d0):
            tmp = np.interp(np.arange(s2) / (s2 - 1) *
                            (s1 - 1), np.arange(s1), self.ref[i, :])
            ori_new[i, np.argsort(ori[i, :])] = tmp
        return ori_new

    def __getitem__(self, index):
        sample = self.data[index]
        size = 8388608

        index = self.selector(self.mode)

        data = scipy.io.loadmat(
            self.dir +
            sample +
            "/" +
            sample +
            ".mat")["val"]
        labels = np.array(
            h5py.File(
                self.dir +
                sample +
                "/" +
                sample +
                "-arousal.mat",
                "r")["data"]["arousals"]).squeeze()

        d0 = data.shape[0]  # rows
        d1 = data.shape[1]  # colmns

        data = self.anchor(data)

        if d1 < size:
            diff = size - d1
            data = np.concatenate((data, np.zeros((d0, diff))), axis=1)
            labels = np.concatenate((labels, np.zeros(diff)))

        return data[index, :].astype(np.float32), labels.astype(np.float32)

    def __len__(self):
        return len(self.data)
