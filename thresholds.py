from os import listdir
import sys

import h5py
import numpy as np

from sklearn.metrics import confusion_matrix
from utils.domino import DominoConverter
from utils.neptune import NeptuneHelper
from utils.utils import ensure_path


FQS = 200.0 / 16.0


def downsample(lbl):
    size = 8388608
    diff = size - lbl.shape[0]
    labels = np.mean(np.concatenate((lbl, np.zeros(diff))).reshape(-1, 16), 1)
    labels[labels > 0.0] = 1.0
    labels[labels < 0.0] = -1.0
    return labels


def to_s(num):
    return num / (FQS * 60 * 60)


class ThresholdComputer:
    def __init__(self, path: str):
        self.parser = DominoConverter(min=3, fs=FQS)
        self.nept = NeptuneHelper(True)

        record_names = listdir(path)

        self.labels = [downsample(np.array(
            h5py.File(f"{path}{r}/{r}-arousal.mat", "r")["data"][
                "arousals"
            ]
        ).squeeze()) for r in record_names]

        self.predictions = [np.load(f"{path}{r}/{r}-predictions.npy")[l > -0.5]
                            for r, l in zip(record_names, self.labels)]

        for lbl in self.labels:
            self.nept.log(self.parser.get_number_arousals(
                lbl[lbl > -0.5]) / to_s(lbl.shape[0]), "trueIndex")

    def norm(self, y_true: list, y_pred: list) -> float:
        return np.linalg.norm(np.array(y_true) - np.array(y_pred))

    def corrcoef(self, y_true: list, y_pred: list) -> float:
        return np.corrcoef(np.array(y_true), np.array(y_pred))[0, 1]

    def compute(self):
        for lbl, prd in zip(self.labels, self.predictions):
            for thr in range(101):
                predictions = (prd >= thr * 0.01) * 1.0
                labels = lbl[lbl > -0.5]

                cm = confusion_matrix(labels, predictions)

                tp, tn, fp, fn = 0, 0, 0, 0

                tn = cm[0, 0]

                if cm.shape == (2, 2):
                    fp = cm[0, 1]
                    fn = cm[1, 0]
                    tp = cm[1, 1]

                self.nept.log(tp, f"{thr}/true_positive")
                self.nept.log(fp, f"{thr}/false_positive")
                self.nept.log(tn, f"{thr}/true_negative")
                self.nept.log(fn, f"{thr}/false_negative")

                self.nept.log(self.parser.get_number_arousals(
                    predictions) / to_s(lbl.shape[0]), f"{thr}/prdIndex")


def run_threshold(path: str):
    tc = ThresholdComputer(path)
    tc.compute()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_threshold(path=ensure_path(sys.argv[1]))
    else:
        print("python thresholds.py <path>")
