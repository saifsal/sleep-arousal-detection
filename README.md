# sleep-arousal-detection

PyTorch scripts that score sleep arousals on polysomnography with a 1D U-Net. Inspired by [DeepSleep](https://github.com/GuanLab/DeepSleep) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

The network maps a multi-channel recording to a probability for every sample. Training uses the PhysioNet/Computing in Cardiology Challenge 2018 file layout, which is what `dataset/physiodataset.py` and the `tr04-0933` path in `utils/domino.py` implement. The goal is per-sample arousal detection, then AUROC and AUPRC on held-out recordings.

## Features

- 1D U-Net in `unet/unet_model.py`. Eight downsample and eight upsample stages, sigmoid output, default 13 input channels and 1 class. Training builds it with however many channels the selector keeps.
- Channel sets on the command line (`eeg`, `eog`, `chin`, `emg`, `airflow`, `ecg`) or the `li` five-channel preset used by the weight-conversion scripts (`UNet(5, 1)`).
- `PhysioDataset` reads each recording's `val` matrix and arousal vector, runs the `anchor` remapping against `../li/ref555.npy`, and zero-pads recordings shorter than 8,388,608 samples.
- Four-fold training in `training.py`: Adam for 25 epochs, masked binary cross-entropy (`crossentropy_cut`), checkpoint of the best validation loss, then AUROC and AUPRC on the held-out fold.
- Optional [Neptune](https://neptune.ai) logging from `training.py` of loss, AUROC, AUPRC, GPU memory, and the checkpoint. `thresholds.py` always opens a Neptune run.
- `thresholds.py` sweeps decision thresholds on precomputed prediction arrays and logs confusion counts and an arousal index.
- `DominoConverter` turns a binary arousal track into Domino annotation text, and can write an EDF via `pyedflib`.
- `convert_weights.py` and `check_weights.py` copy a Keras checkpoint into this U-Net and compare the tensors. Both import `unet.unet0`, which is not in the repository, so they do not run.

The repository has no screenshots and no saved metrics.

## Examples

Train on EEG and ECG, with Neptune left off, then run the `li` preset with Neptune on:

```bash
python3 training.py none /path/to/records eeg ecg
python3 training.py neptune /path/to/records li
```

Sweep thresholds on prediction files already stored beside the arousal files:

```bash
python3 thresholds.py /path/to/records
```

With no arguments the scripts print:

```
python training.py <neptune> <path> <channel names ...>
python thresholds.py <path>
```

A five-channel network preserves length on an input whose time axis is the pool stride. `UNet(5, 1)` applied to a tensor of shape `(1, 5, 32768)` returns `(1, 1, 32768)`.

Recording directory:

```
/path/to/records/<id>/<id>.mat                 # SciPy loadmat, array "val"
/path/to/records/<id>/<id>-arousal.mat         # HDF5, data/arousals
/path/to/records/<id>/<id>-predictions.npy     # thresholds.py only
```

`<id>` is the directory name. `utils/domino.py` uses `tr04-0933` as its hardcoded example. Those files are not shipped in this repo.

Checkpoints are `state_dict` files written to the working directory. The name joins the channel names, the fold index, and 10 hex characters, for example `eeg-ecg-0-d0fe9d23d5`.

## Prerequisites

- A Python environment that can install the pins in `requirements.txt` (see Install). The CUDA 10.2 torch pin is the one the file asks for.
- The `neptune.new` API. It is not listed in `requirements.txt`. `import neptune.new` succeeds on `neptune-client` 0.16.19, and that version warns that `neptune.init` support ends in `neptune-client==1.0.0`.
- Challenge-style recordings, as above.
- `../li/ref555.npy` relative to the working directory. From a clone at `/path/sleep-arousal-detection`, the loader opens `/path/li/ref555.npy`.
- For a real training run, a CUDA device. The model can be constructed on CPU, and the epoch loop still calls `torch.cuda.memory_allocated(0)` and `torch.cuda.memory_reserved(0)`.
- For Neptune runs, `NEPTUNE_API_TOKEN` and `NEPTUNE_PROJECT` in the environment.
- For `thresholds.py`, a `<id>-predictions.npy` for every recording. This repository does not create those files.
- For the weight scripts, `unet/unet0.py` and the Keras and PyTorch weight files named in those scripts. None of them are in the tree.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

On Python 3.12.3 with pip 24.0 that command fails:

```
No matching distribution found for torch==1.9.0+cu102
```

PyPI's torch builds for that interpreter start at 2.2.0. There is no second install path in the repo (`setup.py`, `pyproject.toml`, and lockfiles are absent).

The pins are `torch==1.9.0+cu102`, `torchvision==0.10.0+cu102`, `torchaudio===0.9.0`, `h5py==2.10.0`, `numpy==1.20.3`, `scipy==1.6.2`, `scikit-learn==0.24.2`, and `pyedflib==0.1.22`. `torchvision` and `torchaudio` are not imported. `pyedflib` is used only by `DominoConverter.write_edf`.

Importing `training.py` or `thresholds.py` also needs a Neptune client that still exposes `neptune.new`. `import neptune.new` succeeds on `neptune-client` 0.16.19.

Run the scripts from the repository root. `dataset`, `unet`, and `utils` have no `__init__.py`; Python imports them because the script directory is on `sys.path`.

## Usage

### Channels

Names are lowercased. Order on the command line does not matter. Unknown names are ignored. If `li` is present, it wins and the other names are dropped.

| Name | Index used at predict time | Index drawn for each training item |
| --- | --- | --- |
| `eeg` | 0 | one of 0–5 |
| `eog` | 6 | 6 |
| `chin` | 7 | 7 |
| `emg` | 8 | one of 8–9 |
| `airflow` | 10 | 10 |
| `ecg` | 12 | 12 |

The `li` preset predicts with indices `[0, 8, 7, 10, 12]`. During training those five inputs are drawn from `0–6`, `8–9`, `7`, `10`, and `12`. Index 11 is never selected.

`utils/domino.py` labels four of these indices in its example export: 0 `F3-M2`, 8 `ABD`, 10 `AIRFLOW`, 12 `ECG`.

### Training

```bash
python3 training.py <neptune> <path> <channel names ...>
```

`<neptune>` is the string `none` to skip Neptune. Any other first argument enables it. `<path>` is the recording directory.

What one run does:

1. Builds four folds from `listdir` order (not shuffled). Each fold tests on one quarter and trains on the other three.
2. Splits that training list 80% / 20% into train and validation. The comment in `training.py` calls the training portion 60% of the original recordings: three of the four folds, then 80% of that chunk. Fold sizes differ by one recording when the count is not divisible by 4.
3. Trains for 25 epochs with Adam (`lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=1e-5`) and batch size 1.
4. Computes `crossentropy_cut`, binary cross-entropy with predictions clamped to `[1e-7, 1-1e-7]` and targets `<= -0.5` removed.
5. Saves the `state_dict` whenever the sum of validation batch losses improves.
6. Reloads that file on CPU and logs, per test recording, the fraction of labels equal to 1. When any positive label remains after the mask, it also logs AUROC and AUPRC.

`PhysioDataset.anchor` remaps each leading channel against `../li/ref555.npy` before the channel selector runs.

### Threshold sweep

```bash
python3 thresholds.py <path>
```

For each `listdir` entry the script reads `<id>-arousal.mat` and `<id>-predictions.npy`. Labels are zero-padded to 8,388,608 samples, averaged in groups of 16 (the script's 200 Hz rate divided by 16, so 12.5 Hz, length 524,288), then thresholded from 0 to 1 in steps of 0.01. A vector longer than 8,388,608 samples makes the pad raise. Arousals are runs of positives separated by at least 3 seconds and at least 3 seconds long (`DominoConverter`). Counts are printed and sent to Neptune. `NeptuneHelper(True)` runs before the directory listing, so the sweep needs a token.

### Domino export

Import `DominoConverter` from `utils/domino.py`. `get_arousals` / `get_number_arousals` find the runs. `write_anno` writes a Domino text file. `write_edf` writes an EDF and imports `pyedflib` at call time.

Running the module as a program is a one-off for `training/tr04-0933/tr04-0933` and writes `john_doe.txt` and `john_doe.edf` in the working directory. Without that recording, the command raises `FileNotFoundError` for `training/tr04-0933/tr04-0933.mat`:

```bash
python3 -m utils.domino
```

### Weight conversion

`convert_weights.py` and `check_weights.py` are the Keras-to-PyTorch path for a 5-channel network. On a tree that matches this repository both exit immediately:

```
ModuleNotFoundError: No module named 'unet.unet0'
```

## Configuration

Configuration is the command line plus the process environment. There is no config file.

| Input | Effect |
| --- | --- |
| First argument of `training.py` | `none` disables Neptune. Anything else enables it. |
| Recording path | Directory of `<id>/` folders. A missing trailing `/` is added. |
| Channel names | See the table above. `li` selects the five-channel preset. |
| `NEPTUNE_API_TOKEN` | Read by `neptune.init()` because the code passes no token. |
| `NEPTUNE_PROJECT` | Read by the same call because the code passes no project. |
| `../li/ref555.npy` | Reference array for `PhysioDataset.anchor`, relative to the working directory. |

Hyperparameters that are constants in `training.py`: 25 epochs, Adam as cited above, batch size 1, pad length 8,388,608, four folds, 80/20 train/valid inside the training folds.

## Development and testing

There is no test suite and no CI. The check that does not need the pinned wheels is a byte-compile from the repo root:

```bash
python3 -m compileall -f training.py thresholds.py convert_weights.py check_weights.py dataset unet utils
```

That command exits 0 on Python 3.12.3. It does not import PyTorch or load data.

Agent and contributor instructions live in `AGENTS.md` (canonical), `CLAUDE.md`, and `.github/copilot-instructions.md`.

Layout: `training.py`, `thresholds.py`, `convert_weights.py`, and `check_weights.py` are the scripts. `dataset/` loads recordings and splits folds. `unet/` is the network. `utils/` holds Neptune, Domino export, a CUDA print helper, and a small `.git` reader. `requirements.txt` is the dependency pin file. `LICENSE` is GPLv3.

## Project status

The Python sources are the initial import. There is no changelog, release, or roadmap in the tree.

`training.py` and `thresholds.py` are the scripts that get as far as their command lines. A training run still needs the reference `.npy`, the recordings, and a CUDA device for the memory calls in the epoch loop. A threshold run needs prediction `.npy` files and a Neptune login, because logging is hard-wired on. Weight conversion is blocked on the missing `unet.unet0` module. The code is not packaged for pip.

## Licence

GNU General Public License v3.0. The text is in [LICENSE](LICENSE). The Python files do not add their own copyright header.
