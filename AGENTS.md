# sleep-arousal-detection

## Overview

sleep-arousal-detection is a set of PyTorch research scripts that score sleep arousals on polysomnography. A 1D U-Net maps each recording to a per-sample arousal probability. The loaders read the PhysioNet/Computing in Cardiology Challenge 2018 training layout: one directory per recording, `<id>.mat` (SciPy `loadmat` array `val`), and `<id>-arousal.mat` (HDF5, `data/arousals`). The sample path in `utils/domino.py` is the challenge record `tr04-0933`. The project is inspired by [DeepSleep](https://github.com/GuanLab/DeepSleep) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

Current status: the Python tree is the initial import. There is no installable package, no test suite, and no CI. `training.py` and `thresholds.py` are the runnable entry points, and both stop until external recordings, a reference `.npy`, and (when logging is enabled) Neptune credentials are supplied. `convert_weights.py` and `check_weights.py` cannot start, because `unet/unet0.py` is not in the tree. The license in `LICENSE` is the GNU General Public License v3.0.

## Repository layout

```
.
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── LICENSE
├── requirements.txt
├── training.py                 # 4-fold train, then AUROC / AUPRC
├── thresholds.py               # threshold sweep on saved predictions
├── convert_weights.py          # Keras weights -> PyTorch state_dict
├── check_weights.py            # compare those tensors
├── dataset/
│   ├── physiodataset.py        # PhysioDataset, physio_channels()
│   └── utils.py                # Mode, ChannelSelector, DataSplitter
├── unet/
│   ├── unet_model.py           # UNet
│   └── unet_parts.py           # DoubleConv, Down, Up, OutConv
├── utils/
│   ├── domino.py               # arousal intervals, Domino text, EDF
│   ├── git.py                  # branch and short hash from .git files
│   ├── neptune.py              # NeptuneHelper
│   └── utils.py                # print_cuda_info, ensure_path
└── .github/
    └── copilot-instructions.md
```

There is no `tests/` directory, no `unet/unet0.py`, no `li/` directory, and no `__init__.py` in `dataset/`, `unet/`, or `utils/`. Imports are absolute from the repository root (`from dataset...`, `from unet...`, `from utils...`).

## Setup

Run commands from the repository root so Python can import `dataset`, `unet`, and `utils`.

`requirements.txt` pins:

```
torch==1.9.0+cu102
torchvision==0.10.0+cu102
torchaudio===0.9.0
h5py==2.10.0
numpy==1.20.3
scipy==1.6.2
scikit-learn==0.24.2
pyedflib==0.1.22
```

On Python 3.12.3 with pip 24.0, `pip3 install -r requirements.txt` fails with `No matching distribution found for torch==1.9.0+cu102`. PyPI's torch builds for that interpreter start at 2.2.0. The `+cu102` pin is a CUDA 10.2 wheel, and this file is the only install declaration in the repo. There is no `setup.py`, `pyproject.toml`, or conda environment file.

`torchvision` and `torchaudio` are pinned and are not imported. `pyedflib` is imported only inside `DominoConverter.write_edf`.

`utils/neptune.py` does `import neptune.new as neptune`. That module is not listed in `requirements.txt`. The import succeeds on `neptune-client` 0.16.19. On that version, `neptune.init` warns that support ends in `neptune-client==1.0.0` and points callers at `init_run`. `NeptuneHelper` calls `neptune.init()` with no project and no token, so the 0.16 client reads `NEPTUNE_API_TOKEN` and `NEPTUNE_PROJECT` from the environment.

Files the repo does not contain, and that the scripts open anyway:

- A recording directory passed on the command line (see Build, run and test).
- `../li/ref555.npy`, resolved from the process working directory. From the repo root that is a sibling of the clone, not a path inside it. `PhysioDataset.__init__` loads it before any batch.
- For `thresholds.py`, `<id>/<id>-predictions.npy` next to each arousal file. No script in this repo writes that array.
- For the weight scripts, `unet/unet0.py` (`get_unet`), plus `weights_01.h5` (convert) or `li/weights_01.h5` and `li/weights_01` (check).

## Build, run and test

There is no compile step and no package build. There is no pytest, unittest, tox, or CI test job.

Byte-compile, from the repo root. This does not import PyTorch. Verified exit code 0 on Python 3.12.3:

```bash
python3 -m compileall -f training.py thresholds.py convert_weights.py check_weights.py dataset unet utils
```

Usage lines, verified exit code 0 once imports succeed (`python3 training.py` with no arguments, and the same for `thresholds.py`):

```bash
python3 training.py
python3 thresholds.py
```

```
python training.py <neptune> <path> <channel names ...>
python thresholds.py <path>
```

Those two processes were run with Python 3.12.3, torch 2.14.1+cu130 (CPU, `torch.cuda.is_available()` is false), current numpy, scipy, scikit-learn, and h5py, and `neptune-client` 0.16.19. That combination is an entry-point check. The pinned stack remains `requirements.txt`.

Weight scripts, verified, both raise `ModuleNotFoundError: No module named 'unet.unet0'`:

```bash
python3 convert_weights.py
python3 check_weights.py
```

Model smoke test, same torch build. `UNet(n_channels=5, n_classes=1)` on `torch.zeros(1, 5, 32768)` returns shape `(1, 1, 32768)`. 32768 is the network's total pool stride (`2 * 4**7`). The training pad length 8388608 is divisible by that stride. The 8388608-sample forward pass was not executed.

```bash
python3 -c "import torch; from unet.unet_model import UNet; m=UNet(5,1); m.eval(); y=m(torch.zeros(1,5,32768)); print(tuple(y.shape))"
```

Training and threshold commands, as implemented. A full train was not run: the reference array, the challenge recordings, and a CUDA device are not in this environment.

```bash
python3 training.py none /path/to/records eeg ecg
python3 training.py neptune /path/to/records li
python3 thresholds.py /path/to/records
```

Verified failures of those entry points:

- `python3 training.py none /tmp/empty-records eeg` leaves Neptune disabled, then `PhysioDataset` raises `FileNotFoundError` for `../li/ref555.npy`.
- `python3 training.py neptune /tmp/empty-records eeg` raises `NeptuneMissingApiTokenException` inside `NeptuneHelper` before the dataset is built. `neptune.init` also emits a deprecation warning pointing at `init_run`.
- `python3 thresholds.py /tmp/does-not-exist-sad` raises the same missing-token exception. With no arguments, `thresholds.py` prints usage and does not call Neptune.

`training.py` treats the first argument as Neptune off only when `lower()` equals `none` (`none`, `None`, and `NONE` all qualify). Any other string, including `false`, turns logging on. Further arguments are lowercased channel names. `ensure_path` appends a trailing `/` when the path lacks one.

Recording directory the loaders expect:

```
/path/to/records/<id>/<id>.mat
/path/to/records/<id>/<id>-arousal.mat
/path/to/records/<id>/<id>-predictions.npy   # thresholds.py only
```

`training.py` details encoded in the script:

- Channels come from `physio_channels()` unless the name set contains `li` (see Gotchas). `ChannelSelector` draws one index from each selected pool on every non-predict item. Predict mode uses the fixed index.
- `DataSplitter` lists the recording directory and cuts `range(n)` into 4 chunks with `np.array_split`, in `listdir` order. Fold `i` tests on chunk `i` and trains on the other three.
- Each training chunk is then `random_split` 80% / 20% into train and valid. The comment on that line calls the training portion 60% of the recordings: three of the four folds, then 80% of that chunk. Chunk sizes differ by one recording when the count is not divisible by 4. The split of record ids is unsorted `listdir` order. Channel bootstrap uses a `numpy` Generator; the 80/20 split uses PyTorch's default generator.
- `DataLoader` batch size stays at the default of 1. The train loader shuffles. Valid and test loaders do not.
- `UNet(n_channels=len(channel_selector))` (one output class), Adam, `lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=1e-5`, 25 epochs.
- Loss is `crossentropy_cut`: elementwise binary cross-entropy, predictions clamped to `[1e-7, 1-1e-7]`, targets `<= -0.5` dropped, then the mean of what remains.
- The best validation loss is the sum of batch losses. A new minimum saves `state_dict` to the working directory. The file name is `model_name`: channel names, fold index, and 10 hex characters from `urandom(5)`, joined with `-`. Example shape: `eeg-ecg-0-d0fe9d23d5`. When Neptune is on, `neptune.init` receives the `source_files` globs listed in Gotchas, and the checkpoint file is uploaded.
- After the epochs, the best file is loaded onto CPU with `torch.load(..., map_location="cpu")` and the held-out fold is scored. For each test record the script logs the fraction of labels equal to `1.0`. AUROC and AUPRC are logged only when the masked label vector contains a `1`.

`thresholds.py` treats every `listdir` name as a recording. For each arousal vector it zero-pads to 8,388,608 samples, averages non-overlapping groups of 16 (`FQS = 200/16 = 12.5`), and maps means `> 0` to `1` and means `< 0` to `-1`. The downsampled length is 524,288. A vector longer than 8,388,608 makes `np.zeros` raise on a negative count. Thresholds `0.00, 0.01, ..., 1.00` are compared with `<id>-predictions.npy`, masked by the downsampled labels `> -0.5`. Each threshold logs confusion counts and an arousal index through `DominoConverter(min=3, fs=12.5)`. The index is the arousal count divided by `len(labels) / (FQS * 60 * 60)`, the duration in hours of that downsampled series. Neptune is constructed with `NeptuneHelper(True)` before `listdir`.

`utils/domino.py` under `__main__` reads `training/tr04-0933/tr04-0933.mat` and the matching arousal file, and writes `john_doe.txt` and `john_doe.edf` in the working directory. Both `python3 -m utils.domino` and `python3 utils/domino.py` reach that load and, without the file, raise `FileNotFoundError` for `training/tr04-0933/tr04-0933.mat`. Importing `DominoConverter` does not read those paths.

## Code style and conventions

Python, 4-space indentation, scripts and modules run from the repo root. There is no formatter, linter, or type-checker config, so match the file you are editing.

`utils/utils.py`, `utils/neptune.py`, `dataset/utils.py`, and `thresholds.py` use docstrings and annotations on some functions. `unet/` and `training.py` are mostly unannotated. Annotations use bare `list` and `str`.

`training.py`, `thresholds.py`, and `utils/domino.py` keep their side effects under `if __name__ == "__main__"`. `convert_weights.py` and `check_weights.py` run their conversion and comparison at import time.

Channel names, checkpoint file names, and Neptune field names are plain strings built in the callers. There is no config object.

The U-Net in `unet/unet_model.py` is fixed: `DoubleConv` is Conv1d kernel 7 padding 3, ReLU, BatchNorm1d, twice. Widths are 15, 18, 21, 25, 30, 60, 120, 240, 480. `down1` max-pools by 2 and `up8` transposed-convolves by 2; the other downs and ups use 4. `OutConv` is Conv1d kernel 1 plus sigmoid. Defaults are `n_channels=13`, `n_classes=1`. Training passes the selected channel count and leaves `n_classes` at 1.

## CI and tooling

Not applicable for continuous integration. The repository has no `.github/workflows` files, no pre-commit configuration, no Makefile, and no Docker file.

The tooling that is present is `requirements.txt` and the command-line scripts above. Nothing in the tree runs formatters, linters, or tests automatically.

## Working rules for agents

- Keep changes small and focused.
- Run the tests before committing. Use the commands in Build, run and test. There is no pytest suite to run.
- Never commit secrets. Leave `NEPTUNE_API_TOKEN` and any other credentials in the environment.
- Update the docs when behaviour changes, including `README.md` when the change is user-visible.
- Use conventional commit messages.
- Keep `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` in sync. Edit `AGENTS.md`, then copy that text into the other two files. Those copies may add only a one-line header stating that `AGENTS.md` is canonical. Do not replace them with symlinks.

## Gotchas

- `pip3 install -r requirements.txt` does not succeed on Python 3.12. See Setup.
- `import neptune.new` is required to import `training.py` or `thresholds.py`, and the package is absent from `requirements.txt`. `thresholds.py` always constructs `NeptuneHelper(True)`. `training.py` does too unless the first argument lowercases to `none`.
- `convert_weights.py` imports `from unet.unet0 import get_unet` and loads `weights_01.h5` from the working directory, then writes `li/weights_01`. `check_weights.py` imports the same missing module and loads `li/weights_01.h5` plus `li/weights_01`. The Keras input paths differ. Both files run on import.
- `PhysioDataset` loads `../li/ref555.npy` from the working directory. `anchor` runs on the unpadded matrix: for each of the first `ref.shape[0]` rows it stretches that reference row to the recording length with `np.interp` and writes it back in ascending order of the original row. Zero-padding and channel selection happen after that.
- Recordings shorter than 8388608 samples are zero-padded to that length, on the signal and on the labels. The loss keeps targets `> -0.5`, so the padded zeros are trained as non-arousals. Longer recordings are left at their original length.
- Inside every epoch, `training.py` calls `torch.cuda.memory_allocated(0)` and `torch.cuda.memory_reserved(0)` with no availability check. The module device is CUDA only when `torch.cuda.is_available()` is true; the memory calls still pass device index 0. `print_cuda_info` is skipped when CUDA is absent, and when it runs it prints `get_device_properties(0)` on every device index.
- After the training batches of the first epoch, the script calls `model.eval()` and never calls `model.train()`. Later epochs therefore train in eval mode.
- `train_set.mode` and `valid_set.mode` are set on the `random_split` `Subset` objects. `PhysioDataset.__getitem__` reads `self.mode` on the dataset, which stays `Mode.valid` until the test dataset itself is set to `Mode.predict`. `Mode.train` and `Mode.valid` both bootstrap channels. Only `Mode.predict` uses the fixed indices.
- Folds follow unsorted `listdir` order.
- If the channel set contains `li`, that preset replaces every other name: predict indices `[0, 8, 7, 10, 12]`, and training draws from `0..6`, `8..9`, `7`, `10`, and `12`. Names absent from `physio_channels()` are skipped. A set of only unknown names builds `UNet(n_channels=0)`, which constructs. Index 11 is unused. The CLI names are `eeg` (pool 0–5, fixed 0), `eog` (6), `chin` (7), `emg` (pool 8–9, fixed 8), `airflow` (10), and `ecg` (12). The only signal labels in the source are in `utils/domino.py`: index 0 `F3-M2`, 8 `ABD`, 10 `AIRFLOW`, 12 `ECG`.
- Nothing in the repo writes the `<id>-predictions.npy` files `thresholds.py` reads. Training logs AUROC and AUPRC and does not save a prediction array.
- In `thresholds.py`, a one-class confusion matrix stores `cm[0, 0]` as true negatives and leaves the other three counts at 0.
- `NeptuneHelper` source globs (`../*.py`, `*.sh`, `../dataset/*.py`, `../li/*.py`, `../unet/*.py`, `../utils/*.py`) and `utils/git.py` (`../.git/HEAD`, `../.git/refs/heads/<branch>`) are relative to the working directory. From the repo root the git helper raises `FileNotFoundError` for `../.git/HEAD`. Checked from `unet/` while `HEAD` was `main` at the initial commit `e30d925`, it returned `main` and `e30d92`. `git.py` is not imported by the other modules. It reads the loose ref file, not packed-refs, and it does not call `git`.
- An all-masked label vector makes `torch.mean` run on an empty tensor inside `crossentropy_cut`.
- `crossentropy_cut` is masked binary cross-entropy on float targets, with values `<= -0.5` removed before the mean.
