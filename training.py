import sys
import torch
import torch.optim as optim
import numpy as np


from sklearn.metrics import roc_auc_score, average_precision_score
from dataset.physiodataset import PhysioDataset, physio_channels
from dataset.utils import ChannelSelector, DataSplitter, Mode
from unet.unet_model import UNet
from utils.neptune import NeptuneHelper
from utils.utils import ensure_path, print_cuda_info


def crossentropy_cut(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    y_pred_f = torch.clamp(y_pred_f, min=1e-7, max=(1.0 - 1e-7))

    mask = torch.gt(y_true_f, -0.5)

    losses = -(
        y_true_f * torch.log(y_pred_f) + (1.0 - y_true_f) *
        torch.log(1.0 - y_pred_f)
    )
    losses = torch.masked_select(losses, mask)
    masked_loss = torch.mean(losses)
    return masked_loss


def training(use_neptune, path, channels):

    nept = NeptuneHelper(use_neptune)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng()

    channel_selector = ChannelSelector(rng, channels, physio_channels())

    data_splitter = DataSplitter(
        path, PhysioDataset, channel_selector.indices, rng)

    model_sets = list(
        zip(data_splitter.get_train_sets(), data_splitter.get_test_sets())
    )

    for i, model_set in enumerate(model_sets):
        model_name = channel_selector.model_name(i)

        model = UNet(n_channels=len(channel_selector))
        model.to(device)

        optimizer = optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5
        )

        train_size = int(len(model_set[0]) * 0.8)  # 60 percent of original
        valid_size = len(model_set[0]) - train_size
        train_set, valid_set = torch.utils.data.random_split(
            model_set[0], [train_size, valid_size]
        )

        train_set.mode = Mode.train
        valid_set.mode = Mode.valid

        train = torch.utils.data.DataLoader(train_set, shuffle=True)
        valid = torch.utils.data.DataLoader(valid_set)

        epochs = 25
        min_valid_loss = np.inf

        for epoch in range(epochs):

            train_loss = 0.0

            nept.log(f"Epoch nr {epoch}")
            nept.log(torch.cuda.memory_allocated(0))

            for data, labels in train:

                data, labels = data.to(device), labels.to(device)

                # Clear the gradients
                optimizer.zero_grad()
                # Forward pass
                target = model(data)
                # Find the loss
                loss = crossentropy_cut(labels, target)
                # Calculate gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # Calculate loss
                train_loss += loss.item()

                nept.log(torch.cuda.memory_allocated(
                    0), "training/batch/mem_all")
                nept.log(torch.cuda.memory_reserved(
                    0), "training/batch/mem_res")

            valid_loss = 0.0
            model.eval()

            for data, labels in valid:

                data, labels = data.to(device), labels.to(device)

                # Forward pass
                target = model(data)
                # Find the loss
                loss = crossentropy_cut(labels, target)
                # Calculate loss
                valid_loss += loss.item()

                nept.log(torch.cuda.memory_allocated(
                    0), "validation/batch/mem_all")
                nept.log(torch.cuda.memory_reserved(
                    0), "validation/batch/mem_res")

            nept.log(
                f"Epoch {epoch} \t\t Training Loss: {train_loss:.2f}\tValidation Loss: {valid_loss:.2f}"
            )
            nept.log(train_loss / len(train), str(i) + "/training/epoch/loss")
            nept.log(valid_loss / len(valid),
                     str(i) + "/validation/epoch/loss")
            nept.log(torch.cuda.memory_allocated(0), "epoch/mem_all")
            nept.log(torch.cuda.memory_reserved(0), "epoch/mem_res")

            if min_valid_loss > valid_loss:
                nept.log(
                    f"\nValidation Loss Decreased({min_valid_loss:.2f}->{valid_loss:.2f})\tSaving The Model\n"
                )
                min_valid_loss = valid_loss
                torch.save(model.state_dict(), model_name)
                nept.upload_model(model_name)

        model = UNet(n_channels=len(channel_selector))
        model.to(torch.device("cpu"))
        model.load_state_dict(
            torch.load(f"./{model_name}", map_location=torch.device("cpu"))
        )
        model.eval()

        model_set[1].mode = Mode.predict
        test = torch.utils.data.DataLoader(model_set[1])

        for data, labels in test:

            labels = labels.squeeze()
            t = labels[labels > -0.5].numpy().squeeze()

            nept.log(len(labels[labels == 1.0]) / len(labels), "fraction")

            t_pred = model(data).detach().numpy().squeeze()[labels > -0.5]

            if 1 in t:
                nept.log(roc_auc_score(t, t_pred), str(i) + "/AUROC")
                nept.log(average_precision_score(t, t_pred), str(i) + "/AUPRC")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print_cuda_info()

    if len(sys.argv) > 3:
        use_neptune = sys.argv[1].lower() != "none"
        channels = set([s.lower() for s in sys.argv[3:]])
        training(
            use_neptune=use_neptune,
            path=ensure_path(
                sys.argv[2]),
            channels=channels)
    else:
        print("python training.py <neptune> <path> <channel names ...>")
