import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from torchaudio.datasets import VoxCeleb1Identification
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from common.common import DATASET_DIR, SAMPLE_RATE
from models.ecapa import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin

from models.preprocess import get_spectrum_feats
from models.wavlm_ecapa import WavLM_ECAPA, WavLM_ECAPA_Weighted_Fixed, WavLM_ECAPA_Weighted_Unfixed

MODEL = os.getenv("KNN_MODEL", default="WAVLM_ECAPA_WEIGHTED")
MODEL_IN_DIR = os.getenv("KNN_MODEL_IN_DIR", default=None)
MODEL_IN_DIR = None if (MODEL_IN_DIR == "None" or MODEL_IN_DIR is None) else Path(MODEL_IN_DIR)
MODEL_OUT_DIR = Path(os.getenv("KNN_MODEL_OUT_DIR", default="experiments/models"))
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = True if os.getenv("KNN_DEBUG", default="False") == "True" else False
NOF_EPOCHS = int(os.getenv("KNN_NOF_EPOCHS", default=10))
BATCH_SIZE = int(os.getenv("KNN_BATCH_SIZE", default=32))
VIEW_STEP = int(os.getenv("KNN_VIEW_STEP", default=50))


def collate_with_padding(batch):
    """
    Picks a random max_length sized segment from the recording, if it's too short, zero pad it.

    Calls `default_collate` function.
    """
    max_length = 5 * SAMPLE_RATE
    new_batch = []
    lengths = []
    for tensor, sr, speaker_id, filename in batch:
        if tensor.shape[1] - max_length > 0:
            idx = np.random.choice(tensor.shape[1] - max_length)
        else:
            idx = 0
        tensor = tensor[:, idx:]
        tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]))

        new_batch.append((tensor, sr, speaker_id, filename))
        lengths.append(tensor.shape[1] / max_length)
    return default_collate(new_batch), torch.tensor(lengths)


def load_model_with_classifier(device, model_name="ecapa_tdnn", classifier_name="classifier"):
    print(f"Loading model & classifier from {MODEL_IN_DIR}...")
    model.load_state_dict(
        torch.load(MODEL_IN_DIR / f"{model_name}.state_dict", map_location=device)
    )
    classifier.load_state_dict(
        torch.load(MODEL_IN_DIR / f"{classifier_name}.state_dict", map_location=device)
    )
    return model, classifier


if __name__ == "__main__":
    # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    # However, input tensor of the model has to have constant length and the model cannot change (i.s., it doesn't have
    # layers that are only activated on certain conditions or different number of times in every run).
    torch.backends.cudnn.benchmark = True
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voxceleb1 = VoxCeleb1Identification(root=DATASET_DIR, subset="train")
    train_dataloader = DataLoader(
        voxceleb1, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_padding
    )

    model_name = "ecapa_tdnn"
    classifier_name = "classifier"

    if MODEL == "ECAPA":
        model = ECAPA_TDNN(input_size=80, lin_neurons=192, device=device_str)
        classifier = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
        model.extract_features = get_spectrum_feats
    elif MODEL == "WAVLM_ECAPA":
        model = WavLM_ECAPA(device_str)
        classifier = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
    elif MODEL == "WAVLM_ECAPA_WEIGHTED":
        model = WavLM_ECAPA_Weighted_Fixed(device_str)
        classifier = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
    elif MODEL == "WAVLM_ECAPA_WEIGHTED_UNFIXED":
        model = WavLM_ECAPA_Weighted_Unfixed(device_str)
        classifier = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
    else:
        raise Exception("Unknown model name.")

    if MODEL_IN_DIR is not None:
        model, classifier = load_model_with_classifier(device, model_name, classifier_name)

    model.to(device)
    model.train()
    classifier.to(device)
    classifier.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": classifier.parameters()}], lr=0.001, weight_decay=float(2e-5)
    )
    scheduler = CyclicLR(
        optimizer, base_lr=1e-8, max_lr=1e-3, step_size_up=1000, cycle_momentum=False, mode="triangular2"
    )
    if MODEL_IN_DIR is not None and MODEL != "WAVLM_ECAPA_WEIGHTED_UNFIXED":
        optimizer.load_state_dict(
            torch.load(MODEL_IN_DIR / "optimizer.state_dict", map_location=device)
        )
        scheduler_filename = MODEL_IN_DIR / "scheduler.state_dict"
        if os.path.isfile(scheduler_filename):
            scheduler.load_state_dict(
                torch.load(scheduler_filename, map_location=device)
            )

    criterion = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))

    print(f"Starting training with batch size {BATCH_SIZE} and {NOF_EPOCHS} epochs...")
    print(f"Model trained: {MODEL}")
    for epoch in range(NOF_EPOCHS):
        print(f"Epoch {epoch + 1}")
        iteration = 0
        loss_acc = 0
        hits_acc = 0
        for batch, lengths in train_dataloader:
            iteration += 1

            x = model.extract_features(batch[0], lengths)
            x = x.to(device)

            batch_labels = batch[2].unsqueeze(1).to(device)
            lengths.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            cls_out = classifier(outputs).squeeze(1)
            loss = criterion(cls_out, batch_labels)
            loss.backward()  # Compute gradients
            optimizer.step()
            scheduler.step()

            # Compute stats
            loss_acc += loss.item()
            _, pred_labels = torch.max(cls_out, 1)
            hits_acc += (pred_labels == batch_labels.squeeze()).sum().item()

            # Print stats
            if iteration % VIEW_STEP == 0:
                print(
                    f"Iteration {iteration}, average loss: {loss_acc / VIEW_STEP}, prediction accuracy "
                    f"{hits_acc / (VIEW_STEP * BATCH_SIZE)}"
                )
                print(
                    f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB, "
                    f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3} GB"
                )
                loss_acc = 0
                hits_acc = 0

            # Stop training after x iterations
            if DEBUG is True and iteration == 300:
                break

        torch.save(model.state_dict(), MODEL_OUT_DIR / f"{model_name}.{epoch}.state_dict")
        torch.save(classifier.state_dict(), MODEL_OUT_DIR / f"{classifier_name}.{epoch}.state_dict")
        torch.save(optimizer.state_dict(), MODEL_OUT_DIR / f"optimizer.{epoch}.state_dict")
        torch.save(scheduler.state_dict(), MODEL_OUT_DIR / f"scheduler.{epoch}.state_dict")
