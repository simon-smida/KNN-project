import os
from pathlib import Path

import torch
from torch.utils.data import default_collate
from torchaudio.datasets import VoxCeleb1Identification
from torch.utils.data import DataLoader

from common.common import DATASET_DIR, SAMPLE_RATE
from models.ecapa import ECAPA_TDNN, Classifier
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin

from models.preprocess import get_spectrum_feats
from models.wavlm_ecapa import WavLM_ECAPA, WavLM_ECAPA_Weighted

MODEL = os.getenv("KNN_MODEL", default="WAVLM_ECAPA_WEIGHTED")
MODEL_IN_DIR = os.getenv("KNN_MODEL_IN_DIR", default=None)
MODEL_OUT_DIR = Path(os.getenv("KNN_MODEL_OUT_DIR", default="experiments/models"))
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = True if os.getenv("KNN_DEBUG", default="False") == "True" else False
NOF_EPOCHS = int(os.getenv("KNN_NOF_EPOCHS", default=10))
BATCH_SIZE = int(os.getenv("KNN_BATCH_SIZE", default=16))
VIEW_STEP = int(os.getenv("KNN_VIEW_STEP", default=50))


def collate_with_padding(batch):
    """
    Performs zero right padding and then calls `default_collate` function.

    Also, trims all recordings to 5 seconds.
    """
    # max_length = max([(b[0].shape[1]) for b in batch])
    max_length = 5 * SAMPLE_RATE
    new_batch = []
    lengths = []
    for tensor, sr, speaker_id, filename in batch:
        tensor = tensor[:max_length]
        tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]))
        new_batch.append((tensor, sr, speaker_id, filename))
        lengths.append(tensor.shape[1] / max_length)
    return default_collate(new_batch), torch.tensor(lengths)


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

    if MODEL == "ECAPA":
        model = ECAPA_TDNN(input_size=80, lin_neurons=192, device=device_str)
        classify = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)

        if MODEL_IN_DIR is not None:
            MODEL_IN_DIR = Path(MODEL_IN_DIR)
            print(f"Loading models from {MODEL_IN_DIR}...")
            model.load_state_dict(
                torch.load(MODEL_IN_DIR / "ecapa_tdnn.state_dict", map_location=device)
            )
            classify.load_state_dict(
                torch.load(MODEL_IN_DIR / "classifier.state_dict", map_location=device)
            )
        model.extract_features = get_spectrum_feats
    elif MODEL == "WAVLM_ECAPA":
        model = WavLM_ECAPA(device_str)
        classify = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
    elif MODEL == "WAVLM_ECAPA_WEIGHTED":
        model = WavLM_ECAPA_Weighted(device_str)
        classify = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)
    else:
        raise Exception("Unknown model name.")

    model.to(device)
    model.train()
    classify.to(device)
    classify.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": classify.parameters()}], lr=0.001, weight_decay=0.000002
    )
    if MODEL_IN_DIR is not None:
        optimizer.load_state_dict(
            torch.load(MODEL_IN_DIR / "optimizer.state_dict", map_location=device)
        )

    criterion = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))

    print(f"Starting training with batch size {BATCH_SIZE} and {NOF_EPOCHS} epochs...")
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
            cls_out = classify(outputs).squeeze(1)
            loss = criterion(cls_out, batch_labels)
            loss.backward()  # Compute gradients
            optimizer.step()

            batch_labels = batch_labels.squeeze(1)
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

        torch.save(model.state_dict(), MODEL_OUT_DIR / f"ecapa_tdnn.{epoch}.state_dict")
        torch.save(classify.state_dict(), MODEL_OUT_DIR / f"classifier.{epoch}.state_dict")
        torch.save(optimizer.state_dict(), MODEL_OUT_DIR / f"optimizer.{epoch}.state_dict")
