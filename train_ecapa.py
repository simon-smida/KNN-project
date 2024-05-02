import os
from pathlib import Path

import torch
from torch.utils.data import default_collate
from torchaudio.datasets import VoxCeleb1Identification
from torch.utils.data import DataLoader

from common.common import DATASET_DIR
from models.ecapa import ECAPA_TDNN, Classifier
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin

BATCH_SIZE = int(os.getenv("KNN_BATCH_SIZE", default=4))
MODEL_OUT_DIR = Path(os.getenv("KNN_MODEL_OUT_DIR", default="experiments/models"))
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG = True if os.getenv("KNN_DEBUG", default="False") == "True" else False
MODEL_IN_DIR = os.getenv("KNN_MODEL_IN_DIR", default=None)
NOF_EPOCHS = int(os.getenv("KNN_NOF_EPOCHS", default=10))


VIEW_STEP = 50


def collate_with_padding(batch):
    """
    Performs zero right padding and then calls `default_collate` function.
    """
    max_length = max([(b[0].shape[1]) for b in batch])
    new_batch = []
    lengths = []
    for (tensor, sr, speaker_id, filename) in batch:
        tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]))
        new_batch.append((tensor, sr, speaker_id, filename))
        lengths.append(tensor.shape[1] / max_length)
    return default_collate(new_batch), torch.tensor(lengths)


def get_spectrum_feats(wavs: torch.Tensor, legths: torch.Tensor):
    features = [fbank(wav) for wav in wavs]
    x = torch.stack(features, dim=0).squeeze(dim=1)
    x = torch.transpose(x, 1, 2)
    x = mean_var_norm(x, lengths)
    return torch.transpose(x, 1, 2)


if __name__ == "__main__":
    # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    # However, input tensor of the model has to have constant length and the model cannot change (i.s., it doesn't have
    # layers that are only activated on certain conditions or different number of times in every run).
    torch.backends.cudnn.benchmark = True
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voxceleb1 = VoxCeleb1Identification(root=DATASET_DIR, subset='train')
    train_dataloader = DataLoader(voxceleb1, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_padding)

    model = ECAPA_TDNN(input_size=80, lin_neurons=192, device=device_str)
    classify = Classifier(input_size=192, lin_neurons=192, out_neurons=1252)

    if MODEL_IN_DIR is not None:
        MODEL_IN_DIR = Path(MODEL_IN_DIR)
        print(f"Loading models from {MODEL_IN_DIR}...")
        model.load_state_dict(torch.load(MODEL_IN_DIR / "ecapa_tdnn.state_dict", map_location=device))
        classify.load_state_dict(torch.load(MODEL_IN_DIR / "classifier.state_dict", map_location=device))

    model.to(device)
    model.train()
    classify.to(device)
    classify.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": classify.parameters()}], lr=0.001, weight_decay=0.000002
    )
    if MODEL_IN_DIR is not None:
        optimizer.load_state_dict(torch.load(MODEL_IN_DIR / "optimizer.state_dict", map_location=device))

    criterion = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))
    fbank = Fbank(n_mels=80, left_frames=0, right_frames=0, deltas=False)
    mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)

    # if device_str == "cuda":
    #     print("Perform memory usage analysis...")
    #     torch.cuda.memory._record_memory_history(True)
    print(f"Starting training with batch size {BATCH_SIZE} and {NOF_EPOCHS} epochs...")
    for epoch in range(NOF_EPOCHS):
        print(f"Epoch {epoch + 1}")
        iteration = 0
        loss_acc = 0
        hits_acc = 0
        for batch, lengths in train_dataloader:
            iteration += 1

            x = get_spectrum_feats(batch[0], lengths).to(device)
            batch_labels = batch[2].unsqueeze(1).to(device)
            lengths.to(device)

            outputs = model(x)
            cls_out = classify(outputs)

            loss = criterion(cls_out, batch_labels)
            loss.backward()  # Compute gradients
            optimizer.step()
            optimizer.zero_grad()

            # Compute stats
            loss_acc += loss.item()
            _, pred_labels = torch.max(outputs, 2)
            hits_acc += torch.sum(torch.eq(batch_labels, pred_labels)).item()

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
                print(f"Device stats: {torch.cuda.memory_stats(device)}")
                loss_acc = 0
                hits_acc = 0

            # Stop training after x iterations
            if DEBUG is True and iteration == 300:
                break

        torch.save(model.state_dict(), MODEL_OUT_DIR / f"ecapa_tdnn.{epoch}.state_dict")
        torch.save(classify.state_dict(), MODEL_OUT_DIR / f"classifier.{epoch}.state_dict")
        torch.save(optimizer.state_dict(), MODEL_OUT_DIR / f"optimizer.{epoch}.state_dict")

        # if device_str == "cuda":
        #     torch.cuda.memory._dump_snapshot(MODEL_OUT_DIR / f"my_snapshot.{epoch}.pickle")
        torch.cuda.empty_cache()
