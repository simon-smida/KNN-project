import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from models import preprocess
from models.ecapa import ECAPA_TDNN
from common.common import DATASET_DIR

from pathlib import Path
from sklearn import metrics
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import minDCF
from torch.utils.data import DataLoader
from torchaudio.datasets import VoxCeleb1Verification
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


MODEL_NAME = os.getenv("KNN_MODEL", default="speechbrain/spkrec-ecapa-voxceleb")
MODEL_FILENAME = Path(
    os.getenv("KNN_MODEL_FILENAME", default="experiments/ecapa0/ecapa_tdnn.state_dict")
)
# When running on less powerful devices, it might be helpful to evaluate on a smaller number of samples
EVAL_FIRST = int(os.getenv("KNN_EVAL_FIRST")) if os.getenv("KNN_EVAL_FIRST") is not None else None

SCORES_DIR = Path("experiments/scores")
SCORES_DIR.mkdir(parents=True, exist_ok=True)
DET_DIR = Path("experiments/det")
DET_DIR.mkdir(parents=True, exist_ok=True)

LOG_INTERVAL = 500
EXPECTED_SAMPLE_RATE = 16_000


def to_filename(string):
    return "".join(s for s in str(string) if s.isalnum() or s in ["-", "_", "."])


def plot_det_curve(
    labels, scores, filename=(DET_DIR / to_filename(MODEL_NAME)), model_name="My model"
):
    """
    Plot DET curve and save it to a file called `filename`.

    :return false positive rate, false negative rate, thresholds
    """
    fpr, fnr, thresholds = metrics.det_curve(labels, scores)
    display = metrics.DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=model_name)
    display.plot()

    plt.savefig(filename)
    return fpr, fnr, thresholds


def calculate_eer(fpr, fnr, thresholds):
    """
    Calculates the Equal Error Rate (EER).

    EER = the error rate at which the false positive rate equals the false negative rate.
    Arguments of this function are the output values of `metrics.det_curve` function. Namely, false positive rate,
    false negative rate, and thresholds.
    """
    index = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[index]
    eer = fpr[index]

    return eer, eer_threshold


def calculate_minDCF(scores, labels) -> (float, float):
    """
    Calculate the minimum Detection Cost Function (minDCF) for the given scores and labels.
    :param scores: np.array of scores
    :param labels: np.array of either 0 or 1 (floats)
    :return: c_min and threshold
    """
    positives = scores[np.where(labels == 1)]
    negatives = scores[np.where(labels == 0)]

    # p_target, c_false_alarm, and c_miss are taken from the official Voxceleb 2021 website
    # https://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition2021.html
    return minDCF(positives, negatives, p_target=0.05, c_miss=1, c_fa=1)


def print_eer(eer, thr):
    msg = f"Model achieves EER = {(eer * 100):.2f}% at threshold {thr:.2f}"
    if EVAL_FIRST is not None:
        msg += f" on the first {EVAL_FIRST} samples of the test split."
    print(msg)


def print_minDCF(c, thr):
    msg = f"Model achieves minDCF = {c:.3f} at threshold {thr:.3f}"
    if EVAL_FIRST is not None:
        msg += f" on the first {EVAL_FIRST} samples of the test split."
    print(msg)


wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")


def get_normalized_embeddings_wavlm(inp, model):
    features = wav2vec_feature_extractor(inp.squeeze(), return_tensors="pt", sampling_rate=16000)
    embeddings = model(**features).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings


def get_normalized_embeddings_speechbrain(inp, model):
    """
    SpeechBrain model has some sort of feature extractor built in. Plus, it doesn't use forward method, but rather
    encode_batch.
    """
    embeddings = model.encode_batch(inp.squeeze(1))
    return embeddings


@torch.no_grad()
def evaluate_on_voxceleb1(
    model, get_embeddings, similarity_fn, logger, first_n=None, device=torch.device("cpu")
):
    """
    Evaluate the model on the VoxCeleb1 test split.

    :param model: model to evaluate
    :param get_embeddings: function taking model input and model as arguments and returning embeddings
    :param similarity_fn: function to compute similarity between two embeddings
    :param logger: logger instance
    :param first_n: how many samples to evaluate on (if None -> evaluate on the whole test split)
    :param device: should you use GPU for evaluation?
    :return:
    """
    model = model.eval()
    model = model.to(device)
    dataset = VoxCeleb1Verification(root=DATASET_DIR, download=False)
    train_dataloader = DataLoader(dataset)

    with open(SCORES_DIR / to_filename(MODEL_NAME), "w") as f:
        i = 0
        scores = []
        labels = []
        for left, right, sr, t, _, _ in train_dataloader:
            assert sr == EXPECTED_SAMPLE_RATE
            i += 1
            left_embedding = get_embeddings(left, model)
            right_embedding = get_embeddings(right, model)

            left_embedding = left_embedding.to(device)
            right_embedding = right_embedding.to(device)

            distance = similarity_fn(left_embedding, right_embedding)
            labels.append(t.item())
            scores.append(distance.item())
            f.write(f"{t} {distance.item()}\n")

            if i % LOG_INTERVAL == 0:
                logger.info(f"Processed {i} samples.")

            if first_n is not None and i >= first_n:
                break

    labels = np.array(labels)
    scores = np.array(scores)
    return labels, scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_name = MODEL_NAME
    if model_name == "microsoft/wavlm-base-sv":
        # Pre-trained WavLM + x-vector head model trained with an Additive Margin Softmax loss.
        # Training data sampled @ 16kHz.
        model = WavLMForXVector.from_pretrained(model_name)
        get_embeddings = get_normalized_embeddings_wavlm
    elif model_name == "speechbrain/spkrec-ecapa-voxceleb":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
        )
        get_embeddings = get_normalized_embeddings_speechbrain
    elif model_name == "ecapa-tdnn":
        model = ECAPA_TDNN(input_size=80, lin_neurons=192, device=device_str)
        logger.info(f"Loading model from file {MODEL_FILENAME}...")
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        get_embeddings = preprocess.get_embeddings
    else:
        raise Exception("Unknown model name.")

    similarity = torch.nn.CosineSimilarity(dim=-1)

    labels, scores = evaluate_on_voxceleb1(
        model, get_embeddings, similarity, logger, EVAL_FIRST, device
    )
    fpr, fnr, thresholds = plot_det_curve(
        labels, scores, filename=(DET_DIR / to_filename(MODEL_NAME)), model_name=model_name
    )
    eer, thr = calculate_eer(fpr, fnr, thresholds)
    c_min, c_threshold = calculate_minDCF(torch.tensor(scores), torch.tensor(labels))
    print_eer(eer, thr)
    print_minDCF(c_min, c_threshold)
