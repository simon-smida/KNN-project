import numpy as np
import torch

from speechbrain.utils.metric_stats import minDCF
from sklearn import metrics
from torchaudio.datasets import VoxCeleb1Verification
import matplotlib.pyplot as plt

from common.common import DATASET_DIR
from torch.utils.data import DataLoader

# TODO make sure parent dir exists
SCORES_FILENAME = "experiments/scores/scores.txt"
DEFAULT_DET_FILENAME = "experiments/det/curve.png"


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

    with open(SCORES_FILENAME, "w") as f:
        i = 0
        scores = []
        labels = []
        for left, right, sr, t, _, _ in train_dataloader:
            assert sr == 16000
            i += 1
            left_embedding = get_embeddings(left, model)
            right_embedding = get_embeddings(right, model)

            left_embedding = left_embedding.to(device)
            right_embedding = right_embedding.to(device)

            distance = similarity_fn(left_embedding, right_embedding)
            labels.append(t.item())
            scores.append(distance.item())
            f.write(f"{t} {distance.item()}\n")

            if i % 500 == 0:
                logger.info(f"Processed {i} samples.")

            if first_n is not None and i >= first_n:
                break

    labels = np.array(labels)
    scores = np.array(scores)
    return labels, scores


def plot_det_curve(labels, scores, filename=DEFAULT_DET_FILENAME, model_name="My model"):
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
