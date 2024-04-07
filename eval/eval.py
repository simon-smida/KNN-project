import numpy as np
import torch

from sklearn import metrics
import matplotlib.pyplot as plt

from datasets.voxceleb1 import VoxCeleb1

SCORES_FILENAME = "scores.txt"
DEFAULT_DET_FILENAME = "det_curve.png"


def get_normalized_embeddings(inp, feature_extractor, model):
    features = feature_extractor(inp, return_tensors="pt", sampling_rate=16000)
    embeddings = model(**features).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings


@torch.no_grad()
def evaluate_on_voxceleb1(
    model, feature_extractor, similarity_fn, first_n=None, device=torch.device("cpu")
):
    model = model.to(device)
    dataset = VoxCeleb1.load(split="test")

    with open(SCORES_FILENAME, "w") as f:
        i = 0
        scores = []
        labels = []
        for t, left, right in dataset.test_iter():
            i += 1
            left_embedding = get_normalized_embeddings(left, feature_extractor, model)
            right_embedding = get_normalized_embeddings(right, feature_extractor, model)

            left_embedding = left_embedding.to(device)
            right_embedding = right_embedding.to(device)

            distance = similarity_fn(left_embedding, right_embedding)
            labels.append(t)
            scores.append(distance.item())
            f.write(f"{t} {distance.item()}\n")

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
