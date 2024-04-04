import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from eval.eval import evaluate_on_voxceleb1, plot_det_curve, calculate_eer

# When running on less powerful devices, it might be helpful to evaluate on a smaller number of samples
FIRST_N = 200


def print_eer(eer, thr):
    msg = f"Model achieves EER = {eer} at threshold {thr}"
    if FIRST_N is not None:
        msg += f" on the first {FIRST_N} samples of the test split."
    print(msg)


if __name__ == "__main__":
    # Pre-trained WavLM + x-vector head model trained with an Additive Margin Softmax loss.
    # Training data sampled @ 16kHz.
    model_name = "microsoft/wavlm-base-sv"
    model = WavLMForXVector.from_pretrained(model_name)
    model.eval()
    # A feature extractor processes the speech signal (as numpy float array) to the model's input format
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    similarity = torch.nn.CosineSimilarity(dim=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, scores = evaluate_on_voxceleb1(model, feature_extractor, similarity, FIRST_N, device)
    fpr, fnr, thresholds = plot_det_curve(labels, scores, filename="det_curve.png", model_name=model_name)
    eer, thr = calculate_eer(fpr, fnr, thresholds)
    print_eer(eer, thr)
