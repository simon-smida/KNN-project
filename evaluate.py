import logging

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from speechbrain.inference.speaker import EncoderClassifier
from eval.eval import evaluate_on_voxceleb1, plot_det_curve, calculate_eer

# When running on less powerful devices, it might be helpful to evaluate on a smaller number of samples
FIRST_N = None
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"


def print_eer(eer, thr):
    msg = f"Model achieves EER = {eer} at threshold {thr}"
    if FIRST_N is not None:
        msg += f" on the first {FIRST_N} samples of the test split."
    print(msg)


wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")


def get_normalized_embeddings_wavlm(inp, model):
    features = wav2vec_feature_extractor(inp, return_tensors="pt", sampling_rate=16000)
    embeddings = model(**features).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings


def get_normalized_embeddings_speechbrain(inp, model):
    """
    SpeechBrain model has some sort of feature extractor built in. Plus, it doens't use forward method, but rather
    encode_batch.
    """
    embeddings = model.encode_batch(torch.tensor(inp))
    return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    else:
        raise Exception("Unknown model name.")

    similarity = torch.nn.CosineSimilarity(dim=-1)

    labels, scores = evaluate_on_voxceleb1(
        model, get_embeddings, similarity, logger, FIRST_N, device
    )
    fpr, fnr, thresholds = plot_det_curve(
        labels, scores, filename="det_curve.png", model_name=model_name
    )
    eer, thr = calculate_eer(fpr, fnr, thresholds)
    print_eer(eer, thr)
