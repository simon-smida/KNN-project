import torch
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization


fbank = Fbank(n_mels=80, left_frames=0, right_frames=0, deltas=False)
mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)


def get_spectrum_feats(wavs: torch.Tensor, lengths: torch.Tensor):
    """
    Get spectrum features for batch of recordings in wavs.

    Because batch is padded, lenghts represent what fraction of the current vector is original recording,
    the rest is padding.
    """
    features = [fbank(wav) for wav in wavs]
    x = torch.stack(features, dim=0).squeeze(dim=1)
    x = torch.transpose(x, 1, 2)
    x = mean_var_norm(x, lengths)
    return torch.transpose(x, 1, 2)


def get_embeddings(wav: torch.Tensor, model: torch.nn.Module):
    """
    Return embeddings for a recording `wav` and model `model`

    wav is a tensor representing one recording returned from torchaudio Voxceleb1Identification dataset.
    """
    x = fbank(wav.squeeze(1)).squeeze(0).transpose(0, 1)
    x = mean_var_norm(x, torch.ones(x.shape[1]))
    x = torch.transpose(x, 0, 1).unsqueeze(dim=0)
    embeddings = model(x)
    return embeddings
