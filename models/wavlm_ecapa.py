import torch

from itertools import chain
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from models.ecapa import ECAPA_TDNN


class WavLM_ECAPA(torch.nn.Module):
    """
    Microsoft's WavLM base is used for feature extraction and is kept fixed during training, ECAPA_TDNN takes these
    features and produces embeddings out of them.
    """

    def __init__(self, device_str):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-base")
        self.ecapa = ECAPA_TDNN(input_size=768, lin_neurons=192, device=device_str)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            x = x.squeeze(0)

        with torch.no_grad():
            x = self.wavlm(x)
        x = self.ecapa(x.last_hidden_state)
        return x

    def parameters(self, only_trainable=False):
        return self.ecapa.parameters(only_trainable)

    def extract_features(self, wavs: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/huggingface/transformers/issues/14908
        return self.extractor(
            wavs.squeeze(), return_tensors="pt", sampling_rate=16000, return_attention_mask=False
        )["input_values"]


class WavLM_ECAPA_Weighted(WavLM_ECAPA):
    """
    Improves base WavLM_ECAPA by computing weighted sum across all WavLM hidden layers output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_sum = torch.nn.Conv1d(
            in_channels=13, out_channels=1, kernel_size=1
        )  # bias=False?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            x = x.squeeze(0)

        with torch.no_grad():
            # x.hidden_states returns tensor with dimensions (batch_size, frames, channels=768) in a tuple
            x = self.wavlm(x, output_hidden_states=True)

        # Calculate weighted sum of hidden layers
        x = torch.stack([i for i in x.hidden_states], dim=1)
        vx = x.view(x.shape[0], x.shape[1], -1)
        vx = self.weighted_sum(vx)
        x = vx.view(x.shape[0], 1, x.shape[2], x.shape[3]).squeeze(1)

        x = self.ecapa(x)  # Returns tensor with dimensions (batch_size, 1, embeddings)
        return x

    def parameters(self, only_trainable=False):
        return chain(
            self.ecapa.parameters(only_trainable), self.weighted_sum.parameters(only_trainable)
        )
