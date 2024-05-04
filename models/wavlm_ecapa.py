import torch
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
        x = x.squeeze()
        with torch.no_grad():
            x = self.wavlm(x)
        x = self.ecapa(x.last_hidden_state)
        return x

    def parameters(self, only_trainable=False):
        return self.ecapa.parameters(only_trainable)

    def extract_features(self, wavs: torch.Tensor, lengths: torch.Tensor):
        # https://github.com/huggingface/transformers/issues/14908
        return self.extractor(wavs.squeeze(), return_tensors="pt", sampling_rate=16000,
                              return_attention_mask=False)["input_values"]
