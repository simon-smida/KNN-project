# Popis problému


# Zhodnocení existujících přístupů

Na cem to trenujou?

- Článek (2022): Large-scale self-supervised speech representation learning for automatic
  speaker verification, dostupný např. [zde](https://arxiv.org/abs/2110.05777)
    - Jejich nejlepší systém dosahuje 0.537% (Vox1-O), 0.569%, and 1.180% equal error
      rate (EER) on the three official trials of VoxCeleb1, separately.
    - downstream: x-vector/ECAPA-TDNN
    - UniSpeech-SAT Large
- Článek (2024): ESPnet-SPK: full pipeline speaker embedding toolkit with
  reproducible recipes, self-supervised front-ends, and off-the-shelf models
    - EER 0.39% using WavLM-Large with ECAPA-TDNN on Vox1-0
    - SKA-TDNN??
- ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,
  [dostupné zde](https://arxiv.org/abs/2005.07143)

# Zvolený a připravený dataset 


# Zvolená a připravená metoda vyhodnocení

- Voxleb dataset
- Equal error rate - when the false acceptance rate and false rejection rate are
  equal

# Baseline řešení

- ideálně náš kód, ale můžem se inspirovat
