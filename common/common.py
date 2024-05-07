import logging
import os
from pathlib import Path
from torchaudio.datasets import VoxCeleb1Identification

DATASET_DIR = Path(os.getenv("KNN_DATASET_DIR", default="voxceleb1"))
DATASET_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 16_000


def download_voxceleb(split="train"):
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading VoxCeleb1 into {DATASET_DIR}...")
    VoxCeleb1Identification(root=DATASET_DIR, subset=split, download=True)
