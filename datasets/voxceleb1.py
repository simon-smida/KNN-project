import logging
import os
import zipfile
from pathlib import Path
from os.path import dirname

import requests

TRAIN_SPLIT_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip?download=true"
TEST_SPLIT_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip?download=true"

DOWNLOAD_BLOCK_SIZE = 1024
LOG_INTERVAL = 100 * 1024 * DOWNLOAD_BLOCK_SIZE


class VoxCeleb1:
    def __init__(self, root_dir: Path, logger: logging.Logger):
        self.root_dir = root_dir
        self.logger = logger

    def _download_file(self, url, filename):
        self.logger.info(f"Downloading file from {url} to {filename}...")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                self._download_file_chunks(f, r)
        self.logger.info(f"File {filename} successfully downloaded.")

    def _download_file_chunks(self, f, r):
        downloaded_size = 0
        for chunk in r.iter_content(chunk_size=DOWNLOAD_BLOCK_SIZE):
            downloaded_size += len(chunk)
            f.write(chunk)

            if (downloaded_size // LOG_INTERVAL) > (downloaded_size - len(chunk)) // LOG_INTERVAL:
                self.logger.info(f"Downloaded {downloaded_size / (1024 * 1024)} MB so far...")

    def _unzip_file(self, zip_filename):
        target_filename = zip_filename.stem
        self.logger.info(
            f"Unzipping file {zip_filename} into {target_filename} in the same directory"
        )
        with zipfile.ZipFile(zip_filename, "r") as zip:
            zip.extractall(Path(dirname(zip_filename)) / Path(target_filename))

    def _download_split(self, split_url: str, split_dir: Path):
        """
        Download a split from `split_url` and extract it into `split_dir`.

        :param split_url: URL, from which the split will be downloaded (expects a zip file)
        :param split_dir: directory relative to the VoxCeleb root dir
        :return:
        """
        if os.path.exists(self.root_dir / split_dir):
            self.logger.info(f"Split {split_dir} is already present.")
            return

        zip_filename = Path(f"{self.root_dir}/{split_dir}.zip")
        self._download_file(split_url, zip_filename)
        self._unzip_file(zip_filename)
        self.logger.info(f"Removing {zip_filename}")
        os.remove(zip_filename)
        self.logger.info(f"Split {split_dir} has been successfully downloaded.")

    @classmethod
    def load(cls, split="all", root_dir=Path("voxceleb1"), logger=logging.getLogger(__name__)):
        """
        Load the VoxCeleb1 dataset split(s) from an unofficial Hugging face repository.

        :param split: either "all", "train" or "test"
        :param root_dir: root directory in which the dataset splits are downloaded;
            if not absolute, it will be relative to the current working directory
        :param logger: logger instance to be used for logging
        :return: instance of VoxCeleb1 class
        """
        if not root_dir.is_absolute():
            root_dir = Path(os.getcwd() / Path(root_dir))

        dataset = cls(root_dir, logger=logger)

        if split == "all":
            dataset._download_split(TEST_SPLIT_URL, Path("vox1_test"))
            dataset._download_split(TRAIN_SPLIT_URL, Path("vox1_train"))
        elif split == "test":
            dataset._download_split(TEST_SPLIT_URL, Path("vox1_test"))
        elif split == "train":
            dataset._download_split(TRAIN_SPLIT_URL, Path("vox1_train"))
        else:
            raise ValueError(f"Invalid split value: {split}")

        return dataset
