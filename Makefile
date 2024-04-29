.PHONY: venv format download_voxceleb clean train

venv:
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt -r requirements-dev.txt

format:
	venv/bin/ruff format datasets/*.py eval/*.py *.py

download_voxceleb:
	venv/bin/python -c "import logging; from datasets.voxceleb1 import VoxCeleb1; logging.basicConfig(level=logging.INFO); VoxCeleb1.load(split='all')"

evaluate:
	venv/bin/python evaluate.py

train:
	venv/bin/python train_ecapa.py

clean:
	rm -rf venv/
