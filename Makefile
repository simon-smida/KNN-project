.PHONY: venv format download_voxceleb clean train

venv:
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt -r requirements-dev.txt

format:
	venv/bin/ruff format *.py models/wavlm_ecapa.py

download-voxceleb:
	venv/bin/python -c "from common.common import download_voxceleb; download_voxceleb()"

evaluate:
	venv/bin/python evaluate.py

train:
	venv/bin/python train_ecapa.py

clean:
	rm -rf venv/
