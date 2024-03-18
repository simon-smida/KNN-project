venv:
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt

format:
	venv/bin/ruff format *.py
