.PHONY: install train test api app

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m magnetic_ml.train

test:
	pytest

api:
	uvicorn magnetic_ml.api:app --reload --port 8000

app:
	streamlit run app/dashboard.py
