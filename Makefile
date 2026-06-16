.PHONY: setup ingest run-backend run-frontend eval clean help

help:
	@echo "Available commands:"
	@echo "  setup         : Install dependencies and setup .env"
	@echo "  ingest        : Scrape and index documentation"
	@echo "  run-backend   : Start the FastAPI backend"
	@echo "  run-frontend  : Start the Streamlit frontend"
	@echo "  eval          : Run the evaluation harness"
	@echo "  clean         : Remove data and caches"

setup:
	pip install -r requirements.txt
	if [ ! -f .env ]; then cp .env.example .env; fi
	mkdir -p data/chroma_db data/raw_docs data/chunks

ingest:
	python ingestion/ingest.py

run-backend:
	uvicorn app.main:app --reload --port 8000

run-frontend:
	streamlit run frontend/app.py

eval:
	python evaluation/run_eval.py

clean:
	rm -rf data/chroma_db/*
	rm -rf data/raw_docs/*
	rm -rf data/chunks/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
