.PHONY: install corpus index demo cli api test check docker

install:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt

corpus:
	python scripts/build_corpus.py && python scripts/check_leaks.py

index:
	python scripts/build_index.py

demo:
	python scripts/demo.py && python scripts/render_screens.py

cli:
	python -m rag.cli

api:
	uvicorn rag.api:app --host 0.0.0.0 --port 8000

test:
	python -m pytest -q

check: corpus test

docker:
	docker compose up --build
