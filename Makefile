INPUT_DIR = ./audio_files/

.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	pipenv install

.PHONY: show
show:
	@echo "Listing dependencies..."
	pipenv graph

.PHONY: run
run:
	pipenv run python -m transcriber.main -v -i $(INPUT_DIR)
