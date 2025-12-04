INPUT_DIR = "./audio_files/"

.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	pipenv install
	pipenv shell

.PHONY: show
show:
	@echo "Listing dependencies..."
	pipenv graph

.PHONY: run
run:
	pipenv run python -m audio_transcriber.main -v -i $(INPUT_DIR)
