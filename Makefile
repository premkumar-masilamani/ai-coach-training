INPUT_DIR = "/Users/premkumar/Downloads/AudioFiles"

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
	python3 main.py -v -i $(INPUT_DIR)
