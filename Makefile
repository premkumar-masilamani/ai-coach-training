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
	pipenv run python -m transcriber.main -v

.PHONY: ui
ui:
	pipenv run python -m transcriber.ui_app
