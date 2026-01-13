# To use diffrent shell, you need to define both of those.
SHELL  := /bin/bash
SRCENV := source ./env/bin/activate

init: requirements

train:
	$(SRCENV) && python src/train.py

train-dev:
	$(SRCENV) && python src/train.py --dev --max-samples=1000 --epochs 2

clean:
	rm -rf env
	rm -rf data

env:
	python -m venv env

requirements: env
	$(SRCENV) && pip install -r requirements.txt

.PHONY: init requirements clean train train-dev
