# To use diffrent shell, you need to define both of those.
SHELL  := /bin/bash
SRCENV := source ./env/bin/activate

init: requirements

train:
	$(SRCENV) && python src/train.py

train-dev:
	$(SRCENV) && python src/train.py --max-samples=200 --batch-size=8 --epochs 3

clean:
	rm -rf env
	rm -rf data

env:
	python -m venv env

requirements: env
	$(SRCENV) && pip install -r requirements.txt

.PHONY: init requirements clean train train-dev
