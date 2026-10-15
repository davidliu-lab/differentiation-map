# Summary and context path of this makefile.
SUMMARY := "rdmap computes Rambow differentiation maps from H&E whole-slide images. Quickstart: make install, then make shell, then rdmap run --wsi-dir /path/to/slides"
SHELL := /bin/bash

MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH))
VENV_DIR := .venv
PYTHON ?= python3

include $(ROOT_DIR_PATH)/make/show-help.mk

# User-facing variables.
COHORTS ?= cohorts.toml
COHORT ?=
WSI_DIR ?=
MODEL ?=
OUT ?=
NAME ?=
VALIDATION ?=

# Every rdmap call sees the variables from .env (if present).
RDMAP := set -a; [ ! -f .env ] || . ./.env; set +a; $(VENV_DIR)/bin/rdmap
CONFIG_ARGS = $(if $(wildcard $(COHORTS)),-c $(COHORTS))
COHORT_ARGS = $(CONFIG_ARGS) $(if $(COHORT),--cohort $(COHORT),$(if $(WSI_DIR),--wsi-dir $(WSI_DIR)))
MODEL_ARGS = $(if $(MODEL),--model $(MODEL)) $(if $(OUT),--out $(OUT))

$(VENV_DIR)/touchfile:
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR) --prompt "(rdmap)"
	touch $(VENV_DIR)/touchfile

.PHONY: help
## Show this help message.
help: show-help

.PHONY: install
## Create the virtual environment (.venv) and install rdmap with compatible (latest) dependency versions.
install: $(VENV_DIR)/touchfile pyproject.toml
	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install -e ".[notebook]"

.PHONY: install-locked
## Like install, but with the exact dependency versions used to build and verify the released model.
install-locked: $(VENV_DIR)/touchfile requirements-locked.txt pyproject.toml
	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install -r requirements-locked.txt -e ".[notebook]"

.PHONY: shell
## Open a bash shell with the virtual environment activated and the variables from .env exported.
shell: $(VENV_DIR)/touchfile
	exec bash --init-file <(echo "source ~/.bashrc 2>/dev/null; source $(VENV_DIR)/bin/activate; set -a; [ ! -f .env ] || . ./.env; set +a")

.PHONY: run
## Run the whole pipeline (tile, embed, predict, map, clusters) for COHORT=name or WSI_DIR=/path.
run:
	$(RDMAP) run $(COHORT_ARGS) $(MODEL_ARGS)

.PHONY: tiles
## Extract tiles for COHORT=name or WSI_DIR=/path.
tiles:
	$(RDMAP) tile $(COHORT_ARGS)

.PHONY: features
## Extract UNI-2 and CONCH features for COHORT=name or WSI_DIR=/path.
features:
	$(RDMAP) embed $(COHORT_ARGS)

.PHONY: fit
## Train a new model on the labelled COHORT=name and write it to OUT=outputs/<run>.
fit:
	$(RDMAP) fit $(COHORT_ARGS) $(if $(OUT),--out $(OUT))

.PHONY: predict
## Predict clusters and slide calls for COHORT=name or WSI_DIR=/path with MODEL (default: released model).
predict:
	$(RDMAP) predict $(COHORT_ARGS) $(MODEL_ARGS)

.PHONY: map
## Map clusters onto the slides (tile tables, QuPath GeoJSON, slide cluster maps) for COHORT=name or WSI_DIR=/path.
map:
	$(RDMAP) map $(COHORT_ARGS) $(MODEL_ARGS)

.PHONY: clusters
## Draw the cluster summary and per-cluster tile samples for COHORT=name or WSI_DIR=/path.
clusters:
	$(RDMAP) clusters $(COHORT_ARGS) $(MODEL_ARGS)

.PHONY: evaluate
## Evaluate MODEL on the labelled COHORT=name.
evaluate:
	$(RDMAP) evaluate $(COHORT_ARGS) $(MODEL_ARGS)

.PHONY: release-model
## Copy a trained model from OUT=outputs/<run> to models/NAME (optionally with VALIDATION=metrics.json) and generate its MODEL_CARD.md.
release-model:
	test -n "$(OUT)" && test -n "$(NAME)" || (echo "Usage: make release-model OUT=outputs/<run> NAME=<model-name>"; exit 1)
	$(RDMAP) release-model --src $(OUT)/model --name $(NAME) $(if $(VALIDATION),--validation $(VALIDATION))

.PHONY: clean
## Remove all outputs (predictions, tables and figures). Tiles and features are kept.
clean:
	rm -rf outputs
