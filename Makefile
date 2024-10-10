SHELL := /bin/bash
.ONESHELL:

#* Variables
CONDA := conda
CONDA_BASE := $(shell $(CONDA) info --base)
PYTHON := python
CONDA_ENV := py311
VENV_DIR := .venv

#* Commands
.PHONY: all
all: conda_env venv_activate install_deps install_project

#* Activate Conda Environment
.PHONY: conda_env
conda_env:
	@type -P $(CONDA) &> /dev/null || { echo "Please install conda: https://docs.conda.io/en/latest/miniconda.html"; exit 1; }
	# Source conda.sh and activate the environment
	. $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

#* Create and Activate Virtual Environment
.PHONY: venv_activate
venv_activate:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	# Check shell and activate venv accordingly
	@if [ -n "$$BASH_VERSION" ]; then \
		source $(VENV_DIR)/bin/activate; \
	elif [ -n "$$ZSH_VERSION" ]; then \
		source $(VENV_DIR)/bin/activate; \
	elif [ -n "$$FISH_VERSION" ]; then \
		. $(VENV_DIR)/bin/activate.fish; \
	else \
		echo "Unsupported shell. Please activate the virtual environment manually."; \
		exit 1; \
	fi

#* Install Dependencies and Submodules
.PHONY: install_deps
install_deps:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	git submodule update --init --recursive

#* Install the Project and Meltingpot in Editable Mode
.PHONY: install_project
install_project:
	@echo "Installing project in editable mode..."
	pip install -e .
	@echo "Installing meltingpot in editable mode..."
	pip install -e meltingpot/
