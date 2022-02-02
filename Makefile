SHELL=/bin/bash

.DEFAULT_GOAL := standard

standard:
	source ~/.bashrc
	pip install .
	pip install cython_helper/

editable:
	pip install -e .
	pip install -e cython_helper/

venv:
	/venv/bin/pip install -e .
	/venv/bin/pip install -e cython_helper/