#!/bin/sh
set -x -e

# ruff
ruff format --diff lib/ train/
ruff check lib/ train/

# mypy
mypy lib/nn_lib/ train/nn_train/

# test
pytest -vv
