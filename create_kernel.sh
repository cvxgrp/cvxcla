#!/bin/bash

name="cvxcla"

poetry install
poetry run pip install ipykernel
poetry run python -m ipykernel install --user --name=$name
