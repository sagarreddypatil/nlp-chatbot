#!/bin/bash

source ./venv/bin/activate

export TRANSFORMERS_CACHE=./models
python3 -m discord.main
