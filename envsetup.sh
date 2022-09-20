#!/bin/bash

# remember the current directory
DIR=$(pwd)

# change workingdir to the directory of this script
cd "$(dirname "$0")"

source ./venv/bin/activate
export TRANSFORMERS_CACHE="$(pwd)/models"

# change back to the original directory
cd "$DIR"