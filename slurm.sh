#!/bin/bash

#SBATCH --job-name=nlp-chatbot   # job name
#SBATCH --output=gpu.out # output log file
#SBATCH --partition=mctesla-gpu # GPU2 partition
#SBATCH --gres=gpu:1     # Request 1 GPU

./launch.sh
