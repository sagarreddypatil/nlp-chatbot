#!/bin/bash

#SBATCH --job-name=nlp-chatbot   # job name
#SBATCH --output=stdout.log # output log file
#SBATCH --partition=gorman-gpu # GPU2 partition
#SBATCH --gres=gpu:1     # Request 1 GPU

# Slurm script for running this bot on Purdue's mctesla server in LWSN

./launch.sh
