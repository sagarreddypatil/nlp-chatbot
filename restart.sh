#!/bin/bash
module load slurm

jobid=$(squeue -o "%i %j" | grep nlp-chatbot | awk '{print $1}')
scancel $jobid

sbatch slurm.sh

if [[ "$1" != "--no-output" ]]; then
  tail -f -n +1 stdout.log
fi

