module load slurm

jobid=$(squeue -o "%i %j" | grep nlp-chatbot | awk '{print $1}')
scancel $jobid

sbatch slurm.sh
