#!/bin/bash
#SBATCH -J mf_vs  # A single job name for the array

#SBATCH --nodes=1                  # Ensure that all cores are on one machine
##SBATCH --mem=400
##SBATCH -t 3-00:00                      # Maximum execution time (D-HH:MM)
#SBATCH -t 2-00:00:00                      # Maximum execution time (D-HH:MM)
#SBATCH --partition=2080-galvani                     # stands for --partition
#SBATCH --gres=gpu:1             # optionally type and number of gpus
##SBATCH --partition=cpu-galvani
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=16             # Number of CPU cores per task

#SBATCH -o /home/wu/wkn676/RL_social/logs/%A_%a.out  # Standard output
#SBATCH -e /home/wu/wkn676/RL_social/logs/%A_%a.err  # Standard error
##SBATCH --array=0-9                  # maps 1 to N to SLURM_ARRAY_TASK_ID below
#SBATCH --mail-type=END            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=silja.kessler@student.uni-tuebingen.de  # Email to which notifications will be sent

# Use conda
source ~/.bashrc
conda activate RL_social

cd /home/wu/wkn676/emergent_transmission_2026/optimizations
#srun singularity exec /home/wu/wkn676/RL_social/singularity.sif python opti_mfree_vshaping_runsbatch.py
srun python opti_mfree_vshaping_runsbatch.py