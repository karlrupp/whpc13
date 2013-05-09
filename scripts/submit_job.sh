#!/bin/bash
### Job settings
#SBATCH --job-name testjob
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SALLOC --gres=gpu:1

### Environment setup
. /etc/profile
module load cuda/5.0
module load mpi

### Run task
srun ./pbratu


