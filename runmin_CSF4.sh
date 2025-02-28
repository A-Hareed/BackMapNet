#!/bin/bash --login
#SBATCH -p multicore     # (--partition=multicore) 
#SBATCH -n 40           # Can specify 2 to 40 cores in the multicore partition

export OMP_NUM_THREADS=$SLURM_NTASKS

