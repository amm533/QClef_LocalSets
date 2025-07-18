#!/bin/bash
#SBATCH --job-name=quantum_job
#SBATCH --account=ona01
#SBATCH --qos=bl_short
#SBATCH --time=00:05:00
#SBATCH --output=quantum_%j.out
#SBATCH --error=quantum_%j.err

module load python
python initial_example.py
