#!/bin/bash
#SBATCH --job-name=seal2schema_diffusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /scratch/mahantas/recon_seal2schema
uv run python main.py mode=train model=diffusion data.batch_size=2
