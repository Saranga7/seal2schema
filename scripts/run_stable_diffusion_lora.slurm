#!/bin/bash
#SBATCH --job-name=seal2schema_sd_lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python/3.10.14
module load cuda/12.8

cd /scratch/mahantas/recon_seal2schema
uv run python main.py mode=train model=stable_diffusion_lora data.batch_size=1
