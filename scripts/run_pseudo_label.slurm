#!/bin/bash
#SBATCH --job-name=seal2schema_pseudo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /scratch/mahantas/recon_seal2schema
uv run python main.py mode=pseudo_label
