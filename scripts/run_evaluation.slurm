#!/bin/bash
#SBATCH --job-name=seal2schema_pix2pix       # Job name
#SBATCH --output=output_logs/seal2schema_pix2pix%j.log        # Output file (%j = job ID)
#SBATCH --mail-user=saranga.boo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=output_logs/seal2schema_pix2pix%j.err         # Error file
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --partition=gpu              # Partition/queue name (adjust if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # CPU cores
#SBATCH --time=00-12:00:00           # Max runtime (DD-HH:MM:SS)


# Load modules (adjust to your cluster's Python/CUDA module names)
module load python/3.10.14
module load cuda/12.8

uv run python main.py mode=evaluate \
  evaluate.checkpoint_path=outputs/2026-05-27/14-12-28_dinov3_vitb16_frozen_mask_fpn_cropjitter_sharp/checkpoints/best.pth \
  output_dir=outputs/eval_dinov3_cropjitter_sharp_skeleton_metrics


exit 0
