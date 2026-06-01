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

cd /scratch/mahantas/recon_seal2schema
# uv run python main.py mode=train model=pix2pix
# uv run python main.py mode=train model=diffusion 

uv run python main.py mode=train model=dino_decoder \
  run_name=dino_decoder_pretrained_higher_skeleton_weights \
  data.input_mode=seal_mask \
  data.augment.enabled=true \
  data.augment.crop_jitter=0.06 \
  data.augment.crop_scale_jitter=0.05 \
  data.augment.input_affine_enabled=false \
  model.foreground_weight=6.0 \
  model.bce_loss_weight=40.0 \
  model.dice_loss_weight=20.0 \
  model.skeleton_cldice_loss_weight=5.0 \
  model.skeleton_l1_loss_weight=5.0 \
  train.lr=0.00005 \
  train.max_epochs=100 \
  train.save_every=20 \
  train.selection_metric=skeleton_chamfer \
  train.selection_mode=min \
  train.eval_after.enabled=true \
  train.eval_after.checkpoint=best \
  train.wandb.mode=online \
  train.wandb.name=dino_decoder_pretrained_higher_skeleton_weights


exit 0
