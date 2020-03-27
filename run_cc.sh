#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j.out

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip

make requirements
make train NTU_RGBD_DATA_PATH="/home/albanmdb/scratch/datasets/NTU_RGB+D/" 
			EVALUATION_TYPE=cross_view \
			MODEL_TYPE=FUSION \
			USE_POSE=True \
			USE_IR=False \
			PRETRAINED=False \
			USE_CROPPED_IR=False \
			LEARNING_RATE=1e-4 \
			WEIGHT_DECAY=0.0 \
			GRADIENT_THRESHOLD=10 \
			EPOCHS=15 \
			BATCH_SIZE=16 \
			ACCUMULATION_STEPS=1 \
			SUB_SEQUENCE_LENGTH=20 \
			NORMALIZE_SKELETON=True \
			NORMALIZATION_TYPE=1-COORD-SYS \
			KINEMATIC_CHAIN_SKELETON=False \
			AUGMENT_DATA=False \
			USE_VALIDATION=True \
			EVALUATE_TEST=True \
			SEED=0