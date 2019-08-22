#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-36:00
#SBATCH --output=%N-%j.out

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip

make requirements
make train NTU_RGBD_DATA_PATH="../../../datasets/" EVALUATION_TYPE=cross_subject \
			LEARNING_RATE=3e-5 \
			EPOCHS=50 \
			BATCH_SIZE=64 \
			SUB_SEQUENCE_LENGTH=20 \
			INCLUDE_POSE=False \
			INCLUDE_RGB=True \
			CONTINUOUS_FRAMES=False \
			NORMALIZE_SKELETON=True \
			NORMALIZATION_TYPE=1-COORD-SYS \
			AUGMENT_DATA=False \
			USE_VALIDATION=True \
			EVALUATE_TEST=True