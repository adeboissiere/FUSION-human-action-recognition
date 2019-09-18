#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out
nvidia-smi

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip

make requirements
make train NTU_RGBD_DATA_PATH="/home/albanmdb/scratch/datasets/" EVALUATION_TYPE=cross_subject \
			MODEL_TYPE=base-IR \
			LEARNING_RATE=1e-4 \
			WEIGHT_DECAY=0.0 \
			GRADIENT_THRESHOLD=1 \
			EPOCHS=50 \
			BATCH_SIZE=16 \
			SUB_SEQUENCE_LENGTH=10 \
			NORMALIZE_SKELETON=False \
			NORMALIZATION_TYPE=1-COORD-SYS \
			KINEMATIC_CHAIN_SKELETON=False \
			AUGMENT_DATA=False \
			USE_VALIDATION=True \
			EVALUATE_TEST=True