#!/usr/bin/env bash
#SBATCH --partition=russ_reserved
#SBATCH --job-name=MMHG
#SBATCH --output=slurm_logs/train-wikiweb2m-%j.out
#SBATCH --error=slurm_logs/train-wikiweb2m-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH --exclude matrix-1-20,matrix-1-18,matrix-0-34 

ulimit -c unlimited
#module load cuda-11.1.1

#export WANDB_WATCH='gradients'
export PYTHONPATH=.

MODEL_NAME='t5-base'
#MODEL_NAME='google/flan-t5-base'
#MODEL_NAME='google/long-t5-local-base'
#MODEL_NAME='facebook/opt-350m'
TASK='section'
CONTEXT='section_only'
DESCRIPTION=${MODEL_NAME}-${TASK}-${CONTEXT}

python language_modelling/run_generation.py \
    --dataset wikiweb2m \
    --model_name_or_path ${MODEL_NAME} \
    --task ${TASK} \
    --context ${CONTEXT} \
    --max_input_length 512 \
    --max_output_length 128 \
    --epochs 90 \
    --steps_per_epoch 3000 \
    --val_steps_per_epoch 300 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --per_device_val_batch_size 32 \
    --dataloader_num_workers 4 \
    --grad_accumulation_steps 4 \
    --fp16 \
    --wandb_project MMHG \
    --wandb_run ${DESCRIPTION}
