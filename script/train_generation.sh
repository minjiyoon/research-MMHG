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
module load cuda-11.1.1

export WANDB_PROJECT='MMHG'
#export WANDB_WATCH='gradients'
export PYTHONPATH=.

MODEL_NAME='t5-base'
#MODEL_NAME='google/flan-t5-base'
#MODEL_NAME='google/long-t5-local-base'
#MODEL_NAME='facebook/opt-350m'
TASK='section_summarization'
CONTEXT='section_only'
DESCRIPTION=${MODEL_NAME}-${TASK}-${CONTEXT}

python language_modelling/run_generation.py \
    --dataset wikiweb2m \
    --model_name_or_path ${MODEL_NAME} \
    --task ${TASK} \
    --context ${CONTEXT} \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir model/PLMs/${MODEL_NAME}-${TASK}-${CONTEXT} \
    --overwrite_output_dir \
    --logging_strategy steps \
    --logging_first_step True \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --prediction_loss_only \
    --max_steps 10000 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --dataloader_num_workers 64 \
    --fp16 \
    --report_to wandb \
    --run_name ${DESCRIPTION}
