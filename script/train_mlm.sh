#!/usr/bin/env bash
#SBATCH --partition=russ_reserved
#SBATCH --job-name=MMHG
#SBATCH --output=slurm_logs/train-mmhg-%j.out
#SBATCH --error=slurm_logs/train-mmhg-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH --exclude matrix-1-20,matrix-1-18,matrix-0-34 

ulimit -c unlimited
module load cuda-11.1.1

export WANDB_PROJECT='MMHG'
export PYTHONPATH=.

DESCRIPTION='no-position'
RANDOM_INIT='True'
LAYOUT='l1'

python3 language_modelling/convert_bert_to_tdo.py \
    --layout ${LAYOUT} \
    --random_init ${RANDOM_INIT} \
    --description ${DESCRIPTION}


python3 language_modelling/run_mlm_stream.py \
    --model_name_or_path model/PLMs/text-decoder-only-${LAYOUT}-${DESCRIPTION} \
    --dataset oag \
    --dataset_domain CS \
    --sample_depth 2 \
    --sample_num 2 \
    --position_type no_position \
    --duplicate_encoding False \
    --do_train \
    --do_eval \
    --output_dir model/PLMs/text-decoder-only-${LAYOUT}-mlm-${DESCRIPTION} \
    --overwrite_output_dir \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 50 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --dataloader_num_workers 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --pad_to_max_length \
    --report_to wandb \
    --run_name ${LAYOUT}-mlm-${DESCRIPTION}
