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

POSITION_TYPE='indiv'
LORA_TYPE='none'
MODEL_NAME='text-decoder-only'
LAYOUT='l1'
POOLING_METHOD='cls'
DESCRIPTION=position-${POSITION_TYPE}

python language_modelling/run_node_classification.py \
    --model_name_or_path model/PLMs/${MODEL_NAME}-${LAYOUT}-mlm-${DESCRIPTION} \
    --pooling ${POOLING_METHOD} \
    --dataset oag \
    --dataset_domain CS \
    --sample_depth 2 \
    --sample_num 2 \
    --position_type ${POSITION_TYPE} \
    --duplicate_encoding True \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir model/PLMs/${MODEL_NAME}-${LAYOUT}-nc-${DESCRIPTION} \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 100 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better False \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dataloader_num_workers 32 \
    --pad_to_max_length \
    --report_to wandb \
    --run_name ${LAYOUT}-nc-${DESCRIPTION}
