export PYTHONPATH=.

POOLING_METHOD='max'

#--model_name_or_path data/PLMs/${MODEL_NAME} \
python language_modelling/run_decoder_only.py \
    --model_name_or_path bert \
    --dataset_name oag \
    --dataset_config_name paper \
    --do_train \
    --do_eval \
    --do_predict \
    --pooling ${POOLING_METHOD} \
    --output_dir data/PLMs/oag_decoder_only \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 100 \
    --load_best_model_at_end \
    --metric_for_best_model micro_f1 \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --pad_to_max_length
