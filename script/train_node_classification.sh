export PYTHONPATH=.

DESCRIPTION='v1'
MODEL_NAME='text-decoder-only'
LAYOUT='s1'
POOLING_METHOD='cls'

python language_modelling/run_node_classification.py \
    --model_name_or_path model/PLMs/${MODEL_NAME}-${LAYOUT}-mlm-${DESCRIPTION} \
    --pooling ${POOLING_METHOD} \
    --dataset oag \
    --dataset_domain CS \
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
    --pad_to_max_length
