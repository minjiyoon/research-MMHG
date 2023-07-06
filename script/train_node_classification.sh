export PYTHONPATH=.

MODEL_NAME='text-decoder-only'
LAYOUT='s1'
POOLING_METHOD='max'

python language_modelling/run_node_classification.py \
    --model_name_or_path model/PLMs/${MODEL_NAME}-${LAYOUT}-mlm \
    --pooling ${POOLING_METHOD} \
    --dataset oag \
    --dataset_domain CS \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir model/PLMs/${MODEL_NAME}-${LAYOUT}-nc \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
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
