export PYTHONPATH=.

DESCRIPTION='pretrain'
LAYOUT='f8'

python3 language_modelling/convert_bert_to_tdo.py --layout ${LAYOUT}

python3 language_modelling/run_mlm_stream.py \
    --model_name_or_path model/PLMs/text-decoder-only-${LAYOUT} \
    --dataset oag \
    --dataset_domain CS \
    --do_train \
    --do_eval \
    --output_dir model/PLMs/text-decoder-only-${LAYOUT}-mlm-${DESCRIPTION} \
    --overwrite_output_dir \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 50 \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 32 \
    --dataloader_num_workers 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --pad_to_max_length  
