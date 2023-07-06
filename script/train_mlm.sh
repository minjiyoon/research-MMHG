export PYTHONPATH=.

LAYOUT='s1'

#python3 language_modelling/convert_bert_to_tdo.py --layout ${LAYOUT}

python3 language_modelling/run_mlm_stream.py \
    --model_name_or_path model/PLMs/text-decoder-only-${LAYOUT} \
    --dataset oag \
    --dataset_domain CS \
    --do_train \
    --do_eval \
    --output_dir model/PLMs/text-decoder-only-${LAYOUT}-mlm \
    --overwrite_output_dir \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 10 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --eval_accumulation_steps 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --pad_to_max_length
