#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning summary generation models"""
import wandb
import logging
import math
import os
import random
import sys
from time import perf_counter

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit

import torch
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import evaluate
from wikiweb2m.cider import Cider

#from model import TDOConfig, TDOForMaskedLM, TDOForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    dataset: Optional[str] = field(
        default='wikiweb2m', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task: Optional[str] = field(
        default='section_summarization', metadata={"help": "The domain of OAG datasets"}
    )
    context: Optional[str] = field(
        default='section_only', metadata={"help": "The domain of OAG datasets"}
    )

    sample_depth: Optional[int] = field(
        default=1, metadata={"help": "neighborhood hops for the computation graph sampling."}
    )
    sample_num: Optional[int] = field(
        default=5, metadata={"help": "number of neighbors to sample per node."}
    )
    position_type: Optional[str] = field(
        default='no_position', metadata={"help": "position encoding methods for neighbors (node_type, layer, layer_node_type, metapath)."}
    )
    duplicate_encoding: Optional[bool] = field(
        default=False, metadata={"help": "how to encode computation graphs"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pooling: str = field(
        default='cls', metadata={"help": "Which pooling method to use (max, cls, attentive)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.duplicate_encoding is False and data_args.position_type != "no_position":
        raise ValueError(
                f"duplicate_encoding: {data_args.duplicate_encoding} and "
                + f"position_type: {data_args.position_type} cannot be set together"
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    if "t5" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    elif "opt" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    else:
        config = TDOConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir)

        model = TDOForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir)
        if data_args.position_type != "no_position":
            model.text_decoder.set_neighbor_position_ids(raw_dataset.position_ids)

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    prefix = "summarize: "
    def preprocess_function(examples):
        inputs = [prefix + section_text for section_text in examples["section_rest_sentence"]]
        model_inputs = tokenizer(inputs, padding='longest', max_length=1024, truncation=True, return_tensors="pt")

        text_target = examples["section_clean_1st_sentence"]
        labels = tokenizer(text_target, padding='longest', max_length=128, truncation=True, return_tensors="pt")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = load_dataset("parquet", data_files={"train": f"./wikiweb2m/raw/{data_args.task}_train.parquet",
                                                "val": f"./wikiweb2m/raw/{data_args.task}_val.parquet",
                                                "test": f"./wikiweb2m/raw/{data_args.task}_test.parquet"})

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1024,
            writer_batch_size=1024,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_names={"train": f"wikiweb2m/cache/{data_args.task}_{data_args.context}_train",
                                "val": f"wikiweb2m/cache/{data_args.task}_{data_args.context}_val",
                                "test": f"wikiweb2m/cache/{data_args.task}_{data_args.context}_test"},
            desc="Preprocessing WikiWeb2M dataset",
        )

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def compute_metrics1(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        cider_scorer = Cider()
        cands = {idx: [pred] for idx, pred in enumerate(decoded_preds)}
        refs = {idx: [label] for idx, label in enumerate(decoded_labels)}
        cider_score, _ = cider_scorer.compute_score(refs, cands)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        gen_len = np.mean(prediction_lens)

        return {'bleu': bleu_results['bleu'], 'rouge': rouge_results['rougeL'], 'cider': cider_score, 'gen_len': gen_len}

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name_or_path)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["val"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics1,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(dataset["test"], metric_key_prefix="predict")

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # Wandb logging
    combined_args = {**vars(data_args), **vars(model_args)}
    wandb.config.update(combined_args)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params})
    wandb.config.update({"trainable_params": trainable_params})
    wandb.run.summary["Total Parameters"] = total_params
    wandb.run.summary["Trainable Parameters"] = trainable_params


if __name__ == "__main__":
    main()
