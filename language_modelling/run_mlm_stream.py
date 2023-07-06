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
""" Finetuning document classification models"""
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_metric
import numpy as np
from scipy.special import expit

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from model import TDOConfig, TDOForMaskedLM, TDOForSequenceClassification
from data import OAGDataset
from sklearn.metrics import f1_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_length: Optional[int] = field(
        default=64, metadata={"help": "The maximum total input sequence length after tokenization."}
    )
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

    train_ratio: Optional[float] = field(
        default=0.5, metadata={"help": "Ratio of the training set"}
    )
    eval_ratio: Optional[float] = field(
        default=0.1, metadata={"help": "Ratio of the evaluation set"}
    )
    predict_ratio: Optional[float] = field(
        default=0.4, metadata={"help": "Ratio of the prediction set"}
    )

    dataset: Optional[str] = field(
        default='oag', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_domain: Optional[str] = field(
        default='CS', metadata={"help": "The domain of OAG datasets"}
    )

    label_type: Optional[str] = field(
        default='L1', metadata={"help": "Label type for the node classification."}
    )
    sample_depth: Optional[int] = field(
        default=2, metadata={"help": "neighborhood hops for the computation graph sampling."}
    )
    sample_num: Optional[int] = field(
        default=2, metadata={"help": "number of neighbors to sample per node."}
    )

    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pooling: str = field(
        default='max', metadata={"help": "Which pooling method to use (max, cls, attentive)."}
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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

    raw_dataset = OAGDataset(data_args)
    dataset = raw_dataset.load_dataset()

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if "text-decoder-only" in model_args.model_name_or_path:
        #tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
        #model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
        config = TDOConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = TDOForMaskedLM.from_pretrained(model_args.model_name_or_path)

    # Preprocessing the datasets
    def preprocess_function(examples):
        #batch = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=data_args.max_length)
        #return batch

        batch = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
                )

        encoder_hidden_states = []
        encoder_attention_mask = []
        labels = []
        for seed_id in examples["id"]:
            outputs = raw_dataset.sample_computation_graph(seed_id)
            encoder_hidden_states.append(outputs['feats'])
            encoder_attention_mask.append(outputs['attention_mask'])
            labels.append(outputs['label'])

        batch['encoder_hidden_states'] = encoder_hidden_states
        batch['encoder_attention_mask'] = encoder_attention_mask
        batch['labels'] = labels
        return batch


    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=256,
            writer_batch_size=256,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_name=f"oag_{data_args.sample_depth}_{data_args.sample_num}",
            desc="Sampling computation graphs on dataset",
            #num_proc=30
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(dataset)), 3):
        logger.info(f"Sample {index} of the dataset: {dataset[index]}.")

    # Use prediction set during pretraining
    train_samples = int(data_args.train_ratio * len(dataset)) +  int(data_args.predict_ratio * len(dataset))
    train_dataset = dataset.select(range(train_samples))

    eval_samples = int(data_args.eval_ratio * len(dataset))
    eval_dataset = dataset.select(range(train_samples, train_samples + eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=data_args.max_length,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
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

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset is not None:
        kwargs["dataset_tags"] = data_args.dataset
        if data_args.dataset_domain is not None:
            kwargs["dataset_args"] = data_args.dataset_domain
            kwargs["dataset"] = f"{data_args.dataset} {data_args.dataset_domain}"
        else:
            kwargs["dataset"] = data_args.dataset

    # Needed to avoid TPU hanging
    sys.exit()

if __name__ == "__main__":
    main()
