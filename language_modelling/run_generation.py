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
import os
import sys
from time import perf_counter
from tqdm.auto import tqdm

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import evaluate
from wikiweb2m import load_wikiweb2m, WikiWeb2M
from wikiweb2m.cider import Cider

#from model import TDOConfig, TDOForMaskedLM, TDOForSequenceClassification

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

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

    dataset: Optional[str] = field(
        default='wikiweb2m', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task: Optional[str] = field(
        default='section_summarization', metadata={"help": "The domain of OAG datasets"}
    )
    context: Optional[str] = field(
        default='section_only', metadata={"help": "The domain of OAG datasets"}
    )
    max_input_length: Optional[int] = field(
        default=512, metadata={"help": "maximum token length of input text"}
    )
    max_output_length: Optional[int] = field(
        default=128, metadata={"help": "maximum token length of output text"}
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

    wandb_project: Optional[str] = field(
        default='MMHG', metadata={"help": "wandb project name"}
    )
    wandb_run: Optional[str] = field(
        default='default', metadata={"help": "wandb run name"}
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
    # See all possible arguments in src/transformers/train_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

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

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}"
        + f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(train_args.seed)
    # Wandb logging
    wandb.init(project=data_args.wandb_project, name=data_args.wandb_run)
    combined_args = {**vars(data_args), **vars(model_args), **vars(train_args)}
    wandb.config.update(combined_args)

    # Prepare pretrained model
    if "t5" in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)
    elif "opt" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        config = TDOConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir)
        model = TDOForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir)
        if data_args.position_type != "no_position":
            model.text_decoder.set_neighbor_position_ids(raw_dataset.position_ids)

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    # Wandb logging
    wandb.watch(model)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params})
    wandb.config.update({"trainable_params": trainable_params})

    if train_args.fp16:
        model = model.float()
    elif train_args.f16:
        model = model.bfloat16()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.nn.DataParallel(model).to(device)

    # Prepare Dataset
    start_time = perf_counter()
    train_data, val_data, test_data = load_wikiweb2m(data_args.task)
    print(f'Loading wikiweb2m done: {perf_counter()-start_time}')
    start_time = perf_counter()
    train_dataset = WikiWeb2M(data_args, train_data, tokenizer)
    val_dataset = WikiWeb2M(data_args, val_data, tokenizer)
    test_dataset = WikiWeb2M(data_args, test_data, tokenizer)
    print(f'Initialize datasets: {perf_counter()-start_time}')

    ngpus = torch.cuda.device_count()
    num_workers = train_args.dataloader_num_workers
    train_batch_size = train_args.per_device_train_batch_size #* ngpus
    val_batch_size = train_args.per_device_eval_batch_size #* ngpus

    #start_time = perf_counter()
    dataloader_params = {"num_workers": num_workers, "prefetch_factor": 10, "pin_memory": True, "shuffle": True, "drop_last": True}
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=train_batch_size, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate, batch_size=val_batch_size, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, collate_fn=test_dataset.collate, batch_size=val_batch_size, **dataloader_params)
    print(f'Initialize dataloaders: {perf_counter()-start_time}')


    # Example Dataset
    #raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
    #column_names = raw_datasets["train"].column_names
    #text_column = "article"
    #summary_column = "highlights"
    #prefix = "summarize: "

    #def preprocess_function(examples):
    #    inputs = examples[text_column]
    #    targets = examples[summary_column]
    #    inputs = [prefix + inp for inp in inputs]
    #    model_inputs = tokenizer(inputs, max_length=data_args.max_input_length, padding="max_length", truncation=True)
    #    labels = tokenizer(text_target=targets, max_length=data_args.max_output_length, padding="max_length", truncation=True)

    #    labels["input_ids"] = [
    #        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #    ]
    #    model_inputs["labels"] = labels["input_ids"]
    #    return model_inputs

    #train_dataset = raw_datasets["train"].map(
    #    preprocess_function,
    #    batched=True,
    #    num_proc=train_args.dataloader_num_workers,
    #    remove_columns=column_names,
    #    load_from_cache_file=True,
    #    desc="Running tokenizer on dataset",
    #)

    #max_target_length = data_args.max_output_length
    #eval_dataset = raw_datasets["validation"].map(
    #    preprocess_function,
    #    batched=True,
    #    num_proc=train_args.dataloader_num_workers,
    #    remove_columns=column_names,
    #    load_from_cache_file=True,
    #    desc="Running tokenizer on dataset",
    #)

    #label_pad_token_id = -100
    #data_collator = DataCollatorForSeq2Seq(
    #    tokenizer,
    #    model=model,
    #    label_pad_token_id=label_pad_token_id,
    #    pad_to_multiple_of=8,
    #)

    #train_dataloader = DataLoader(
    #    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_args.per_device_train_batch_size
    #)
    #val_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=train_args.per_device_eval_batch_size)
    train_args.max_steps = 3 * len(train_dataloader)
    print("TRAIN DATA LENGTH: ", len(train_dataloader))
    print("VAL DATA LENGTH:", len(val_dataloader))

    # Evaluate loop
    def evaluate_step(model, dataloader, prefix="eval"):
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        cider_scorer = Cider()

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        all_generated_texts = []
        all_labels = []
        total_loss = 0.
        step = 0
        eval_step = 1000
        progress_bar = tqdm(range(eval_step))
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if step == eval_step:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss.sum()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu()
                #predictions = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

                labels = batch["labels"].cpu()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                total_loss += loss.item()
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                all_generated_texts.extend(decoded_preds)
                all_labels.extend(decoded_labels)
                progress_bar.update(1)
                step += 1

        cands = {idx: [pred] for idx, pred in enumerate(all_generated_texts)}
        refs = {idx: [label] for idx, label in enumerate(all_labels)}
        cider_score, _ = cider_scorer.compute_score(refs, cands)
        rouge_results = rouge.compute(predictions=all_generated_texts, references=all_labels)
        bleu_results = bleu.compute(predictions=all_generated_texts, references=all_labels)

        results = {
                f'{prefix}_loss': total_loss / step / val_batch_size,
                f'{prefix}_bleu': bleu_results['bleu'],
                f'{prefix}_rouge': rouge_results['rougeL'],
                f'{prefix}_cider': cider_score,
            }
        wandb.log(results)
        print(results)

    # Prepare training
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            weight_decay=train_args.weight_decay, eps=1e-8)

    num_warmup_steps = int(train_args.warmup_ratio * train_args.max_steps)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=train_args.max_steps)

    # Training loop
    step = 0
    progress_bar = tqdm(range(train_args.max_steps))
    model.train()
    while step < train_args.max_steps:
        for batch in train_dataloader:
            if step == train_args.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            step += 1

            if step % train_args.logging_steps == 0:
                wandb.log({"train_loss_batch": loss.item()})
                print(f"[{step} steps] training loss: {loss}")
            if step % train_args.eval_steps == 0:
                evaluate_step(model, train_dataloader, prefix="train")
                evaluate_step(model, val_dataloader, prefix="eval")

    #evaluate_step(model, test_dataloader, prefix="test")


if __name__ == "__main__":
    main()
