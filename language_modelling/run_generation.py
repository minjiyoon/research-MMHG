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
import json
import sys
import wandb
import warnings
from dataclasses import dataclass, field
import time
from time import perf_counter
from tqdm.auto import tqdm
from typing import Optional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only display errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import evaluate
from wikiweb2m import load_wikiweb2m, WikiWeb2M
from wikiweb2m.cider import Cider

from language_modelling import utils
#from model import TDOConfig, TDOForMaskedLM, TDOForSequenceClassification

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

best_acc1 = 0  # Variable to keep track of best model so far.

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
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
    sample_num: optional[int] = field(
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
    log_dir: Optional[str] = field(
        default='log', metadata={"help": "logging dir"}
    )
    resume: Optional[str] = field(
        default=None, metadata={"help": "path to latest checkpoint (default: none)"}
    )

@dataclass
class TrainingArguments:
    seed: optional[int] = field(
        default=None, metadata={"help": "seed for initializing training."}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "What precision to train in."}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "What precision to train in."}
    )

    test: Optional[bool] = field(
        default=False, metadata={"help": "evaluate model on validation set."}
    )

    per_device_train_batch_size: optional[int] = field(
        default=4, metadata={"help": "Batch size per device during training."}
    )
    per_device_val_batch_size: optional[int] = field(
        default=4, metadata={"help": "Batch size per device during evaluation/test."}
    )
    dataloader_num_workers: optional[int] = field(
        default=4, metadata={"help": "Number of threads to read data."}
    )

    start_epoch: optional[int] = field(
        default=0, metadata={"help": "Starting epoch."}
    )
    epochs: optional[int] = field(
        default=90, metadata={"help": "Total number of epochs."}
    )
    steps_per_epoch: optional[int] = field(
        default=2000, metadata={"help": "Number of training steps per epoch."}
    )
    save_epoch: optional[int] = field(
        default=10, metadata={"help": "Starting epoch."}
    )
    print_freq: optional[int] = field(
        default=10, metadata={"help": "print frequency (default: 10)"}
    )


    learning_rate: optional[float] = field(
        default=0.0001, metadata={"help": "initial learning rate."}
    )
    adam_beta1: optional[float] = field(
        default=0.9, metadata={"help": "beta1 for Adam."}
    )
    adam_beta2: optional[float] = field(
        default=0.95, metadata={"help": "beta2 for AdamDecay."}
    )
    weight_decay: optional[float] = field(
        default=0.01, metadata={"help": "Weight decay parameter."}
    )
    grad_accumulation_steps: optional[int] = field(
        default=4, metadata={"help": "number of gradient accumulation steps."}
    )
    grad_clip: optional[float] = field(
        default=1.0, metadata={"help": "gradient clipping amount."}
    )
    lr_warmup_steps: optional[int] = field(
        default=2000, metadata={"help": "Number of steps to warm up lr."}
    )
    lr_schedule_step_size: optional[int] = field(
        default=5, metadata={"help": "Number of steps before decaying lr."}
    )
    lr_schedule_gamma: optional[float] = field(
        default=0.1, metadata={"help": "Decay parameter for learning rate scheduler."}
    )
    lr_warmup_steps: optional[int] = field(
        default=2000, metadata={"help": "Number of steps to warm up lr."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    text_model: str = field(
        default="t5-base", metadata={"help": "text model to encode neighbor texts"}
    )
    visual_model: str = field(
        default="openai/clip-vit-base-patch16", metadata={"help": "visual model to encode neighbor images"}
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    if data_args.duplicate_encoding is False and data_args.position_type != "no_position":
        raise ValueError(
                f"duplicate_encoding: {data_args.duplicate_encoding} and "
                + f"position_type: {data_args.position_type} cannot be set together"
            )

    log_dir = os.path.join(data_args.log_dir, data_args.wandb_name)
    while os.path.exists(log_dir):
        log_dir = os.path.join(data_args.log_dir, f'{data_args.wandb_name}_{i}')
        i += 1
    os.makedirs(log_dir)

    combined_args = {**vars(data_args), **vars(model_args), **vars(train_args)}
    with open(os.path.join(log_dir, f'args.json'), 'w') as wf:
        json.dump(vars(combined_args), wf, indent=4)

    # Wandb logging
    wandb.init(project=data_args.wandb_project, name=data_args.wandb_run)
    wandb.config.update(combined_args)

    print(f'Logging to {log_dir}.')

    if train_args.seed is not None:
        random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')

    # Prepare distributed data parallel
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, data_args, model_args, train_args, log_dir))


def main_worker(gpu, world_size, data_args, model_args, train_args, log_dir):
    global best_acc1
    print("Use GPU: {} for training".format(gpu))
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1337', world_size=world_size, rank=gpu)

    # Prepare pretrained model
    if "t5" in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)
    elif "opt" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
        config = TDOConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir)
        model = TDOForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir)
        if data_args.position_type != "no_position":
            model.text_decoder.set_neighbor_position_ids(raw_dataset.position_ids)

    #tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    if train_args.fp16:
        model = model.float()
    elif train_args.bf16:
        model = model.bfloat16()

    param_counts_text = utils.get_params_count_str(model)
    with open(os.path.join(log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)

    # Wandb logging
    if gpu % world_size == 0
        _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
        wandb.watch(model)
        wandb.config.update({"total_params": total_trainable_params + total_nontrainable_params})
        wandb.config.update({"trainable_params": total_trainable_params})
        wandb.config.update({"non_trainable_params": total_nontrainable_params})


    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer_cls = torch.optim.AdamW
    print('Using torch.optim.AdamW as the optimizer.')
    optimizer = optimizer_cls(model.parameters(), train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            weight_decay=train_args.weight_decay, eps=1e-8)

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    scheduler_steplr = StepLR(optimizer, step_size=data_args.lr_schedule_step_size * data_args.steps_per_epoch, gamma=data_args.lr_schedule_gamma)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=data_args.lr_warmup_steps, after_scheduler=scheduler_steplr)

    # Detecting last checkpoint.
    if data_args.resume:
        if os.path.isfile(data_args.resume):
            print("=> loading checkpoint '{}'".format(data_args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(data_args.resume, map_location=loc)
            train_args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(data_args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(data_args.resume))

    cudnn.benchmark = True

    # Prepare Dataset
    start_time = perf_counter()
    train_data, val_data, test_data = load_wikiweb2m(data_args.task)
    print(f'Loading wikiweb2m done: {perf_counter()-start_time}')
    start_time = perf_counter()
    train_dataset = WikiWeb2M(data_args, train_data, tokenizer, model_args.visual_model)
    val_dataset = WikiWeb2M(data_args, val_data, tokenizer, model_args.visual_model)
    test_dataset = WikiWeb2M(data_args, test_data, tokenizer, model_args.visual_model)
    print(f'Initialize datasets: {perf_counter()-start_time}')
    print(f'Training with {len(train_dataset)} examples, validating with {len(val_dataset)} examples, testing with {len(test_dataset)} examples.')

    # Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)

    # Dataloader
    start_time = perf_counter()
    train_loader = DataLoader(train_dataset, batch_size=train_args.per_device_train_batch_size,
            shuffle=False, num_workers=train_args.dataloader_num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=train_args.per_device_val_batch_size,
            shuffle=False, num_workers=train_args.dataloader_num_workers, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=train_args.per_device_val_batch_size,
            shuffle=False, num_workers=train_args.dataloader_num_workers, pin_memory=True, sampler=test_sampler)
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

    if train_args.test:
        evaluate_loop(test_loader, model, tokenizer, criterion, epoch, args)
        return

    for epoch in range(train_args.start_epoch, train_args.epochs):
        if epoch == 0:
            evaluate_loop(val_loader, model, tokenizer, criterion, epoch-1, args)

        train_sampler.set_epoch(epoch)
        # train for one epoch
        train_loop(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)
        # evaluate on validation set
        acc1 = evaluate_loop(val_loader, model, tokenizer, criterion, epoch, args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
i
        if (epoch % save_epoch == 0 or is_best) and gpu % world_size == 0:
            # Only save non-frozen parameters.
            #stripped_state_dict = {
            #    k: v for k, v in model.state_dict().items() if
            #    ('.lm' not in k and '.visual_model' not in k)
            #}
            #stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
            utils.save_checkpoint({
                'epoch': epoch + 1,
                #'state_dict': stripped_state_dict,
                'best_acc1': best_acc1,
                #'optimizer' : optimizer.state_dict(),
                #'scheduler' : scheduler.state_dict()
            }, is_best, os.path.join(log_dir, 'ckpt'))


def train_loop(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)
    ngpus_per_node = torch.cuda.device_count()

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    forward_time = utils.AverageMeter('Forward', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')

    progress = utils.ProgressMeter(train_args.steps_per_epoch, [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, batch in enumerate(train_dataloader):
        actual_step = epoch * train_args.steps_per_epoch + i + 1
        data_time.update(time.time() - end)

        batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items()}

        forward_start = time.time()
        outputs = model(**batch)
        forward_time.update(time.time() - forward_start)

        loss = outputs.loss
        loss = loss / train_args.grad_accumulation_steps
        losses.update(loss.item(), batch["input_ids"].size(0))
        loss.backward()

        # Update weights
        if ((i + 1) % train_args.grad_accumulation_steps == 0) or (i == train_args.steps_per_epoch - 1):
            if train_args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            print('=' * 80)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if actual_step == 1 or (i + 1) % train_args.print_freq == 0:
            losses.all_reduce()
            batch_time.all_reduce()
            data_time.all_reduce()
            forward_time.all_reduce()
            ex_per_sec = (train_args.per_device_train_batch_size / batch_time.avg) * ngpus_per_node

            progress.display(i + 1)

            if gpu % world_size == 0:
                wandb.log({"train/loss": losses.avg}, step=actual_step)
                wandb.log({"metrics/total_secs_per_batch": batch_time.avg}, step=actual_step)
                wandb.log({"metrics/data_secs_per_batch": data_time.avg}, step=actual_step)
                wandb.log({"metrics/total_secs_captioning": forward_time.avg}, step=actual_step)
                wandb.log({"metrics/examples_per_sec": ex_per_sec}, step=actual_step)

            losses.reset()
            batch_time.reset()
            data_time.reset()
            forward_time.reset()

        if i == train_args.steps_per_epoch - 1:
            break

        lr_scheduler.step()
        curr_lr = scheduler.get_last_lr()
        if actual_step == 1 or (i + 1) % train_args.print_freq == 0:
            if gpu % world_size == 0:
                wandb.log({"train/lr": curr_lr[0]}, step=actual_step)


# Evaluate loop
def evaluate_loop(model, dataloader, prefix="eval"):
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
    #wandb.log(results)
    print(results)

if __name__ == "__main__":
    main()
