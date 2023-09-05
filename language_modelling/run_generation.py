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
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from dataclasses import dataclass, field
import time
from time import perf_counter
import tqdm
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
from torchmetrics import BLEUScore
from warmup_scheduler import GradualWarmupScheduler

from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    HfArgumentParser,
    set_seed,
    get_scheduler,
)
from transformers.optimization import Adafactor
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import evaluate
from wikiweb2m import load_wikiweb2m, WikiWeb2M
from wikiweb2m.cider import Cider

from language_modelling import utils
from model import T5Image
#from model import TDOConfig, TDOForMaskedLM, TDOForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

best_acc1 = 0  # Variable to keep track of best model so far.

@dataclass
class Arguments:
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
    log_dir: Optional[str] = field(
        default='log', metadata={"help": "logging dir"}
    )
    resume: Optional[str] = field(
        default=None, metadata={"help": "path to latest checkpoint (default: none)"}
    )

    seed: Optional[int] = field(
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

    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per device during training."}
    )
    per_device_val_batch_size: Optional[int] = field(
        default=4, metadata={"help": "Batch size per device during evaluation/test."}
    )
    dataloader_num_workers: Optional[int] = field(
        default=4, metadata={"help": "Number of threads to read data."}
    )

    start_epoch: Optional[int] = field(
        default=0, metadata={"help": "Starting epoch."}
    )
    epochs: Optional[int] = field(
        default=90, metadata={"help": "Total number of epochs."}
    )
    steps_per_epoch: Optional[int] = field(
        default=2000, metadata={"help": "Number of training steps per epoch."}
    )
    val_steps_per_epoch: Optional[int] = field(
        default=1000, metadata={"help": "Number of training steps per epoch."}
    )
    save_epoch: Optional[int] = field(
        default=1, metadata={"help": "Starting epoch."}
    )
    print_freq: Optional[int] = field(
        default=50, metadata={"help": "print frequency (default: 10)"}
    )


    learning_rate: Optional[float] = field(
        default=0.001, metadata={"help": "initial learning rate."}
    )
    adam_beta1: Optional[float] = field(
        default=0.9, metadata={"help": "beta1 for Adam."}
    )
    adam_beta2: Optional[float] = field(
        default=0.95, metadata={"help": "beta2 for AdamDecay."}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Weight decay parameter."}
    )
    grad_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "number of gradient accumulation steps."}
    )
    grad_clip: Optional[float] = field(
        default=1.0, metadata={"help": "gradient clipping amount."}
    )
    lr_warmup_steps: Optional[int] = field(
        default=2000, metadata={"help": "Number of steps to warm up lr."}
    )
    lr_schedule_step_size: Optional[int] = field(
        default=5, metadata={"help": "Number of steps before decaying lr."}
    )
    lr_schedule_gamma: Optional[float] = field(
        default=0.1, metadata={"help": "Decay parameter for learning rate scheduler."}
    )

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    decoder_only: Optional[bool] = field(
        default=False, metadata={"help": "how to encode computation graphs"}
    )
    text_model: str = field(
        default="t5-base", metadata={"help": "text model to encode neighbor texts"}
    )
    visual_model: str = field(
        default="openai/clip-vit-base-patch16", metadata={"help": "visual model to encode neighbor images"}
    )
    n_visual_tokens: int = field(
        default=4, metadata={"help": "visual model to encode neighbor images"}
    )
    freeze_lm: Optional[bool] = field(
        default=False, metadata={"help": "evaluate model on validation set."}
    )


def main():

    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.duplicate_encoding is False and args.position_type != "no_position":
        raise ValueError(
                f"duplicate_encoding: {args.duplicate_encoding} and "
                + f"position_type: {args.position_type} cannot be set together"
            )

    i = 0
    log_dir = os.path.join(args.log_dir, args.wandb_run)
    while os.path.exists(log_dir):
        i += 1
        log_dir = os.path.join(args.log_dir, f'{args.wandb_run}_{i}')
    os.makedirs(log_dir)

    combined_args = {**vars(args), **vars(args), **vars(args)}
    with open(os.path.join(log_dir, f'args.json'), 'w') as wf:
        json.dump(combined_args, wf, indent=4)

    # Wandb logging
    run = wandb.init(project=args.wandb_project, name=f'{args.wandb_run}_{i}')
    run.config.update(combined_args)

    print(f'Logging to {log_dir}.')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')

    # Prepare distributed data parallel
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, log_dir, run))


def main_worker(gpu, world_size, args, log_dir, run):
    global best_acc1
    print("Use GPU: {} for training".format(gpu))
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1337', world_size=world_size, rank=gpu)

    # Prepare pretrained model
    if args.context in ("section_all", "all"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        model = T5Image(args, tokenizer)
    elif "t5" in args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    elif "opt" in args.model_name_or_path:
        args.decoder_only = True
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        config = TDOConfig.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir)
        model = TDOForMaskedLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir)
        if args.position_type != "no_position":
            model.text_decoder.set_neighbor_position_ids(raw_dataset.position_ids)

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    if args.fp16:
        model = model.float()
    elif args.bf16:
        model = model.bfloat16()

    param_counts_text = utils.get_params_count_str(model)
    with open(os.path.join(log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)

    # Wandb logging
    if gpu % world_size == 0:
        _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
        run.watch(model)
        run.config.update({"total_params": total_trainable_params + total_nontrainable_params})
        run.config.update({"trainable_params": total_trainable_params})
        run.config.update({"non_trainable_params": total_nontrainable_params})


    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    if "t5" in args.model_name_or_path:
        print('Using Adafactor as the optimizer.')
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
        scheduler = None
    elif "opt" in args.model_name_or_path:
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(model.parameters(), args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.weight_decay, eps=1e-8)
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        scheduler_steplr = StepLR(optimizer, step_size=(args.lr_schedule_step_size * args.steps_per_epoch) // args.grad_accumulation_steps, gamma=args.lr_schedule_gamma)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)

    # Detecting last checkpoint.
    if args.resume:
        checkpoint_path = os.path.join(args.log_dir, args.resume, 'ckpt.pth.tar')
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc {})".format(checkpoint_path, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True

    # Prepare Dataset
    start_time = perf_counter()
    train_data, val_data, test_data, id_list = load_wikiweb2m(args.task)
    print(f'Loading wikiweb2m done: {perf_counter()-start_time}')
    start_time = perf_counter()
    train_dataset = WikiWeb2M(args, train_data, id_list["train"], tokenizer, args.visual_model)
    val_dataset = WikiWeb2M(args, val_data, id_list["val"], tokenizer, args.visual_model)
    test_dataset = WikiWeb2M(args, test_data, id_list["test"], tokenizer, args.visual_model)
    print(f'Initialize datasets: {perf_counter()-start_time}')
    print(f'Training with {len(train_dataset)} examples, validating with {len(val_dataset)} examples, testing with {len(test_dataset)} examples.')

    # Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)

    # Dataloader
    start_time = perf_counter()
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
            shuffle=False, num_workers=args.dataloader_num_workers, prefetch_factor=10, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_val_batch_size,
            shuffle=False, num_workers=args.dataloader_num_workers, prefetch_factor=10, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.per_device_val_batch_size,
            shuffle=False, num_workers=args.dataloader_num_workers, prefetch_factor=10, pin_memory=True, sampler=test_sampler)
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
    #    model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True)
    #    labels = tokenizer(text_target=targets, max_length=args.max_output_length, padding="max_length", truncation=True)

    #    labels["input_ids"] = [
    #        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #    ]
    #    model_inputs["labels"] = labels["input_ids"]
    #    return model_inputs

    #train_dataset = raw_datasets["train"].map(
    #    preprocess_function,
    #    batched=True,
    #    num_proc=args.dataloader_num_workers,
    #    remove_columns=column_names,
    #    load_from_cache_file=True,
    #    desc="Running tokenizer on dataset",
    #)

    #max_target_length = args.max_output_length
    #eval_dataset = raw_datasets["validation"].map(
    #    preprocess_function,
    #    batched=True,
    #    num_proc=args.dataloader_num_workers,
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
    #    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    #)
    #val_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.test:
        evaluate_loop(test_loader, model, tokenizer, epoch, args, run)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #if epoch == 0:
        #    evaluate_loop(val_loader, model, tokenizer, epoch-1, args, run)

        train_sampler.set_epoch(epoch)
        # train for one epoch
        train_loop(train_loader, model, tokenizer, optimizer, epoch, scheduler, args, run)
        # evaluate on validation set
        acc1 = evaluate_loop(val_loader, model, tokenizer, epoch, args, run)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if gpu % world_size == 0 and (is_best or epoch == 0):
            # Only save non-frozen parameters.
            #stripped_state_dict = {
            #    k: v for k, v in model.state_dict().items() if
            #    ('.lm' not in k and '.visual_model' not in k)
            #}
            #stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
            state = {
                'epoch': args.epochs,
                'best_acc1': acc1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            if scheduler is not None:
                state['scheduler'] = scheduler.state_dict()
            print('=> save best val model ...', os.path.join(log_dir, 'ckpt.pth.tar'))
            torch.save(state, os.path.join(log_dir, 'ckpt.pth.tar'))
    # Test
    checkpoint_path = os.path.join(log_dir, 'ckpt.pth.tar')
    print("=> loading best val checkpoint '{}'".format(checkpoint_path))
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded best val checkpoint '{}'".format(checkpoint_path))
    evaluate_loop(test_loader, model, tokenizer, args.epochs, args, run, "test")

def train_loop(train_loader, model, tokenizer, optimizer, epoch, scheduler, args, run):
    gpu, world_size = dist.get_rank(), dist.get_world_size()
    ngpus_per_node = torch.cuda.device_count()

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    forward_time = utils.AverageMeter('Forward', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')

    if gpu % world_size == 0:
        progress = utils.ProgressMeter(args.steps_per_epoch, [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    if args.decoder_only:
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items()}
        forward_start = time.time()
        outputs = model(**batch)
        forward_time.update(time.time() - forward_start)

        loss = outputs.loss
        if args.decoder_only:
            logits = outputs.logits
            # only consider loss on reference summary just like seq2seq models
            shift_logits = logits[..., args.max_input_length:-1, :].contiguous()
            shift_labels = batch['labels'][..., (args.max_input_length + 1):].contiguous()
            # summary_loss
            #summary_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            #losses.update(summary_loss.item(), batch["input_ids"].size(0))
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses.update(loss.item(), batch["input_ids"].size(0))
        else:
            losses.update(loss.item(), batch["input_ids"].size(0))
        loss = loss / args.grad_accumulation_steps
        loss.backward()

        # Update weights
        if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                if args.grad_clip > 2:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.zero_grad()

            actual_step = (epoch * args.steps_per_epoch + i + 1) // args.grad_accumulation_steps
            if actual_step == 1 or actual_step % args.print_freq == 0:
                losses.all_reduce()
                batch_time.all_reduce()
                data_time.all_reduce()
                forward_time.all_reduce()
                ex_per_sec = (args.per_device_train_batch_size / batch_time.avg) * ngpus_per_node

                if gpu % world_size == 0:
                    progress.display(i + 1)
                    #curr_lr = scheduler.get_last_lr()
                    #run.log({"train/lr": curr_lr[0]}, step=actual_step)
                    run.log({"train/loss": losses.avg}, step=actual_step)
                    run.log({"metrics/total_secs_per_batch": batch_time.avg}, step=actual_step)
                    run.log({"metrics/data_secs_per_batch": data_time.avg}, step=actual_step)
                    run.log({"metrics/total_secs_captioning": forward_time.avg}, step=actual_step)
                    run.log({"metrics/examples_per_sec": ex_per_sec}, step=actual_step)

                losses.reset()
                batch_time.reset()
                data_time.reset()
                forward_time.reset()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == args.steps_per_epoch - 1:
            break


# Evaluate loop
def evaluate_loop(val_loader, model, tokenizer, epoch, args, run, prefix="val"):
    gpu, world_size = dist.get_rank(), dist.get_world_size()
    ngpus_per_node = torch.cuda.device_count()
    bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3, 4]]
    actual_step = ((epoch + 1) * args.steps_per_epoch) // args.grad_accumulation_steps

    batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.AVERAGE)
    losses = utils.AverageMeter('Loss', ':.4e', utils.Summary.AVERAGE)
    bleu1 = utils.AverageMeter('BLEU@1', ':6.2f', utils.Summary.AVERAGE)
    bleu2 = utils.AverageMeter('BLEU@2', ':6.2f', utils.Summary.AVERAGE)
    bleu3 = utils.AverageMeter('BLEU@3', ':6.2f', utils.Summary.AVERAGE)
    bleu4 = utils.AverageMeter('BLEU@4', ':6.2f', utils.Summary.AVERAGE)

    if gpu % world_size == 0:
        progress = utils.ProgressMeter(args.val_steps_per_epoch, [batch_time, losses], prefix=f'{prefix}: ')

    if args.decoder_only:
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        all_generated_captions = []
        all_gt_captions = []
        max_to_display = 5

        for i, batch in enumerate(val_loader):
            batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            if args.decoder_only:
                # only consider loss on reference summary just like seq2seq models
                logits = logits[..., args.max_input_length:-1, :].contiguous()
                labels = batch['labels'][..., (args.max_input_length + 1):].contiguous()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                labels = batch['labels']
                loss = outputs.loss
            losses.update(loss.item(), batch["input_ids"].size(0))

            if prefix == "test":
                if args.decoder_only:
                    generated_ids = model.module.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=32)
                else:
                    generated_ids = model.module.generate(input_ids=batch["input_ids"][..., :args.max_input_length, :].contiguous(), \
                                                    attention_mask=batch["attention_mask"][..., :args.max_input_length, :].contiguous(), max_new_tokens=32)
            else:
                generated_ids = torch.argmax(logits, dim=-1)

            all_generated_ids = [torch.zeros_like(generated_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_generated_ids, generated_ids)
            all_generated_ids[dist.get_rank()] = generated_ids
            generated_ids = torch.cat(all_generated_ids)

            tgt_tokens = labels
            all_tgt_tokens = [torch.zeros_like(tgt_tokens) for _ in range(dist.get_world_size())]
            dist.all_gather(all_tgt_tokens, tgt_tokens)
            all_tgt_tokens[dist.get_rank()] = tgt_tokens
            all_tgt_tokens = torch.cat(all_tgt_tokens)

            if not args.decoder_only:
                all_tgt_tokens[all_tgt_tokens == -100] = tokenizer.pad_token_id
            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gt_captions = tokenizer.batch_decode(all_tgt_tokens, skip_special_tokens=True)

            for cap_i in range(len(generated_captions)):
                stop_idx = generated_captions[cap_i].find('.')
                if stop_idx > 5:
                    all_generated_captions.append(generated_captions[cap_i][:stop_idx])
                else:
                    all_generated_captions.append(generated_captions[cap_i])
                all_gt_captions.append([gt_captions[cap_i]])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and  gpu % world_size == 0:
                progress.display(i + 1)

            if i == args.val_steps_per_epoch - 1:
                break

        if gpu % world_size == 0:
            print('=' * 30)
            print(f'Computing BLEU with {len(all_generated_captions)} generated captions and {len(all_gt_captions)} groundtruth captions.')
            for cap_i, cap in enumerate(all_generated_captions[:max_to_display]):
                print(f'{cap_i}) {cap}')
            print('=' * 30)
            print('Real samples:')
            for cap_i, cap in enumerate(all_gt_captions[:max_to_display]):
                print(f'{cap_i}) {cap}')
            print('=' * 30)

        #utils.postprocess_text(all_generated_captions, all_gt_captions)

        bleu1_score = bleu_scorers[0](all_generated_captions, all_gt_captions)
        bleu1.update(bleu1_score, 1)
        bleu2_score = bleu_scorers[1](all_generated_captions, all_gt_captions)
        bleu2.update(bleu2_score, 1)
        bleu3_score = bleu_scorers[2](all_generated_captions, all_gt_captions)
        bleu3.update(bleu3_score, 1)
        bleu4_score = bleu_scorers[3](all_generated_captions, all_gt_captions)
        bleu4.update(bleu4_score, 1)

    batch_time.all_reduce()
    losses.all_reduce()
    bleu1.all_reduce()
    bleu2.all_reduce()
    bleu3.all_reduce()
    bleu4.all_reduce()

    if gpu % world_size == 0:
        progress.display_summary()
        print(bleu1.avg, bleu2.avg, bleu3.avg, bleu4.avg)

        run.log({f"{prefix}/total_secs_per_batch": batch_time.avg}, step=actual_step)
        run.log({f"{prefix}/loss": losses.avg}, step=actual_step)
        run.log({f"{prefix}/bleu1": bleu1.avg}, step=actual_step)
        run.log({f"{prefix}/bleu2": bleu2.avg}, step=actual_step)
        run.log({f"{prefix}/bleu3": bleu3.avg}, step=actual_step)
        run.log({f"{prefix}/bleu4": bleu4.avg}, step=actual_step)

    return bleu4.avg
    #rouge = evaluate.load("rouge")
    #bleu = evaluate.load("bleu")
    #cider_scorer = Cider()

    #cands = {idx: [pred] for idx, pred in enumerate(all_generated_texts)}
    #refs = {idx: [label] for idx, label in enumerate(all_labels)}
    #cider_score, _ = cider_scorer.compute_score(refs, cands)
    #rouge_results = rouge.compute(predictions=all_generated_texts, references=all_labels)
    #bleu_results = bleu.compute(predictions=all_generated_texts, references=all_labels)


if __name__ == "__main__":
    main()
