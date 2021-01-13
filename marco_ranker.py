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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

# from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from examples.run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)

_softmax = nn.Softmax()


def load_ranker(args):
    # if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    with torch.no_grad():
        model = BertForSequenceClassification.from_pretrained(args.ranker_model, num_labels=2)

        #load ranker tokenizer
        ranker_tokenizer = BertTokenizer.from_pretrained(args.ranker_model)#, do_lower_case=args.do_lower_case)


        if args.fp16:
            model.half()
        model.to(args.device)

        # if args.local_rank != -1:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
        #                                                       output_device=args.local_rank)
        # elif args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        model.eval()
        return model, ranker_tokenizer


def run_eval(model, input_ids, segment_ids, input_mask, args):
    # logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
    ### Evaluation
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        prob = _softmax(logits.item())

        return prob


def convert_ranker_features(tokenizer, query, passage, max_seq_length, device):
    tokens_a = tokenizer.tokenize(query)
    tokens_b = tokenizer.tokenize(passage)
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    segment_ids = torch.LongTensor(segment_ids).to(device).unsqueeze(0)
    input_mask = torch.LongTensor(input_mask).to(device).unsqueeze(0)

    return input_ids, segment_ids, input_mask

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


if __name__ == "__main__":
    load_ranker()
