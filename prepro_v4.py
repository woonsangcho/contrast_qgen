"""
preprocess input data into feature
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from pytorch_pretrained_bert import GPT2Tokenizer
from tqdm import tqdm

from env import END_OF_TEXT_TOKEN


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 lm_labels, weights, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.weights = weights
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


def _get_file_len(corpus):
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line


def _norm_text(text):
    w, *toks = text.strip().split()
    try:
        w = float(w)
    except:
        toks = [w] + toks
        w = 1.0
    return w, ' '.join(toks)


def _get_inputs_from_text(text, tokenizer):
    srcs, tgt = text.strip().split('\t')
    weights = []
    inputs = []
    for src in srcs.split(' EOS '):
        src_weight, src = _norm_text(src)
        context_id = tokenizer.encode(src)
        weights.append(src_weight)
        inputs.append(context_id)
    tgt_weight, tgt = _norm_text(tgt)
    if tgt_weight != 0:
        response_id = tokenizer.encode(tgt)
        weights.append(tgt_weight)
        inputs.append(response_id)
    return weights, inputs



