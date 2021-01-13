

import os
import logging
import torch
import subprocess as sp
from collections import defaultdict
from math import ceil

from torch.nn.utils.rnn import pad_sequence
import numpy as np

from env import END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN
from optim import warmup_linear, noam_decay, noamwd_decay

from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

SEQ_LENGTH_SHRINK_PROP = 0.9

def load_model(model, checkpoint, args, verbose=False):
    with torch.no_grad():
        if checkpoint is None or checkpoint == "None":
            if verbose:
                logger.info('no checkpoint provided for %s!' % model._get_name())
        else:
            if not os.path.exists(checkpoint):
                raise ValueError('checkpoint %s not exist' % checkpoint)
            if verbose:
                logger.info('loading finetuned model from %s' % checkpoint)
            model_state_dict = torch.load(checkpoint)

            model_state_dict = fix_state_dict_namespace(model_state_dict)

            start_model = model
            if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in model_state_dict.keys()):
                logger.info('loading transfomer only')
                start_model = model.transformer
            start_model.load_state_dict(model_state_dict)

        if args.fp16:
            logger.info('in fp16, model.half() activated')
            model.half()

        model.to(args.device)

        return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

class RedditExample(object):
    def __init__(self, conv_id, context, response):
        self.conv_id = conv_id
        self.context = context
        self.response = response

    def __repr__(self):
        return 'conv_id = {}\ncontext = {}\nresponse = {}'.format(self.conv_id, self.context, self.response)

    def __str__(self):
        return self.__repr__()


class InputFeatures(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids, lm_labels, context_len, response_len):
        self.conv_id = conv_id
        self.choices_features = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        self.lm_labels = lm_labels
        self.context_len = context_len
        self.response_len = response_len    # in case we need it


class DynamicBatchingLoader(object):
    def __init__(self, corpus_file, tokenizer, normalize_data,
                 batch_size, max_seq_length, is_train):
        self.corpus = corpus_file
        self.toker = tokenizer
        self.norm = normalize_data
        self.bs = batch_size
        self.max_seq_length = max_seq_length
        self.train = is_train
        self.num_examples = self.get_len(corpus_file)

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples/self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as corpus:
                i = 0
                while True:
                    examples = []
                    cur_bs = 0
                    while True:
                        line = next(corpus).encode('utf-8').decode('utf-8')
                        contents = line.split('\t')
                        src, tgt_all = contents[0], contents[1:]
                        for tgt in tgt_all:
                            if self.norm:
                                src_line = ' '.join(src.strip().split())
                                tgt_line = ' '.join(tgt.strip().split())
                            else:
                                src_line = src.strip()
                                tgt_line = tgt.strip()
                            examples.append(
                                RedditExample(i, src_line, tgt_line),
                            )
                            i += 1
                            cur_bs += 1
                        if cur_bs >= self.bs:
                            break
                    if self.train:
                        features = convert_examples_to_features_dynamic(
                            examples, self.toker, self.max_seq_length)
                    else:
                        features = convert_examples_to_features_eval(
                            examples, self.toker, self.max_seq_length)
                    batch = self._batch_feature(features)
                    yield batch
        except StopIteration:
            pass

    def _batch_feature(self, features):
        input_ids = pad_sequence([torch.tensor(f.choices_features['input_ids'],
                                               dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(
            [torch.tensor(f.choices_features['token_type_ids'],
                          dtype=torch.long)
             for f in features],
            batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        context_len = torch.tensor([f.context_len for f in features],
                                   dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features],
                                    dtype=torch.long)
        return (input_ids, position_ids, token_type_ids, labels,
                context_len, response_len)

    def get_len(self, corpus):
        try:
            n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                     universal_newlines=True).split()[0])
        except:
            n_line = int(sp.check_output(f"find \"\" /c /v {corpus}", shell=True).decode().split()[-1])

        return n_line



def convert_examples_to_features_eval(examples, tokenizer, max_seq_length=512):
    """
    pad on the left
    """
    def get_len(example):
        context_id = tokenizer.encode(example.context)
        return len(context_id)+1

    def featurize(example, max_seq_len):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        # if context is too long, cut from the beginning
        if len(context_id) + 1 > max_seq_length:
            context_id = context_id[len(context_id)+1-max_seq_length:]

        # response is NOT provided in example
        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id  # don't need to do anything

        # if response is too long, cut from the end
        if len(lm_labels) + 1 > max_seq_length:
            lm_labels = lm_labels[:max_seq_length]

        position_ids = list(range(len(input_ids)))

        # pad on left
        pad_len = max_seq_len - len(input_ids)
        # print(len(input_ids), max_seq_len, pad_len)
        input_ids = [0] * pad_len + input_ids
        position_ids = [0] * pad_len + position_ids

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    max_seq_length_tmp = max(map(get_len, examples))
    max_seq_length = min(max_seq_length, max_seq_length_tmp)
    features = [featurize(ex, max_seq_length) for ex in examples]

    return features


def convert_examples_to_features_dynamic(examples, tokenizer, max_seq_length = 512):
    """
    do not pad
    """
    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        # response is provided in example
        response_id = tokenizer.encode(example.response)

        input_ids_len = len(context_id) + len(response_id) + 2
        # print('max_seq_length = %d' % max_seq_length)
        # print('context_len = %d, response_len = %d, total_len = %d' % (len(context_id), len(response_id), input_ids_len))
        if input_ids_len > max_seq_length:
            if len(context_id) > input_ids_len - max_seq_length:
                # cut context from beginning if length of context + response is too long
                # and len of context is long enough to cut
                context_id = context_id[input_ids_len - max_seq_length:]
            else:
                # cut response from end if length of context + response is too long
                # and len of response is long enough to cut
                # if no response is available, discard the data
                if max_seq_length-len(context_id)-2 < 0:
                    # print('discard')
                    return None
                response_id = response_id[:max_seq_length-len(context_id)-2]

        input_ids = context_id + [end_of_text_id] + response_id + [end_of_text_id]
        # print('context_len = %d, response_len = %d, total_len = %d' % (len(context_id), len(response_id), len(input_ids)), '\n')

        # label simplely is next token in sequences. MASK all context_id tokens except for the last one
        lm_labels = [-1] * len(context_id) + response_id + [end_of_text_id] + [-1]

        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)


        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    # discard None feature
    features = [f for f in [featurize(ex) for ex in examples] if f is not None]
    return features


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_eval_list(input_file, tokenizer, max_batch_size, eval_range_begin = -1, eval_range_end = -1, norm=True):
    # create partial eval datasets
    # if eval_range_begin != -1:
    #     # range_begin = 60000
    #     # range_end = 85450
    #     featurized_file = input_file + '_{}_{}.pkl'.format(eval_range_begin, eval_range_end)
    # else:
    featurized_file = input_file + '.pkl'
    # if Path(featurized_file).is_file():
    #     with open(featurized_file, "rb") as read_file:
    #         features = pickle.load(read_file)
    #     print('loaded pre-computed features')
    # else:
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response):
        # create partial eval datasets
        # if i >= eval_range_begin and i < eval_range_end:
        for tgt in tgt_all:
            if norm:
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, src_line, tgt_line))
            i += 1
    print(len(examples))
    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id

        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))


    features = [featurize(e) for e in examples]
    with open(featurized_file, "wb") as write_file:
        pickle.dump(features, write_file)

    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack(
            [torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack(
            [torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
                                                 batch_first=True, padding_value=-1)

        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        return input_ids, position_ids, token_type_ids, labels, context_len, response_len

    # for l in sorted(dataloader_pre):
    for l in dataloader_pre:
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader

def get_eval_list_same_length(input_file, tokenizer, max_batch_size, norm=True):
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response) :
        for tgt in tgt_all:
            if norm:
                src_line = ' '.join(src.strip().split())
                tgt_line = ' '.join(tgt.strip().split())
            else:
                src_line = src.strip()
                tgt_line = tgt.strip()
            examples.append(RedditExample(i, src_line, tgt_line))
            i += 1

    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        response_id = tokenizer.encode(example.response)
        input_ids = context_id + [end_of_text_id]
        lm_labels = response_id

        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             lm_labels, len(context_id), len(response_id))

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long) for f in features],
                                                 batch_first=True, padding_value=-1)

        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        return input_ids, position_ids, token_type_ids, labels, context_len, response_len

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader


def get_eval_list_same_length_with_order(input_file, tokenizer, max_batch_size, norm=True):
    # temporary fix, will replace get_eval_list_same_length in future
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    # content = content[:16]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response):
        if norm:
            src_line = ' '.join(src.strip().split())
            tgt_line = [' '.join(tgt.strip().split()) for tgt in tgt_all]
        else:
            src_line = src.strip()
            tgt_line = [tgt.strip() for tgt in tgt_all]
        examples.append(RedditExample(i, src_line, tgt_line))
        i += 1

    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        input_ids = context_id + [end_of_text_id]
        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             example.response, len(context_id), -1)

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = [f.lm_labels for f in features]

        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        conv_ids = torch.tensor([torch.tensor(f.conv_id, dtype=torch.long) for f in features])

        return input_ids, position_ids, token_type_ids, labels, context_len, response_len, conv_ids

    features = [featurize(e) for e in examples]
    dataloader_pre = defaultdict(list)
    for f in features:
        dataloader_pre[f.context_len].append(f)

    dataloader = []
    for l in sorted(dataloader_pre):
        f = batch_feature_same_len(dataloader_pre[l])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader


def get_eval_list_unsorted(input_file, tokenizer, max_batch_size, norm=True):
    # temporary fix, will replace get_eval_list_same_length in future
    examples = []
    with open(input_file, 'r', encoding="utf-8") as f:
        content = [l.split('\t') for l in f.read().splitlines()]

    # content = content[:16]

    context, response = [c[0] for c in content], [c[1:] for c in content]
    i = 0
    for src, tgt_all in zip(context, response):
        if norm:
            src_line = ' '.join(src.strip().split())
            tgt_line = [' '.join(tgt.strip().split()) for tgt in tgt_all]
        else:
            src_line = src.strip()
            tgt_line = [tgt.strip() for tgt in tgt_all]
        examples.append(RedditExample(i, src_line, tgt_line))
        i += 1

    def featurize(example):
        conv_id = example.conv_id
        context_id = tokenizer.encode(example.context)
        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]

        input_ids = context_id + [end_of_text_id]
        position_ids = list(range(len(input_ids)))

        # TODO: assign TOKEN ID in future
        token_type_id = [0] * len(input_ids)

        return InputFeatures(conv_id, input_ids, position_ids, token_type_id,
                             example.response, len(context_id), -1)

    def batch_feature_same_len(features):
        input_ids = torch.stack([torch.tensor(f.choices_features['input_ids'], dtype=torch.long) for f in features])
        position_ids = torch.stack([torch.tensor(f.choices_features['position_ids'], dtype=torch.long) for f in features])
        token_type_ids = torch.stack([torch.tensor(f.choices_features['token_type_ids'], dtype=torch.long) for f in features])
        labels = [f.lm_labels for f in features]

        context_len = torch.tensor([f.context_len for f in features], dtype=torch.long)
        response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        conv_ids = torch.tensor([torch.tensor(f.conv_id, dtype=torch.long) for f in features])

        return input_ids, position_ids, token_type_ids, labels, context_len, response_len, conv_ids

    features = [featurize(e) for e in examples]
    dataloader = []

    for feat in features:
        f = batch_feature_same_len([feat])
        if len(f[0]) <= max_batch_size:
            dataloader.append(f)
        else:
            start_index = 0
            while True:
                dataloader.append([ff[start_index:start_index + max_batch_size] for ff in f])
                start_index += max_batch_size
                if start_index >= len(f[0]):
                    break
    return dataloader

def set_lr(optimizer, step, schedule, lr,
           warmup_steps, warmup_proportion, n_embd, tot_steps):
    if schedule == 'None':
        lr_this_step = lr
    elif schedule == 'noam':  # transformer like
        lr_this_step = lr * 1e4 * noam_decay(step+1, warmup_steps, n_embd)
    elif schedule == 'noamwd':  # transformer like
        lr_this_step = lr * 1e4 * noamwd_decay(step+1, warmup_steps, n_embd)
    else:
        lr_this_step = lr * warmup_linear(step / tot_steps,
                                          warmup_proportion)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


def get_len_mapper(config):
    seq_len = np.genfromtxt(config, delimiter=',', skip_header=1, dtype=int)
    seq_len_mapper = np.ones(seq_len[-1][0], dtype=int) * 160
    for i in range(len(seq_len)-1):
        seq_len_mapper[seq_len[i][0]] = seq_len[i][1]
        for j in range(seq_len[i][0]+1, seq_len[i+1][0]):
            seq_len_mapper[j] = seq_len[i+1][1] * SEQ_LENGTH_SHRINK_PROP
    return seq_len_mapper
