
import json
import os
from os.path import abspath, dirname, exists, join
import sys
import argparse
import logging
import math
import time
import pickle

import tqdm
import datetime
import numpy as np
import torch
from torch.utils.data import RandomSampler, TensorDataset, DataLoader
from torch.distributed import get_rank, get_world_size

from gpt2_training.train_utils import (
    load_model, DynamicBatchingLoader, boolean_string, set_lr,
    get_eval_list_same_length, get_len_mapper, get_eval_list)
from gpt2_training.eval_utils import eval_model_loss, eval_model_rl_loss, complete_eval_model

from data_loader import BucketingDataLoader, DistributedBucketingDataLoader
from env import PROJECT_FOLDER

from optim import Adam
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from gpt2_training.distributed import (all_reduce_and_rescale_tensors,
                                       all_gather_list)

from marco_ranker import load_ranker, run_eval, convert_ranker_features
from coordinator.coordinator import load_coordinator, AttnCoordConfig


from rl_helpers import compute_losses, compute_batched_losses
from pretrained_bert_ranker.metrics import metrics
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']

import copy

from mylucene import lucene_search
from mylucene import lucene_question_search


# debug, prepare lucene indexing files
engine = lucene_search.LuceneSearch()
# question_engine = lucene_question_search.QuestionLuceneSearch()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=1,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=10000,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", action='store_true')

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--save_step", type=int, default=30000)
parser.add_argument('--pbar', action='store_true', help='turn on progress bar')

# generation
parser.add_argument("--nsamples", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--length", type=int, default=-1)

parser.add_argument("--generation_length", type=int, default=10)
parser.add_argument("--temperature", type=int, default=1)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument('--unconditional', action='store_true',
                    help='If true, unconditional generation.')
parser.add_argument('--is_sampling', action='store_true',
                    help='If true, sampling for generation.')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')

parser.add_argument('--config', help='JSON config file')


# bert ranker args
## Required parameters
parser.add_argument("--ranker_model", default=None, type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--ranker_eval_batch_size",
                    default=10,
                    type=int,
                    help="Total batch size for eval.")

# coordinator args
parser.add_argument("--coordinator_model", default=None, type=str,
                    help="provide path otherwise initialize")

parser.add_argument("--optimize_option", default=None, type=str,
                    help="provide path otherwise initialize")
# parser.add_argument('--use_auxiliary_sl', action='store_true',
#                     help='If true, use auxiliary SL loss.')
parser.add_argument('--use_baseline_1', action='store_true',
                    help='If true, use mean rewards of individual generations.')
parser.add_argument('--use_baseline_2', action='store_true',
                    help='If true, use naive average model generation rewards')
parser.add_argument('--use_baseline_3', action='store_true',
                    help='If true, use self-critic model generation rewards')
# parser.add_argument('--use_baseline_4', action='store_true',
#                     help='If true, use top-k samples generation average rewards ' )
# parser.add_argument('--use_baseline_5', action='store_true',
#                     help='If true, use past training/eval batches average rewards ')
parser.add_argument('--use_baseline_6', action='store_true',
                    help='If true, use target queries from training data as baseline' )

parser.add_argument('--coord_config', type=str,help='JSON config file')
parser.add_argument("--initializer_range", type=float, default=0.02, help='transformer based coordinator configs')
parser.add_argument("--layer_norm_epsilon", type=float, default=1e-5, help='transformer based coordinator configs')
parser.add_argument("--n_ctx", type=int, default=1024, help='transformer based coordinator configs')
parser.add_argument("--n_embd", type=int, default=768, help='transformer based coordinator configs')
parser.add_argument("--n_head", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--n_layer", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--n_clusters", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--vocab_size", type=int, default=50527, help='transformer based coordinator configs')

parser.add_argument('--coord_type', default='fused',type=str,help='specify coordinator type')
parser.add_argument('--reward_type', default='rprec',type=str,help='specify reward type')
parser.add_argument('--use_contrastive_reward', action='store_true',
                    help='If true, minus the negative cluster reward')
parser.add_argument("--checkpoint_rate", type=float, default=0.05, help='validate + save every x % of 1 epoch of data')

parser.add_argument('--genenc_path', default='genenc_embedding/genenc_marco.pkl',type=str,help=' genenc path')

parser.add_argument('--use_contrast_damp', action='store_true',
                    help='If true, limit the effect of negative docs' )
parser.add_argument("--eval_range_begin", type=int, default=-1)
parser.add_argument("--eval_range_end", type=int, default=-1)

parser.add_argument('--ignore_negative', action='store_true',
                    help='If true, ignores the negative weights so no penalization at all')

# do normal parsing
args = parser.parse_args()

assert args.coordinator_model is not None, "Should load a trained coordinator model to evaluate!"

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)


if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

# cpu error avoid
if n_gpu < 1:
    n_gpu = 1

args.num_optim_steps = args.num_optim_steps // args.train_batch_size // n_gpu
args.valid_step = int(args.num_optim_steps * args.checkpoint_rate)
args.save_step = args.valid_step

assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)
logger.info('train batch size = {}, '
            'new train batch size (after gradient accumulation) = {}'.format(
                args.train_batch_size*args.gradient_accumulation_steps,
                args.train_batch_size))
logger.info('new num_optim_steps (after num gpu adjustment) = {}'.format(
                args.num_optim_steps))


np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

if args.fp16:
    config = join(abspath(PROJECT_FOLDER),
                  'config_file/SeqLen_vs_BatchSize_1GPU_fp16.csv')
else:
    config = join(abspath(PROJECT_FOLDER),
                  'config_file/SeqLen_vs_BatchSize_1GPU_fp32.csv')
seq_len_mapper = get_len_mapper(config)
#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)#, do_lower_case=True)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

if args.local_rank == -1:
    train_dataloader = BucketingDataLoader(args.train_input_file,
                                           args.train_batch_size,
                                           args.max_seq_length)
else:
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length)

eval_dataloader_loss = DynamicBatchingLoader(
    args.eval_input_file, enc, args.normalize_data,
    args.eval_batch_size, args.max_seq_length,
    is_train=True)

eval_dataloader_gen = get_eval_list(
    args.eval_input_file, enc, args.eval_batch_size, args.eval_range_begin, args.eval_range_end, True)

logger.info("***** For training dataset *****")
logger.info("***** For dev dataset *****")
logger.info('num example = %d, batch_size = %d, num_batches = %d'
            % (eval_dataloader_loss.num_examples, args.eval_batch_size,
               len(eval_dataloader_gen)))


#########################################################################
# Prepare Model
##########################################################################
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)
logger.info('Loaded model from %s in device %s' %(args.init_checkpoint, args.device))


#########################################################################
# Prepare BERT reranker and Coordinator and Optimizer
##########################################################################
ranker, ranker_tokenizer = load_ranker(args)
logger.info('Loaded pretrained BERT reranker in device %s' % args.device)

# override
args.coord_config = AttnCoordConfig.from_json_file(args.coord_config)
args.gpt2_lm_head = model.lm_head.decoder.weight.clone()
coordinator = load_coordinator(args, type=args.coord_type)
coordinator.train()
logger.info('Loaded coordinator from %s in device %s' %(args.coordinator_model, args.device))

if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in coordinator.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, coordinator.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(coordinator.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False,
                          max_grad_norm=1.0)
    if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                                   verbose=False)
    else:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   verbose=False)
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                     max_grad_norm=1.0)


#########################################################################
# Training !
##########################################################################

if args.local_rank == -1 or get_rank() == 0:
    with open(join(log_dir, 'train_log.txt'), 'a+', buffering=1) as train_logger:
        print('epoch,global_step,step,mean_rl_loss,mean_sl_loss,mean_loss,n_token_real,'
              'n_token_total,epoch_time', file=train_logger)
    with open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1) as eval_logger:
        print('epoch,global_step,step,eval_rl_loss,eval_sl_loss,eval_loss,eval_coord_reward,'
        'eval_naive_reward, eval_baseline_reward', file=eval_logger)
epoch = 0

coordinator.zero_grad()
model.eval()

with open(args.genenc_path, "rb") as read_file:
    genenc = pickle.load(read_file)

complete_eval_model(
    model, enc, eval_dataloader_loss, epoch, args,
    ranker,
    coordinator,
    ranker_tokenizer,
    genenc,
    log_dir,
    engine)


