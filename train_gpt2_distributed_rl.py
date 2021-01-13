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
    get_eval_list_same_length, get_len_mapper)
from gpt2_training.eval_utils import eval_model_generation, eval_model_loss, eval_model_rl_loss

from data_loader import BucketingDataLoader, DistributedBucketingDataLoader
from env import PROJECT_FOLDER

from optim import Adam
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from gpt2_training.distributed import (all_reduce_and_rescale_tensors,
                                       all_gather_list)

from marco_ranker import load_ranker, run_eval, convert_ranker_features
from coordinator.coordinator import load_coordinator, AttnCoordConfig


from rl_helpers import compute_losses, compute_batched_losses
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']

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

parser.add_argument("--train_batch_size", type=int, default=1024,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
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

parser.add_argument("--optimize_option", default="all", type=str,
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
parser.add_argument("--n_ctx", type=int, default=20, help='transformer based coordinator configs')
parser.add_argument("--n_embd", type=int, default=768, help='transformer based coordinator configs')
parser.add_argument("--n_head", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--n_layer", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--n_clusters", type=int, default=2, help='transformer based coordinator configs')
parser.add_argument("--vocab_size", type=int, default=50527, help='transformer based coordinator configs')

parser.add_argument('--coord_type', default='fused',type=str,help='specify coordinator type')
parser.add_argument('--reward_type', default='rprec',type=str,help='specify reward type')
parser.add_argument('--use_contrastive_reward', action='store_true',
                    help='If true, minus the negative cluster reward')
parser.add_argument("--checkpoint_rate", type=float, default=0.033, help='validate + save every x % of 1 epoch of data') #total optim step is set to 3 epochs
parser.add_argument('--use_contrast_damp', action='store_true',
                    help='If true, limit the effect of negative docs' )
parser.add_argument("--ent_wt", type=float, default=0.1, help='entropy weight in the sl loss funciton')
parser.add_argument("--rl_wt", type=float, default=1.0, help='entropy weight in the sl loss funciton')
parser.add_argument("--sl_wt", type=float, default=100.0, help='entropy weight in the sl loss funciton')
# do normal parsing
args = parser.parse_args()


# TODO there might be a better way to write this...
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

# modify num_optim_steps
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

eval_dataloader_gen = get_eval_list_same_length(
    args.eval_input_file, enc, args.eval_batch_size, True)

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
    # FIXME is averaging the best way? init variance will change

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
        print('epoch,global_step,step,eval_rl_loss,eval_sl_loss,eval_loss,sample_generation', file=eval_logger)


global_step = 0
step = 0
epoch = 0

if args.continue_from:
    global_step = args.continue_from
    step = global_step*2 - 1


if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

coordinator.zero_grad()
model.eval()
while True:
    (tr_loss, tr_sl_loss, nb_tr_examples, nb_tr_steps
     ) = 0.0, 0.0, 0, 0
    tr_rl_loss, tr_reward, mean_rl_loss, mean_sl_loss, mean_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        # activate new training mode
        seq_len = batch[0].shape[1]
        max_bs = 128
        tmp_dataset = TensorDataset(*batch)
        tmp_dataloader = DataLoader(tmp_dataset,
                                    sampler=RandomSampler(tmp_dataset),
                                    batch_size=max_bs)
        gas = len(tmp_dataloader)
        for _, tmp_batch in enumerate(tmp_dataloader):
            tmp_batch = tuple(t.to(device) for t in tmp_batch)
            input_ids, position_ids, token_ids, label_ids, *_ = tmp_batch

            rl_loss, sl_loss, loss, reward, sample_response = compute_batched_losses(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=False, coordinator=coordinator)

            if sample_response == -1 or loss.grad_fn == None:
                continue
            step += 1

            if n_gpu > 1:
                loss = loss.mean()
                sl_loss = sl_loss.mean()
                rl_loss = rl_loss.mean()
            if math.isnan(loss.item()):
                # skip mini-batch if NaN
                with open(join(log_dir, 'train_log.txt'), 'a+', buffering=1) as train_logger:
                    print('NaN!!', file=train_logger)
                if not exists(f'{output_dir}/debug_nan_{args.local_rank}.pt'):
                    # save snapshot for debugging
                    debug_snapshot = {
                        'state_dict': {k: v.cpu()
                                       for k, v in model.state_dict().items()},
                        'input_ids': input_ids.cpu(),
                        'position_ids': position_ids.cpu(),
                        'token_ids': (token_ids.cpu() if token_ids is not None else None),
                        'label_ids': label_ids.cpu()}
                    torch.save(debug_snapshot,
                               f'{output_dir}/debug_nan_{args.local_rank}.pt')
                continue
            loss = loss / (args.train_batch_size / input_ids.shape[0])
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_rl_loss += float(rl_loss)
            tr_reward += float(reward)

            tr_loss += float(loss.item()) * (
                args.train_batch_size / input_ids.shape[0])
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps
            mean_rl_loss = tr_rl_loss / nb_tr_steps
            mean_reward = tr_reward / nb_tr_steps
            if sl_loss < INF:
                try:
                    tr_sl_loss += sl_loss.item()
                except:
                    tr_sl_loss += sl_loss
            else:
                tr_sl_loss += mean_sl_loss
            mean_sl_loss = tr_sl_loss / nb_tr_steps

            n_token_total += input_ids.shape[0] * input_ids.shape[1]
            n_token_real += (input_ids != 0).sum().item()

        # gradient update
        # step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in coordinator.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Print log info to file
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_sl_loss = sum(all_gather_list(mean_sl_loss)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
                n_samples_total_all_proc = sum(all_gather_list(nb_tr_examples))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total
                n_samples_total_all_proc = nb_tr_examples


            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"samples/s: {n_samples_total_all_proc/epoch_time:.5f} "
                        f"tok/s: {n_token_real_all_proc//epoch_time} "
                        f"RL loss: {mean_rl_loss:.5f} "
                        f"SL loss: {mean_sl_loss:.5f} Total loss: {mean_loss:.5f}  epoch: {epoch}")
                    pbar.update(1)
                with open(join(log_dir, 'train_log.txt'), 'a+', buffering=1) as train_logger:
                    print('{},{},{},{},{},{},{},{},{}'.format(
                        epoch+1, global_step+1, step+1, mean_rl_loss, mean_sl_loss, mean_loss,
                        n_token_real_all_proc, n_token_total_all_proc, epoch_time),
                        file=train_logger)

            if ((global_step-1) % args.valid_step == 0 and global_step > 1) or (global_step == args.num_optim_steps):
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    if (global_step-1) % args.save_step == 0:
                        torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                             for k, v in coordinator.state_dict().items()},
                            join(output_dir,
                                 f'{args.coord_type}-{args.optimize_option}-step-{global_step}.pkl'))



                    eval_rl_loss, eval_sl_loss, eval_loss, eval_coord_reward, eval_naive_reward, sample_coord_response, sample_naive_response = eval_model_rl_loss(
                        model, enc, eval_dataloader_loss, epoch, args,
                        ranker,
                        coordinator,
                        ranker_tokenizer)

                    # empty cuda cache
                    torch.cuda.empty_cache()

                    # eval_loss, eval_sl_loss = eval_model_loss(
                    #     model, enc, eval_dataloader_loss, epoch, args)
                    # enable generation step evaluation for now
                    # gen_response = eval_model_generation(
                    #     model, enc, eval_dataloader_gen, epoch, args)
                    '''
                    # probably use beam search only for test set
                    if False:
                        gen_response_beam = eval_model_generation(
                            model, enc, eval_dataloader_gen, epoch, args,
                            use_beam_search=True, beam_width=3)
                    '''
                    with open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1, encoding="utf-8") as eval_logger:
                        print('{},{},{},{},{},{},{},{}'.format(
                            epoch+1, global_step+1, step+1, eval_rl_loss, eval_sl_loss, eval_loss,eval_coord_reward, eval_naive_reward),#,sample_coord_response,sample_naive_response),
                            file=eval_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    coordinator.train()
            if global_step >= args.num_optim_steps:
                break

        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if global_step >= args.num_optim_steps:
        break
    epoch += 1

if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
