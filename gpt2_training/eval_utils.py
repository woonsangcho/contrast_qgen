import torch
import tqdm
import logging
import json

import numpy as np
from os.path import join

logger = logging.getLogger(__name__)

from numpy import dot
from numpy.linalg import norm

EOS_ID = 50256

from pretrained_bert_ranker.metrics import metrics
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
import random
from rl_helpers import compute_losses, compute_batched_losses, complete_eval_samples

def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # new_batch = []
            # for t in batch:
            #     if isinstance(t,list):
            #         new_batch.append(t)
            #     else:
            #         new_batch.append(t.to(args.device))
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if not args.no_token_id:
                # some bug for code below..., currently disable it
                """ 
                new_token_ids = []
                tot_len = input_ids.size(1)
                for s in src_len:
                    new_token_ids.append(torch.cat((torch.zeros([1, s], dtype=token_ids.dtype, device=token_ids.device),
                                                    torch.ones([1, tot_len - s], dtype=token_ids.dtype,
                                                               device=token_ids.device)), 1))
                token_ids = torch.stack(new_token_ids, dim=1)
                """
                pass
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            # import pdb; pdb.set_trace()
            # print(input_ids.shape, position_ids.shape, token_ids.shape, label_ids.shape)
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)


def eval_model_rl_loss(model, enc, eval_dataloader, epoch_id, args,
                       ranker,
                       coordinator,
                       ranker_tokenizer
                       ):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    coordinator.eval()
    tot_coord_loss = []
    tot_naive_loss = []
    tot_rl_coord_loss = []
    tot_sl_coord_loss = []
    tot_rl_naive_loss = []
    tot_sl_naive_loss = []
    tot_coord_reward = []
    tot_naive_reward = []
    tot_sample = []
    gen_samples_coord = []
    gen_samples_naive = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch

            with torch.no_grad():
                rl_loss_coord, sl_loss_coord, loss_coord, reward_coord, sample_response = compute_batched_losses(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=coordinator)
                gen_samples_coord.append(sample_response)

                rl_loss_naive, sl_loss_naive, loss_naive, reward_naive, sample_response = compute_batched_losses(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=None)
                gen_samples_naive.append(sample_response)
                # rl_loss_coord, sl_loss_coord, loss_coord, reward_coord, sample_response = rl_loss_naive, sl_loss_naive, loss_naive, reward_naive, sample_response

            if sample_response == -1: # corrupt sample
                continue

            n_sample = input_ids.shape[0]
            tot_coord_loss.append(loss_coord.mean().item() * n_sample)
            tot_naive_loss.append(loss_naive.mean().item() * n_sample)
            tot_rl_coord_loss.append(rl_loss_coord * n_sample)
            tot_sl_coord_loss.append(sl_loss_coord * n_sample)
            tot_rl_naive_loss.append(rl_loss_naive * n_sample)
            tot_sl_naive_loss.append(sl_loss_naive * n_sample)
            tot_coord_reward.append(reward_coord * n_sample)
            tot_naive_reward.append(reward_naive * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: "
          f"\n Val RL Coord Loss {np.sum(tot_rl_coord_loss) / np.sum(tot_sample)} Val SL Coord Loss {np.sum(tot_sl_coord_loss) / np.sum(tot_sample)} Val Coord Loss {np.sum(tot_coord_loss) / np.sum(tot_sample)}"
          f"\n Val Coord Reward {np.sum(tot_coord_reward) / np.sum(tot_sample)}"
          # f"\n Coord Sample Gen: {random.choice(gen_samples_coord)} "
          f"\n "
          f"\n Val RL Naive Loss {np.sum(tot_rl_naive_loss) / np.sum(tot_sample)} Val SL Naive Loss {np.sum(tot_sl_naive_loss) / np.sum(tot_sample)} Val Naive Loss {np.sum(tot_naive_loss) / np.sum(tot_sample)}"
          f"\n Val Naive Reward {np.sum(tot_naive_reward) / np.sum(tot_sample)}")
          # f"\n Naive Sample Gen: {random.choice(gen_samples_naive)} ")
    coordinator.train()

    return np.sum(tot_rl_coord_loss) / np.sum(tot_sample), np.sum(tot_sl_coord_loss) / np.sum(tot_sample), np.sum(tot_coord_loss) / np.sum(tot_sample),np.sum(tot_coord_reward) / np.sum(tot_sample),np.sum(tot_naive_reward) / np.sum(tot_sample), random.choice(gen_samples_coord),random.choice(gen_samples_naive)

def complete_eval_model(model, enc, eval_dataloader, epoch_id, args,
                       ranker,
                       coordinator,
                       ranker_tokenizer,
                       genenc,
                       log_dir,
                       engine):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    coordinator.eval()
    tot_coord_loss = []
    tot_naive_loss = []
    tot_rl_coord_loss = []
    tot_sl_coord_loss = []
    tot_rl_naive_loss = []
    tot_sl_naive_loss = []
    tot_coord_reward = []
    tot_naive_reward = []
    tot_baseline_reward = []
    tot_sample = []
    gen_samples_coord = []
    gen_samples_naive = []
    gen_attention_seq = []
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch

            with torch.no_grad():
                rl_loss_coord, sl_loss_coord, loss_coord, reward_coord, sample_coord_response, passage_str_list, attention_seq, reference_corr, coord_MAP, coord_RPREC, coord_MRR, coord_NDCG, coord_MRR10 = complete_eval_samples(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=coordinator, genenc=genenc, engine=engine)
                if sample_coord_response == -1 or reference_corr == -1:  # corrupt sample
                    continue

                rl_loss_naive, sl_loss_naive, loss_naive, reward_naive, sample_naive_response, _, _, _, naive_MAP, naive_RPREC, naive_MRR, naive_NDCG, naive_MRR10 = complete_eval_samples(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=None, genenc=genenc, engine=engine)
                if sample_naive_response == -1:  # corrupt sample
                    continue

                gen_samples_coord.append(sample_coord_response)
                gen_attention_seq.append(attention_seq)
                gen_samples_naive.append(sample_naive_response)

                # human reference
                reward_baseline, sample_baseline_response, neg_reference, ref_MAP, ref_RPREC, ref_MRR, ref_NDCG, ref_MRR10 = complete_eval_samples(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=None, compute_baseline_reward=True, genenc=None, engine=engine)

                # retrieval baseline
                # reward_retrieve, retrieve_common_response, retrieve_MAP, retrieve_RPREC, retrieve_MRR, retrieve_NDCG, retrieve_MRR10 = complete_eval_samples(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=True, coordinator=None, compute_baseline_reward=False, genenc=None, engine=engine, compute_retrieval_baseline_reward=True)

            n_sample = input_ids.shape[0]
            tot_coord_loss.append(loss_coord.mean().item() * n_sample)
            tot_naive_loss.append(loss_naive.mean().item() * n_sample)
            tot_rl_coord_loss.append(rl_loss_coord * n_sample)
            tot_sl_coord_loss.append(sl_loss_coord * n_sample)
            tot_rl_naive_loss.append(rl_loss_naive * n_sample)
            tot_sl_naive_loss.append(sl_loss_naive * n_sample)
            tot_coord_reward.append(reward_coord[0] * n_sample)  #reward_coord[0]
            tot_naive_reward.append(reward_naive[0] * n_sample)
            tot_baseline_reward.append(reward_baseline[0] * n_sample)
            tot_sample.append(n_sample)


            # print
            with open(join(log_dir, 'complete_eval_log.txt'), 'a+', buffering=1) as complete_eval_logger:
                print(u'{},{},{},{},{},{},{},{}'.format(
                    # reward_coord[0],
                    # reward_naive[0],
                    # reward_baseline[0],
                    ','.join(map(str,reward_coord[0])),
                    ','.join(map(str,reward_naive[0])),
                    ','.join(map(str,reward_baseline[0])),
                    ','.join(map(str,reward_coord[0]-reward_baseline[0])),
                    reference_corr,
                    sample_coord_response[0],
                    sample_naive_response[0],
                    sample_baseline_response[0]
                ),
                file=complete_eval_logger)

            result = {}
            result.update({'reward_coord': reward_coord[0].tolist()})
            result.update({'reward_naive': reward_naive[0].tolist()})
            result.update({'reward_baseline': reward_baseline[0].tolist()})
            # result.update({'reward_retrieve_baseline': reward_retrieve[0].tolist()})
            result.update({'reference_corr': reference_corr})
            result.update({'coord_response': sample_coord_response[0]})
            result.update({'naive_response': sample_naive_response[0]})
            result.update({'baseline_response': sample_baseline_response[0]})
            # result.update({'retrieve_response': retrieve_common_response[0]})
            result.update({'neg_reference': neg_reference[0]})
            result.update({'passages': passage_str_list})
            result.update({'coord_map': coord_MAP})
            result.update({'coord_rprec': coord_RPREC})
            result.update({'coord_mrr': coord_MRR})
            result.update({'coord_ndcg': coord_NDCG})
            result.update({'coord_mrr10': coord_MRR10})
            result.update({'naive_map': naive_MAP})
            result.update({'naive_rprec': naive_RPREC})
            result.update({'naive_mrr': naive_MRR})
            result.update({'naive_ndcg': naive_NDCG})
            result.update({'naive_mrr10': naive_MRR10})
            result.update({'ref_map': ref_MAP})
            result.update({'ref_rprec': ref_RPREC})
            result.update({'ref_mrr': ref_MRR})
            result.update({'ref_ndcg': ref_NDCG})
            result.update({'ref_mrr10': ref_MRR10})
            # result.update({'retrieve_map': retrieve_MAP})
            # result.update({'retrieve_rprec': retrieve_RPREC})
            # result.update({'retrieve_mrr': retrieve_MRR})
            # result.update({'retrieve_ndcg': retrieve_NDCG})
            # result.update({'retrieve_mrr10': retrieve_MRR10})


            eval_log_path = join(log_dir, 'complete_eval_log.json')
            with open(eval_log_path, "a+") as complete_eval_logger:
                complete_eval_logger.write(json.dumps(result) + "\n")


    print(f"\n Epoch {epoch_id}: "
          f"\n Val RL Coord Loss {np.sum(tot_rl_coord_loss) / np.sum(tot_sample)} Val SL Coord Loss {np.sum(tot_sl_coord_loss) / np.sum(tot_sample)} Val Coord Loss {np.sum(tot_coord_loss) / np.sum(tot_sample)}"
          f"\n Val Coord Reward {np.mean(np.stack(tot_coord_reward), 0)}"
          f"\n Coord Sample Gen: {random.choice(gen_samples_coord)} "
          f"\n "
          f"\n Val RL Naive Loss {np.sum(tot_rl_naive_loss) / np.sum(tot_sample)} Val SL Naive Loss {np.sum(tot_sl_naive_loss) / np.sum(tot_sample)} Val Naive Loss {np.sum(tot_naive_loss) / np.sum(tot_sample)}"
          f"\n Val Naive Reward {np.mean(np.stack(tot_naive_reward), 0)}"
          f"\n Naive Sample Gen: {random.choice(gen_samples_naive)} "
          f"\n "
          f"\n Val Baseline Reward {np.mean(np.stack(tot_baseline_reward), 0)}"
    )
    return



def complete_extract_retrieval_intersect_baseline_model(enc, eval_dataloader, args,
                       ranker,
                       ranker_tokenizer,
                       genenc,
                       log_dir,
                       question_engine
                       ):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, please change it back to train after calling this function')
    tot_retrieve_baseline_reward = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch

            doc_count = 20
            batch_size = input_ids.shape[0]
            input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2 = parse_raw_non_retrieved(
                enc, input_ids, doc_count, args, input_ids.shape[0])
            max_passage_len = input_ids_list.shape[2]

            reference_corr = -1.0
            if genenc:
                try:
                    try:
                        reference_corr = dot(genenc[cluster_queries_str[0]], genenc[cluster_queries_str2[0]]) / (
                                norm(genenc[cluster_queries_str[0]]) * norm(genenc[cluster_queries_str2[0]]))
                    except:
                        pass

                    try:
                        reference_corr = dot(genenc[cluster_queries_str[0].split('1.0 ')[1]],
                                             genenc[cluster_queries_str2[0]]) / (
                                                 norm(genenc[cluster_queries_str[0].split('1.0 ')[1]]) * norm(
                                             genenc[cluster_queries_str2[0]]))
                    except:
                        pass
                except:
                    pass

            retrieve_common_response = ""
            k = 100
            if cluster_queries_str == -1 or reference_corr == -1.0:
                retrieve_common_response, cluster_queries_str, cluster_queries_str2 = ['N/A'], ['N/A'], ['N/A']
            else:
                # retrieval baseline
                pool_qs = []
                pool_scores = []
                # pools = []
                n_positive_docs = 10
                for i,p in enumerate(passage_str_list[0][:n_positive_docs]):
                    pool = question_engine.get_candidates([p], k)
                    poolval = list(pool[0].values())
                    pool_q = [' '.join(q[1]) for q in poolval]
                    pool_score = [p[2] for p in poolval]
                    # for p_str, p_score in zip(pool_q, pool_score):
                    pool_qs+=pool_q
                    pool_scores+=pool_score

                retrieve_common_response = [max(set(pool_qs), key=pool_qs.count)]
                # retrieve_common_response = [max(pools, key=pools.get)]

            result = {}
            result.update({'retrieve_response': retrieve_common_response[0]})
            result.update({'reference_response': cluster_queries_str[0]})
            result.update({'neg_reference': cluster_queries_str2[0]})


            #TODO json dump dic update
            retrieved_question_path = join(log_dir, 'extracted_retrieved_question_k_maxoccur_{}.tsv'.format(k))
            with open(retrieved_question_path, "a+", encoding='utf8', newline='') as retrieved_fh:
                retrieved_writer = csv.writer(retrieved_fh, delimiter='\t')
                retrieved_writer.writerow([retrieve_common_response[0], cluster_queries_str[0], cluster_queries_str2[0]])

    return


def complete_print_attention_weights(
       model,
       enc,
       eval_dataloader,
       args,
       ranker,
       coordinator,
       ranker_tokenizer,
       genenc,
       log_dir):
    # use the same signature with eval_model_generation
    logger.info('entering attention weight printing loop')
    coordinator.eval()
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, *_ = batch

            with torch.no_grad():
                sample_coord_response, attention_seq, balancer_seq, reference_corr = complete_eval_attention_weights(model, enc, input_ids, args, eval_mode=True, coordinator=coordinator, genenc=genenc)
                if sample_coord_response == -1 or reference_corr == -1:  # corrupt sample
                    continue

    return
