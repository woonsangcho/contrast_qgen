import numpy as np
import torch

from marco_ranker import load_ranker, run_eval, convert_ranker_features

from gpt2_training.generation import cut_seq_to_eos, beam_search_naive, EOS_ID, multisource_batch_generate_sequence, cut_seq_to_eos_cnt
import torch.nn.functional as F
from pretrained_bert_ranker.metrics import metrics

import time
METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
from numpy import dot
from numpy.linalg import norm


def parse_raw(enc, input_ids, doc_count, args, batch_size):
    full_str = [enc.decode(s).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in input_ids.cpu().numpy()]
    split_str = []
    indiv_idx_pointer = []
    for y in full_str:
        encoded_tokens = [enc.encode(x) + [EOS_ID] for x in y.split('<|endoftext|>')][:doc_count]  # added EOS ID!
        split_str.append(encoded_tokens)
        indiv_idx_pointer.append([len(x)-1 for x in encoded_tokens])  # used to have -1 , but need to put EOS in the input .just like run_gpt2.py
    split_tensors = [torch.tensor(x).unsqueeze(0).to(args.device) for y in split_str for x in y]
    if len(split_tensors) != batch_size * doc_count:
        return -1, -1, -1, -1, -1, -1

    passage_str_list = []
    for s in split_str:
        passage_str_list.append([enc.decode(w).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for w in s])
    # randomize "within" positive cluster and negative cluster order
    input_ids_list = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in split_tensors], batch_first=True, padding_value=EOS_ID).view(batch_size, 20, -1)
    randorder = torch.cat((torch.randperm(10), torch.randperm(10) + 10)).to(args.device)
    input_ids_list = input_ids_list[:, randorder, :]
    indiv_idx_pointer = [[indiv_idx_pointer[0][i] for i in randorder]]
    passage_str_list = [[passage_str_list[0][i] for i in randorder]]

    cluster_queries_str = []
    for y in full_str:
        cluster_queries_str.append([x for x in y.split('<|endoftext|><|endoftext|>')][1].split('<|endoftext|>')[:1][0].strip())

    # negative target query for debugging
    cluster_queries_str2 = []
    for y in full_str:
        cluster_queries_str2.append(
            [x for x in y.split('<|endoftext|><|endoftext|>')][1].split('<|endoftext|>')[1].strip())

    return input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2


def parse_raw_non_retrieved(enc, input_ids, doc_count, args, batch_size):
    full_str = [enc.decode(s).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in input_ids.cpu().numpy()]
    split_str = []
    indiv_idx_pointer = []
    for y in full_str:
        encoded_tokens = [enc.encode(x) + [EOS_ID] for x in y.split('<|endoftext|>')][:doc_count]  # added EOS ID!
        split_str.append(encoded_tokens)
        indiv_idx_pointer.append([len(x)-1 for x in encoded_tokens])  # used to have -1 , but need to put EOS in the input .just like run_gpt2.py
    split_tensors = [torch.tensor(x).unsqueeze(0).to(args.device) for y in split_str for x in y]
    if len(split_tensors) != batch_size * doc_count:
        return -1, -1, -1, -1, -1, -1

    passage_str_list = []
    for s in split_str:
        passage_str_list.append([enc.decode(w).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for w in s])
    # randomize "within" positive cluster and negative cluster order
    input_ids_list = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in split_tensors], batch_first=True, padding_value=EOS_ID).view(batch_size, 20, -1)
    randorder = torch.cat((torch.randperm(10), torch.randperm(10) + 10)).to(args.device)
    input_ids_list = input_ids_list[:, randorder, :]
    indiv_idx_pointer = [[indiv_idx_pointer[0][i] for i in randorder]]
    passage_str_list = [[passage_str_list[0][i] for i in randorder]]

    cluster_queries_str = []
    for y in full_str:
        cluster_queries_str.append([x for x in y.split('<|endoftext|><|endoftext|>')][1].split('<|endoftext|>')[:1][0].strip())

    # negative target query for debugging
    cluster_queries_str2 = []
    for y in full_str:
        cluster_queries_str2.append(
            [x for x in y.split('<|endoftext|><|endoftext|>')][1].split('<|endoftext|>')[1].strip())


    return input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2


def auxiliary_sl(args, batch_size, attention_seq, out, common_vocab_dist_seq, indiv_vocab_dist_seq):
    ent_loss, positive_cluster_loss, negative_cluster_loss = torch.FloatTensor([0.0]).to(args.device), \
                                                                   torch.FloatTensor([0.0]).to(args.device), \
                                                                   torch.FloatTensor([0.0]).to(args.device)
    for m in range(batch_size):
        batch_positive_cluster_loss = 0.0
        batch_negative_cluster_loss = 0.0
        step_ent_loss = 0.0
        decode_len = len(cut_seq_to_eos(out[m].cpu().numpy()))
        for i in range(decode_len):
            step_ent_loss += torch.sum(F.softmax(attention_seq[i], dim=-1) * F.log_softmax(attention_seq[i],
                                                                                           dim=-1))  # maximize entropy to be near uniform
            for j in range(10):
                batch_positive_cluster_loss += F.kl_div(common_vocab_dist_seq[i][m], torch.exp(indiv_vocab_dist_seq[i][m][j])) + F.kl_div(
                    indiv_vocab_dist_seq[i][m][j], torch.exp(common_vocab_dist_seq[i][m]))

            positive_cluster_avg_probs = torch.logsumexp(indiv_vocab_dist_seq[i][m][:10], 0) - torch.log \
                (torch.from_numpy(np.array(float(10))))
            negative_cluster_avg_probs = torch.logsumexp(indiv_vocab_dist_seq[i][m][11:], 0) - torch.log \
                (torch.from_numpy(np.array(float(10))))
            sim_val = F.cosine_similarity(torch.exp(positive_cluster_avg_probs),
                                          torch.exp(negative_cluster_avg_probs), dim=0)

            del positive_cluster_avg_probs, negative_cluster_avg_probs

            for j in range(11, 20):
                negative_cluster_kl = F.kl_div(common_vocab_dist_seq[i][m],
                                               torch.exp(indiv_vocab_dist_seq[i][m][j])) + F.kl_div(
                    indiv_vocab_dist_seq[i][m][j], torch.exp(common_vocab_dist_seq[i][m]))
                batch_negative_cluster_loss += negative_cluster_kl if sim_val * negative_cluster_kl < batch_positive_cluster_loss.item() / 10 else 0.0  # torch.FloatTensor([0.0]).to(args.device)

        positive_cluster_loss += batch_positive_cluster_loss
        negative_cluster_loss += batch_negative_cluster_loss
    sl_loss = positive_cluster_loss - negative_cluster_loss

    # entropy loss
    ent_loss += step_ent_loss

    return sl_loss, ent_loss


def compute_reward_and_baselines(batch_size, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded, eval_mode, indiv_out_decoded, model, input_ids_list, enc, baseline_indiv_idx_pointer, coordinator, cluster_queries_str):
    reward = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list,max_passage_len, args, out_decoded=out_decoded, eval_mode=eval_mode, return_all=True)

    # if eval mode, only get pure reward of argmax generations
    if not eval_mode:
        # baseline 1: average rewards of individual samples
        if args.use_baseline_1:
            mean_indiv_rewards = BaselineOneBatched(ranker, ranker_tokenizer, indiv_out_decoded, passage_str_list,
                                                    max_passage_len, args, eval_mode=eval_mode)
            reward -= mean_indiv_rewards

        # baseline 2: naive average baseline reward
        if args.use_baseline_2:
            base_reward = BaselineTwoBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                             max_passage_len, args, baseline_indiv_idx_pointer, eval_mode=eval_mode)
            reward -= base_reward

        # baseline 3: self-critic baseline reward
        if args.use_baseline_3:
            assert args.is_sampling == True, "To use self-critic, the main model should be sampled, not top-k"
            base_reward = BaselineThreeBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                               max_passage_len, args, baseline_indiv_idx_pointer, coordinator,
                                               eval_mode=eval_mode)
            reward -= base_reward

        # baseline 6: cluster target queries baseline reward
        if args.use_baseline_6:
            base_reward = BaselineSixBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                             max_passage_len, args, coordinator,
                                             cluster_queries_str=cluster_queries_str, eval_mode=eval_mode)
            reward -= base_reward
    return reward

def aggregate_loss(args, rl_loss, sl_loss, ent_loss):
    if args.optimize_option == 'rlsl':
        loss = args.rl_wt * rl_loss + args.sl_wt * sl_loss
    if args.optimize_option == 'all':
        rl_loss *= args.rl_wt
        sl_loss *= args.sl_wt
        ent_loss *= args.ent_wt
        loss = rl_loss + sl_loss + ent_loss
    elif args.optimize_option == 'rl':
        loss = rl_loss * args.rl_wt
    elif args.optimize_option == 'sl':
        loss = sl_loss * args.sl_wt
    return loss

def compute_batched_losses(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=False, coordinator=None):
    batch_size = input_ids.shape[0]
    doc_count = 20

    input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, _ = parse_raw(enc, input_ids, doc_count, args, batch_size)
    if cluster_queries_str == -1:
        return -1, -1, -1, -1, -1
    max_passage_len = input_ids_list.shape[2]

    # clone baseline indiv_idx_pointer for baselines that need to generate
    baseline_indiv_idx_pointer=None
    if args.use_baseline_2 or args.use_baseline_3:
        baseline_indiv_idx_pointer = indiv_idx_pointer[:]

    out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out, attention_seq = multisource_batch_generate_sequence(
        model,
        input_ids_list,
        length=args.generation_length,
        temperature=args.temperature,
        top_k=args.top_k,
        sample=False if eval_mode else args.is_sampling, # argmax when evaluating.
        device=args.device,
        coordinator=coordinator,
        indiv_idx_pointer=indiv_idx_pointer,
        args=args
        )

    # convert generations to tokens
    out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()] #TODO need batched here and below

    # if using individual generations for baseline 1
    indiv_out_decoded = []
    if args.use_baseline_1:
        for s in indiv_out:
            indiv_out_decoded.append([enc.decode(cut_seq_to_eos(w)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for w in s.cpu().numpy()])

    # Auxiliary SL
    if args.optimize_option == "rlsl" or args.optimize_option == "all" or args.optimize_option == 'sl' or eval_mode == True:
        sl_loss, ent_loss = auxiliary_sl(args, batch_size, attention_seq, out, common_vocab_dist_seq, indiv_vocab_dist_seq)
    else:
        sl_loss, ent_loss = torch.FloatTensor([0.0]).to(args.device), torch.FloatTensor([0.0]).to(args.device)

    # Main RL routine
    if args.optimize_option == "rlsl" or args.optimize_option == "all" or args.optimize_option == 'rl' or eval_mode == True:
        reward = compute_reward_and_baselines(batch_size, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded, eval_mode, indiv_out_decoded, model, input_ids_list, enc, baseline_indiv_idx_pointer, coordinator, cluster_queries_str)
        rl_loss = torch.sum( - out_total_log_probs.squeeze(1) * torch.from_numpy(reward).float().to(args.device))
    else:
        rl_loss = torch.FloatTensor([0.0]).to(args.device)

    # Training options for different components of the loss function
    loss = aggregate_loss(args, rl_loss, sl_loss, ent_loss)
    return rl_loss.item(), sl_loss.item(), loss, torch.mean(torch.from_numpy(reward).float().to(args.device)), out_decoded

# 100-passage retrieval task
def retrieval_evaluation(ranker, ranker_tokenizer, passage_str_list, engine, generated_query, args):
    pool = engine.get_candidates(generated_query, 80)
    pool = list(pool[0].values())
    pool = [' '.join(p[1]) for p in pool]

    pool = list(set(pool) - set(passage_str_list[0]))
    # list(set(pool).union(set(passage_str_list[0])))
    combined_pool = passage_str_list[0] + pool
    pool_count = len(combined_pool)

    logits = RankerPoolRewards(ranker, ranker_tokenizer, combined_pool[:20], args, generated_query)
    for start in [20,40,60,80]:
        if pool_count > start:
            partial_logits = RankerPoolRewards(ranker, ranker_tokenizer, combined_pool[start:start+20], args, generated_query)
            logits = np.concatenate((logits, partial_logits), axis=0)

    # map, rprec, mrr, ndcg, mrr10 = all_metrics
    all_metrics = np.zeros((len(METRICS_MAP)))
    pred_labels = logits[:, 1].argsort()[::-1]
    positive_labels_gt = set(list([x for x in range(10)]))
    # negative_labels_gt = set(list([x for x in range(10, 20)]))

    # MAP+RPREC
    all_metrics += metrics(gt=positive_labels_gt, pred=pred_labels, metrics_map=METRICS_MAP)
    MAP, RPREC, _, _, _ = all_metrics

    # MRR,NDCG, MRR10
    agg_metrics = np.zeros((len(METRICS_MAP)))
    agg_logits = np.concatenate((np.mean(logits[:10], 0, keepdims=True), logits[10:]), axis=0)
    pred_labels = agg_logits[:, 1].argsort()[::-1]
    positive_labels_gt = set([0])
    agg_metrics += metrics(gt=positive_labels_gt, pred=pred_labels, metrics_map=METRICS_MAP)
    _, _, MRR,NDCG, MRR10 = agg_metrics

    return MAP, RPREC, MRR, NDCG, MRR10


def complete_eval_samples(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=False, coordinator=None, compute_baseline_reward=False, genenc=None, engine=None, compute_retrieval_baseline_reward=False):
    batch_size = input_ids.shape[0]
    doc_count = 20

    # input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2, retrieved_question = parse_raw(enc, input_ids, doc_count, args, batch_size)
    input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2 = parse_raw_non_retrieved(enc, input_ids, doc_count, args, batch_size)

    if cluster_queries_str == -1:
        return -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1
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
                reference_corr = dot(genenc[cluster_queries_str[0].split('1.0 ')[1]], genenc[cluster_queries_str2[0]]) / (
                        norm(genenc[cluster_queries_str[0].split('1.0 ')[1]]) * norm(genenc[cluster_queries_str2[0]]))
            except:
                pass
        except:
            pass

    # clone baseline indiv_idx_pointer for baselines that need to generate
    baseline_indiv_idx_pointer=None
    if args.use_baseline_2 or args.use_baseline_3:
        baseline_indiv_idx_pointer = indiv_idx_pointer[:]

    # for computing reference baseline reward
    if compute_baseline_reward:
        assert isinstance(cluster_queries_str, list)
        reward = BaselineSixBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                         max_passage_len, args, coordinator,
                                         cluster_queries_str=cluster_queries_str,
                                         eval_mode=eval_mode, return_all=True)

        # 100-passage retrieval evaluation
        MAP, RPREC, MRR, NDCG, MRR10 = 0.0, 0.0, 0.0, 0.0, 0.0
        if engine:
            MAP, RPREC, MRR, NDCG, MRR10 = retrieval_evaluation(ranker, ranker_tokenizer, passage_str_list, engine,
                                                                generated_query=cluster_queries_str, args=args)

        return reward, cluster_queries_str, cluster_queries_str2, MAP, RPREC, MRR, NDCG, MRR10
        # return base_reward, cluster_queries_str, passage_str_list, -1, -1, cluster_queries_str2
    elif compute_retrieval_baseline_reward:
        assert isinstance(retrieved_question, list)
        reward = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args,
                                      out_decoded=retrieved_question, eval_mode=True, return_all=True)
        if engine:
            MAP, RPREC, MRR, NDCG, MRR10 = retrieval_evaluation(ranker, ranker_tokenizer, passage_str_list, engine,
                                                                generated_query=retrieved_question, args=args)

        return reward, retrieved_question, MAP, RPREC, MRR, NDCG, MRR10
    else:
        out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out, attention_seq = multisource_batch_generate_sequence(
            model,
            input_ids_list,
            length=args.generation_length,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False if eval_mode else args.is_sampling, # argmax when evaluating.
            device=args.device,
            coordinator=coordinator,
            indiv_idx_pointer=indiv_idx_pointer,
            args=args
            )

        # convert generations to tokens
        out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()] #TODO need batched here and below
        assert isinstance(out_decoded, list)
        # if using individual generations for baseline 1
        indiv_out_decoded = []
        if args.use_baseline_1:
            for s in indiv_out:
                indiv_out_decoded.append([enc.decode(cut_seq_to_eos(w)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for w in s.cpu().numpy()])

        # Auxiliary SL
        if args.optimize_option == "rlsl" or args.optimize_option == "all" or args.optimize_option == 'sl' or eval_mode == True:
            sl_loss, ent_loss = auxiliary_sl(args, batch_size, attention_seq, out, common_vocab_dist_seq, indiv_vocab_dist_seq)
        else:
            sl_loss, ent_loss = torch.FloatTensor([0.0]).to(args.device), torch.FloatTensor([0.0]).to(args.device)

        # Main RL routine
        if args.optimize_option == "rlsl" or args.optimize_option == "all" or args.optimize_option == 'rl' or eval_mode == True:
            reward = compute_reward_and_baselines(batch_size, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded, eval_mode, indiv_out_decoded, model, input_ids_list, enc, baseline_indiv_idx_pointer, coordinator, cluster_queries_str)
            # since it is complete eval compute rl_loss with rprec
            rl_loss = torch.sum(- out_total_log_probs.squeeze(1) * torch.from_numpy(reward[:, 1]).float().to(args.device))
        else:
            rl_loss = torch.FloatTensor([0.0]).to(args.device)

        # Training options for different components of the loss function
        loss = aggregate_loss(args, rl_loss, sl_loss, ent_loss)

        # 100-passage retrieval evaluation
        MAP, RPREC, MRR, NDCG, MRR10 = 0.0, 0.0, 0.0, 0.0, 0.0
        if engine:
            MAP, RPREC, MRR, NDCG, MRR10 = retrieval_evaluation(ranker, ranker_tokenizer, passage_str_list, engine,
                                                                generated_query=out_decoded, args=args)

        # return rl_loss.item(), sl_loss.item(), loss, torch.mean(torch.from_numpy(reward).float().to(args.device), axis=0), out_decoded, passage_str_list, attention_seq, reference_corr, MAP, RPREC, MRR, NDCG, MRR10
        return rl_loss.item(), sl_loss.item(), loss, reward, out_decoded, passage_str_list, attention_seq, reference_corr, MAP, RPREC, MRR, NDCG, MRR10

def complete_eval_attention_weights(model, enc, input_ids, args, eval_mode, coordinator, genenc):
    batch_size = input_ids.shape[0]
    doc_count = 20

    input_ids_list, full_str, indiv_idx_pointer, passage_str_list, cluster_queries_str, cluster_queries_str2 = parse_raw_non_retrieved(enc, input_ids, doc_count, args, batch_size)
    if cluster_queries_str == -1:
        return -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1
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
                reference_corr = dot(genenc[cluster_queries_str[0].split('1.0 ')[1]], genenc[cluster_queries_str2[0]]) / (
                        norm(genenc[cluster_queries_str[0].split('1.0 ')[1]]) * norm(genenc[cluster_queries_str2[0]]))
            except:
                pass
        except:
            pass

    # clone baseline indiv_idx_pointer for baselines that need to generate
    baseline_indiv_idx_pointer=None
    if args.use_baseline_2 or args.use_baseline_3:
        baseline_indiv_idx_pointer = indiv_idx_pointer[:]

    out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out, attention_seq, balancer_seq = multisource_batch_generate_sequence_for_attentions(
        model,
        input_ids_list,
        length=args.generation_length,
        temperature=args.temperature,
        top_k=args.top_k,
        sample=False if eval_mode else args.is_sampling, # argmax when evaluating.
        device=args.device,
        coordinator=coordinator,
        indiv_idx_pointer=indiv_idx_pointer,
        args=args,
        )

    # convert generations to tokens
    out_decoded = [enc.splitted_decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()]
    assert isinstance(out_decoded, list)

    return out_decoded, attention_seq, balancer_seq, reference_corr


def complete_eval_retrieval_baseline_samples_standalone(passage_str_list, ranker, ranker_tokenizer, args, engine):

    # given 10 documents, retrieve the top-k questions using lucene
    k = 5
    pools = {}
    n_positive_docs = 10
    for p in passage_str_list[0][:n_positive_docs]:
        pool = engine.get_candidates([p], k)
        pool = list(pool[0].values())
        pool_str = [' '.join(p[1]) for p in pool]
        pool_score = [p[2] for p in pool]
        for p_str, p_score in zip(pool_str, pool_score):
            if p_str in pools:
                pools[p_str] += p_score
            else:
                pools[p_str] = p_score
    best_retrieved_question = [max(pools, key=pools.get)]
    assert isinstance(best_retrieved_question, list)

    # pools = []
    # n_positive_docs = 10
    # for p in passage_str_list[0][:n_positive_docs]:
    #     pool = engine.get_candidates([p], k)
    #     pool = list(pool[0].values())
    #     pool = [' '.join(p[1]) for p in pool]
    #     pool_score = [p[2] for p in pool]
    #
    #     pools += pool
    # or with most appearing since intersection is too harsh!!
    # question_counts = collections.Counter(pools)
    # common_question = [question_counts.most_common(1)[0][0]]
    # assert isinstance(common_question, list)

    reward = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args,
                                  out_decoded=best_retrieved_question, eval_mode=True, return_all=True)
    if engine:
        MAP, RPREC, MRR, NDCG, MRR10 = retrieval_evaluation(ranker, ranker_tokenizer, passage_str_list, engine,
                                                            generated_query=best_retrieved_question, args=args)

    return reward, best_retrieved_question, MAP, RPREC, MRR, NDCG, MRR10



def RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded=None, indiv_out_decoded=None, eval_mode=False, return_all=False):

    assert out_decoded or indiv_out_decoded, "one of out_decoded (batch_size) or indiv_out_decoded (batch_size * doccount) should be provided"

    # flatten
    batch_size = len(passage_str_list)
    doc_count = len(passage_str_list[0])
    ranker_input_ids_list, ranker_segment_ids_list, ranker_input_mask_list = [], [], []
    for i in range(batch_size):
        for j in range(doc_count):
            question = out_decoded[i] if out_decoded else indiv_out_decoded[i][j]
            ranker_input_ids, ranker_segment_ids, ranker_input_mask = convert_ranker_features(ranker_tokenizer, question,
                                                                                              passage_str_list[i][j], min(max_passage_len + args.generation_length * 2,512), args.device)
            ranker_input_ids_list.append(ranker_input_ids)
            ranker_segment_ids_list.append(ranker_segment_ids)
            ranker_input_mask_list.append(ranker_input_mask)
            del ranker_input_ids, ranker_segment_ids, ranker_input_mask
            torch.cuda.empty_cache()

    ranker = ranker.module if hasattr(ranker, 'module') else ranker
    with torch.no_grad():
        padded_input_ids_list = torch.nn.utils.rnn.pad_sequence(ranker_input_ids_list, padding_value=0).squeeze(0)
        padded_segment_ids_list = torch.nn.utils.rnn.pad_sequence(ranker_segment_ids_list, padding_value=0).squeeze(0)
        padded_input_mask_list = torch.nn.utils.rnn.pad_sequence(ranker_input_mask_list, padding_value=0).squeeze(0)
        logits = ranker(padded_input_ids_list,
                        token_type_ids=padded_segment_ids_list,
                        attention_mask=padded_input_mask_list,
                        labels=None)
        del padded_input_ids_list, padded_segment_ids_list, padded_input_mask_list
        torch.cuda.empty_cache()

    # reduce
    logits = logits.view(batch_size,doc_count,-1)


    # Rewards calculation
    all_metrics = np.zeros((batch_size, len(METRICS_MAP)))
    for m in range(batch_size):
        pred_labels = logits[m,:, 1].cpu().numpy().argsort()[::-1][:10]
        positive_labels_gt = set(list([x for x in range(10)]))
        negative_labels_gt = set(list([x for x in range(10,20)]))

        all_metrics[m] += metrics(gt=positive_labels_gt, pred=pred_labels, metrics_map=METRICS_MAP)

        # negative contrastive benchmark
        if eval_mode == False and args.use_contrastive_reward:
            all_metrics[m] +=  -metrics(gt=negative_labels_gt, pred=pred_labels, metrics_map=METRICS_MAP)

    # RL loss: weight via rewards
    # map, rprec, mrr, ndcg, mrr10 = all_metrics
    # rew = map
    if return_all:
        return all_metrics

    reward_idx = -1
    if args.reward_type == 'map':
        reward_idx = 0
    elif args.reward_type == 'rprec':
        reward_idx = 1

    return all_metrics[:,reward_idx] # MAP

def RankerPoolRewards(ranker, ranker_tokenizer, combined_pool, args, generated_query):

    # flatten
    pool_count = len(combined_pool)
    max_passage_len = max([len(p) for p in combined_pool])
    ranker_input_ids_list, ranker_segment_ids_list, ranker_input_mask_list = [], [], []

    for j in range(pool_count):
        question = generated_query[0]
        ranker_input_ids, ranker_segment_ids, ranker_input_mask = convert_ranker_features(ranker_tokenizer, question,
                                                                                          combined_pool[j], min(max_passage_len + args.generation_length * 2,512), args.device)
        ranker_input_ids_list.append(ranker_input_ids)
        ranker_segment_ids_list.append(ranker_segment_ids)
        ranker_input_mask_list.append(ranker_input_mask)
        del ranker_input_ids, ranker_segment_ids, ranker_input_mask
        torch.cuda.empty_cache()

    ranker = ranker.module if hasattr(ranker, 'module') else ranker
    with torch.no_grad():
        padded_input_ids_list = torch.nn.utils.rnn.pad_sequence(ranker_input_ids_list, padding_value=0).squeeze(0)
        padded_segment_ids_list = torch.nn.utils.rnn.pad_sequence(ranker_segment_ids_list, padding_value=0).squeeze(0)
        padded_input_mask_list = torch.nn.utils.rnn.pad_sequence(ranker_input_mask_list, padding_value=0).squeeze(0)
        logits = ranker(padded_input_ids_list,
                        token_type_ids=padded_segment_ids_list,
                        attention_mask=padded_input_mask_list,
                        labels=None)
        del padded_input_ids_list, padded_segment_ids_list, padded_input_mask_list
        torch.cuda.empty_cache()

    return logits.cpu().numpy()

def BaselineOneBatched(ranker, ranker_tokenizer, indiv_out_decoded, passage_str_list, max_passage_len, args, eval_mode=False):
    with torch.no_grad():
        mean_indiv_rewards = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, indiv_out_decoded=indiv_out_decoded, eval_mode=eval_mode)
        return mean_indiv_rewards

def BaselineTwoBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, indiv_idx_pointer, eval_mode=False):
    with torch.no_grad():
        # generate
        out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out = multisource_batch_generate_sequence(
            model,
            input_ids_list,
            length=args.generation_length,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False,  # argmax when evaluating.
            device=args.device,
            coordinator=None,
            indiv_idx_pointer=indiv_idx_pointer
        )

        base_out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()]

        # ranker scoring + compute rewards
        base_reward = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded = base_out_decoded, eval_mode=eval_mode)
        return base_reward

# baseline 3: self-critic baseline reward
def BaselineThreeBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args,indiv_idx_pointer, coordinator, eval_mode=False):
    with torch.no_grad():
        out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out = multisource_batch_generate_sequence(
            model,
            input_ids_list,
            length=args.generation_length,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False,  # argmax when evaluating.
            device=args.device,
            coordinator=coordinator,
            indiv_idx_pointer=indiv_idx_pointer
        )

        # convert generations to tokens
        base_out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()]

        # ranker scoring + compute rewards
        base_reward = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, out_decoded=base_out_decoded, eval_mode=eval_mode)

        return base_reward

# baseline 6: use the 'target' query. This should be faster since it avoid generation.
def BaselineSixBatched(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, coordinator, cluster_queries_str, eval_mode=False, return_all=False):
    with torch.no_grad():
        mean_indiv_rewards = RankerRewardsBatched(ranker, ranker_tokenizer, passage_str_list, max_passage_len, args,
                                                  out_decoded=cluster_queries_str, eval_mode=eval_mode, return_all=return_all)
        return mean_indiv_rewards


def RankerRewards(ranker, ranker_tokenizer, out_decoded, passage_str_list, max_passage_len, args):
    ranker_input_ids_list, ranker_segment_ids_list, ranker_input_mask_list = [], [], []
    for p in passage_str_list:
        ranker_input_ids, ranker_segment_ids, ranker_input_mask = convert_ranker_features(ranker_tokenizer, out_decoded,
                                                                                          p, min(max_passage_len + args.generation_length * 2,512), args.device)
        ranker_input_ids_list.append(ranker_input_ids)
        ranker_segment_ids_list.append(ranker_segment_ids)
        ranker_input_mask_list.append(ranker_input_mask)

    with torch.no_grad():
        logits = ranker(torch.nn.utils.rnn.pad_sequence(ranker_input_ids_list, padding_value=0).squeeze(0),
                        token_type_ids=torch.nn.utils.rnn.pad_sequence(ranker_segment_ids_list,
                                                                       padding_value=0).squeeze(0),
                        attention_mask=torch.nn.utils.rnn.pad_sequence(ranker_input_mask_list, padding_value=0).squeeze(
                            0),
                        labels=None)

    # Rewards calculation
    all_metrics = np.zeros(len(METRICS_MAP))
    pred_labels = logits[:, 1].cpu().numpy().argsort()[::-1]
    labels_gt = set(list([x for x in range(10)]))

    all_metrics += metrics(gt=labels_gt, pred=pred_labels, metrics_map=METRICS_MAP)

    # RL loss: weight via rewards
    map, rprec, mrr, ndcg, mrr10 = all_metrics
    rew = map

    return rew

# baseline 1: average rewards of individual samples
def BaselineOne(ranker, ranker_tokenizer, indiv_out_decoded, passage_str_list, max_passage_len, args):
    mean_indiv_rewards = 0.0
    for indiv_gen in indiv_out_decoded:
        mean_indiv_rewards += RankerRewards(ranker, ranker_tokenizer, indiv_gen, passage_str_list, max_passage_len,
                                            args)
    mean_indiv_rewards /= len(indiv_out_decoded)
    return mean_indiv_rewards

# baseline 2: naive average baseline reward
def BaselineTwo(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args):
    with torch.no_grad():
        base_out, base_out_total_log_probs, base_common_vocab_dist_seq, base_indiv_vocab_dist_seq, base_indiv_out = multisource_generate_sequence(
            model,
            input_ids_list,
            length=args.generation_length,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=args.is_sampling,
            device=args.device,
            coordinator=None
        )

        # convert generations to tokens
        base_out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in base_out.cpu().numpy()][0]

        # ranker scoring + compute rewards
        base_reward = RankerRewards(ranker, ranker_tokenizer, base_out_decoded, passage_str_list, max_passage_len, args)
        return base_reward

# baseline 3: self-critic baseline reward
def BaselineThree(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args, coordinator):
    with torch.no_grad():
        base_out, base_out_total_log_probs, base_common_vocab_dist_seq, base_indiv_vocab_dist_seq, base_indiv_out = multisource_generate_sequence(
            model,
            input_ids_list,
            length=args.generation_length,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=False, # KEY PART
            device=args.device,
            coordinator=coordinator
        )

        # convert generations to tokens
        base_out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in base_out.cpu().numpy()][0]

        # ranker scoring + compute rewards
        base_reward = RankerRewards(ranker, ranker_tokenizer, base_out_decoded, passage_str_list, max_passage_len, args)
        return base_reward

# baseline 4: multiple generation samples (top-k) and average of their rewards
def BaselineFour(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args):
    with torch.no_grad():
        raise NotImplementedError

# baseline 5: past training batches average baseline reward
def BaselineFive(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args):
    with torch.no_grad():
        raise NotImplementedError

def compute_losses(model, enc, ranker, ranker_tokenizer, input_ids, args, eval_mode=False, coordinator=None):
    doc_count = 20

    rl_loss, sl_loss, loss, positive_cluster_loss, negative_cluster_loss = torch.FloatTensor([-1.0]).to(args.device),\
                                                                            torch.FloatTensor([-1.0]).to(args.device),\
                                                                            torch.FloatTensor([-1.0]).to(args.device),\
                                                                            torch.FloatTensor([-1.0]).to(args.device),\
                                                                            torch.FloatTensor([-1.0]).to(args.device)


    full_str = [enc.decode(s).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in input_ids.cpu().numpy()][0]
    split_str = [enc.encode(x) for x in full_str.split('<|endoftext|>') if len(x) > 2]
    split_tensors = [torch.tensor(x).unsqueeze(0).to(args.device) for x in split_str]
    input_ids_list = split_tensors[:doc_count]
    max_passage_len = max([len(x[0]) for x in input_ids_list])

    passage_str_list = [enc.decode(s).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in split_str[:doc_count]]

    # assert len(input_ids_list) == 20, "Input is corrupt. Total length is %d, and raw text: %s" % (len(input_ids_list), full_str)
    if len(input_ids_list) != 20:
        return -1, -1, -1, -1, -1

    # generate
    out, out_total_log_probs, common_vocab_dist_seq, indiv_vocab_dist_seq, indiv_out = multisource_generate_sequence(
        model,
        input_ids_list,
        length=args.generation_length,
        temperature=args.temperature,
        top_k=args.top_k,
        sample=False if eval_mode else args.is_sampling, # argmax when evaluating.
        device=args.device,
        coordinator=coordinator
        )

    # convert generations to tokens
    out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in out.cpu().numpy()][0]
    indiv_out_decoded = [enc.decode(cut_seq_to_eos(s)).encode('utf-8', 'backslashreplace').decode('utf-8', 'backslashreplace') for s in indiv_out.cpu().numpy()]


    # Auxiliary SL
    if args.optimize_option == None or args.optimize_option == 'only_sl' or eval_mode == True:
        decode_len = len(cut_seq_to_eos(out[0].cpu().numpy()))
        for i in range(decode_len):
            for j in range(10):
                positive_cluster_loss += F.kl_div(common_vocab_dist_seq[i][0], torch.exp(indiv_vocab_dist_seq[i][j]))

            positive_cluster_avg_probs = torch.logsumexp(indiv_vocab_dist_seq[i][:10], 0) - torch.log \
                (torch.from_numpy(np.array(float(10))))
            negative_cluster_avg_probs = torch.logsumexp(indiv_vocab_dist_seq[i][11:], 0) - torch.log \
                (torch.from_numpy(np.array(float(10))))
            sim_val = F.cosine_similarity(torch.exp(positive_cluster_avg_probs), torch.exp(negative_cluster_avg_probs), dim=0)

            for j in range(11 ,20):
                negative_cluster_kl = F.kl_div(common_vocab_dist_seq[i][0], torch.exp(indiv_vocab_dist_seq[i][j]))
                negative_cluster_loss += negative_cluster_kl if sim_val * negative_cluster_kl < positive_cluster_loss.item( ) / 10  else torch.FloatTensor \
                    ([0.0]).to(args.device)

        sl_loss = positive_cluster_loss - negative_cluster_loss

    # Main RL routine
    reward = 0.0
    if args.optimize_option == None or args.optimize_option == 'only_rl' or eval_mode == True:
        # ranker scoring + compute rewards
        reward = RankerRewards(ranker, ranker_tokenizer, out_decoded, passage_str_list, max_passage_len, args)

        # if eval mode, only get pure reward of argmax generations
        if not eval_mode:
            # baseline 1: average rewards of individual samples
            if args.use_baseline_1:
                mean_indiv_rewards = BaselineOne(ranker, ranker_tokenizer, indiv_out_decoded, passage_str_list, max_passage_len, args)
                reward -= mean_indiv_rewards

            # baseline 2: naive average baseline reward
            if args.use_baseline_2:
                base_reward = BaselineTwo(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list, max_passage_len, args)
                reward -= base_reward

            # baseline 3: self-critic baseline reward
            if args.use_baseline_3:
                assert args.is_sampling == True, "To use self-critic, the main model should be sampled, not top-k"
                base_reward = BaselineThree(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                          max_passage_len, args)
                reward -= base_reward

            # baseline 4: multiple generation samples (top-k) and average of their rewards
            if args.use_baseline_4:
                base_reward = BaselineFour(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                            max_passage_len, args)
                reward -= base_reward

            # baseline 5: past training batches average baseline reward
            if args.use_baseline_5:
                base_reward = BaselineFive(model, input_ids_list, enc, ranker, ranker_tokenizer, passage_str_list,
                                            max_passage_len, args)
                reward -= base_reward

        # loss
        rl_loss = - out_total_log_probs * reward

    # Training options for different components of the loss function
    if args.optimize_option == None:
        loss = rl_loss + sl_loss
    elif args.optimize_option == 'only_rl':
        loss = rl_loss
    elif args.optimize_option == 'only_sl':
        loss = sl_loss
    elif args.optimize_option == 'only_sl_positive_cluster':
        loss = positive_cluster_loss
    elif args.optimize_option == 'only_sl_negative_cluster':
        loss = -negative_cluster_loss
    elif args.optimize_option == 'only_sl_iter':
        raise NotImplementedError

    return rl_loss.item(), sl_loss.item(), loss, reward, out_decoded

