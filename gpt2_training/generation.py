import torch
import torch.nn.functional as F
import numpy as np

EOS_ID = 50256
import time

def multisource_batch_generate_sequence(model, input_ids_list, position_ids=None, token_type_ids=None, temperature=1, top_k=0, length = 10, sample=False, device=None, coordinator=None, indiv_idx_pointer=None, args=None):
    output = input_ids_list.new_zeros([input_ids_list.shape[0],0])
    individual_output = input_ids_list.new_zeros([input_ids_list.shape[0], input_ids_list.shape[1],0])

    model = model.module if hasattr(model, 'module') else model

    prev = input_ids_list
    common_input_ids_list = input_ids_list.clone()

    past_list=None
    total_log_probs_sel = 0
    common_vocab_dist_seq = []
    indiv_vocab_dist_seq = []
    attention_seq = []
    for i in range(length):
        prev, log_probs_sel, common_vocab_dist, indiv_vocab_dist, individual_prev, attention = \
            multisource_batch_pack_generate_next_token(model, input_ids_list, position_ids, token_type_ids, prev,
                                                       temperature, top_k, sample, device, coordinator=coordinator,
                                                       indiv_idx_pointer=indiv_idx_pointer, common_input_ids_list=common_input_ids_list, args=args)
        output = torch.cat((output, prev), dim=1)

        # cache attention weights only for common question generation
        if not torch.all(torch.gt(torch.sum(output == EOS_ID, -1), 0)):
            attention_seq.append(attention)

        # increase one dimension of EOS
        input_ids_list = torch.cat((input_ids_list,torch.LongTensor([EOS_ID]).repeat(input_ids_list.shape[0],20,1).to(device)), dim=2)
        common_input_ids_list = torch.cat((common_input_ids_list,torch.LongTensor([EOS_ID]).repeat(input_ids_list.shape[0],20,1).to(device)), dim=2)

        for j in range(input_ids_list.shape[0]):
            indiv_idx_list = indiv_idx_pointer[j]
            indiv_idx_pointer[j] = [x+1 for x in indiv_idx_list]

        # Place individual predictions at right places
        for m in range(len(input_ids_list)):
            for i, j in zip(range(20), indiv_idx_pointer[m]):
                input_ids_list[m,i,j] = individual_prev[m,i,0]
                common_input_ids_list[m,i,j] = prev[m][0]

        total_log_probs_sel += log_probs_sel
        common_vocab_dist_seq.append(common_vocab_dist)
        indiv_vocab_dist_seq.append(indiv_vocab_dist)

        individual_output = torch.cat((individual_output, individual_prev), dim=2)

        # break early if done decoding
        if torch.all(torch.gt(torch.sum(output == EOS_ID, -1),0)) and torch.all(torch.gt(torch.sum(individual_output == EOS_ID, -1),0)):
            break

    return output, total_log_probs_sel, common_vocab_dist_seq, indiv_vocab_dist_seq, individual_output, attention_seq


def multisource_batch_pack_generate_next_token(model, input_ids_list, position_ids=None, token_type_ids=None, prev=None,
                                          temperature=1, top_k=0, sample=False, device=None,
                                          individual_prev=None, coordinator=None, indiv_idx_pointer=None, common_input_ids_list=None, masked_flat_indices=None, args=None):
    model = model.module if hasattr(model, 'module') else model

    # flatten inputs
    batch_size = input_ids_list.shape[0]
    doc_count = input_ids_list.shape[1]
    total_sample_size = batch_size * doc_count
    bool_initial_iteration = True if len(prev.shape) == 3 else False
    prev = prev.view(total_sample_size, -1) if bool_initial_iteration else prev.repeat(1, doc_count).view(
        total_sample_size, -1)
    input_ids_list = input_ids_list.view(total_sample_size, -1)
    common_input_ids_list = common_input_ids_list.view(total_sample_size, -1)

    # common + indiv logits batch generation
    with torch.no_grad():
        if bool_initial_iteration:
            h, _ = model.transformer(prev, past=None)
            logits = model.lm_head(h)
            selected_logits = [logits[i, j,:] for i,j in zip(range(total_sample_size),sum(indiv_idx_pointer,[]))]
            logits = torch.stack(selected_logits)
            logits = top_k_logits(logits, k=top_k)
            log_probs_stacked = F.log_softmax(logits, -1)
            selected_h = [h[i, j, :].clone() for i, j in zip(range(total_sample_size), sum(indiv_idx_pointer, []))]
            hidden_states_stacked = torch.stack(selected_h)
            # hidden_states_stacked = h[:,-1,:].clone()
            indiv_log_probs_stacked = log_probs_stacked
            del h, selected_logits, logits
            torch.cuda.empty_cache()
        else:
            h, _ = model.transformer(common_input_ids_list)
            logits = model.lm_head(h)
            selected_logits = [logits[i, j, :] for i, j in zip(range(total_sample_size), sum(indiv_idx_pointer, []))]
            logits = torch.stack(selected_logits).squeeze(1)
            logits = top_k_logits(logits, k=top_k)
            log_probs_stacked = F.log_softmax(logits, -1)
            selected_h = [h[i, j, :].clone() for i, j in zip(range(total_sample_size), sum(indiv_idx_pointer, []))]
            hidden_states_stacked = torch.stack(selected_h)
            # hidden_states_stacked = h[:, -1, :].clone()
            del h, selected_logits, logits
            torch.cuda.empty_cache()

            if args.use_baseline_1:
                # separate indiv dists from its own generations: input_ids_list contains separate generations, select certain positions after padding
                indiv_hidden_states, _ = model.transformer(input_ids_list)
                indiv_hidden_states = indiv_hidden_states.detach()
                indiv_logits = model.lm_head(indiv_hidden_states)
                selected_logits = [indiv_logits[i, j, :] for i, j in zip(range(total_sample_size), sum(indiv_idx_pointer,[]))]
                indiv_logits = torch.stack(selected_logits).squeeze(1)
                indiv_logits = top_k_logits(indiv_logits, k=top_k)
                indiv_log_probs_stacked = F.log_softmax(indiv_logits, -1)
                del indiv_hidden_states, selected_logits, indiv_logits
                torch.cuda.empty_cache()
            else: # dummy
                indiv_log_probs_stacked = log_probs_stacked
    # reduce back to batch first shape
    log_probs_stacked = log_probs_stacked.view(batch_size, doc_count, -1)
    indiv_log_probs_stacked = indiv_log_probs_stacked.view(batch_size, doc_count, -1)
    hidden_states_stacked = hidden_states_stacked.view(batch_size, doc_count, -1)

    if coordinator and args.coord_type == 'fused':
        # normal way
        # attention_weights, common_h_weight, common_generative_log_probs = coordinator(hidden_states_stacked)
        # common_deterministic_log_probs = torch.log(torch.bmm(attention_weights, torch.exp(log_probs_stacked))).squeeze(1)
        # common_log_probs = torch.log((1-common_h_weight) * torch.exp(common_deterministic_log_probs) + common_h_weight * torch.exp(common_generative_log_probs))

        # contrastive way
        attention_weights, common_h_weight, common_generative_log_probs = coordinator(hidden_states_stacked)
        common_deterministic_probs = torch.bmm(attention_weights[:, :, :10], torch.exp(log_probs_stacked)[:, :10, :]) - torch.bmm(attention_weights[:,:,10:], torch.exp(log_probs_stacked)[:,10:,:])
        common_deterministic_probs[common_deterministic_probs <= 1e-20] = 1e-35
        normalizing_vals = torch.sum(common_deterministic_probs, -1, keepdim=True).reciprocal()
        common_deterministic_probs = torch.bmm(normalizing_vals, common_deterministic_probs).squeeze(1)
        common_log_probs = torch.log((1 - common_h_weight) * common_deterministic_probs + common_h_weight * torch.exp(common_generative_log_probs))

    elif coordinator and args.coord_type == 'attention':
        # normal way
        # attention_weights = coordinator(hidden_states_stacked)
        # common_log_probs = torch.log(torch.bmm(attention_weights, torch.exp(log_probs_stacked))).squeeze(1)

        # contrastive way
        attention_weights = coordinator(hidden_states_stacked).contiguous()
        common_deterministic_probs = torch.bmm(attention_weights[:, :, :10],torch.exp(log_probs_stacked)[:, :10, :]) - torch.bmm(attention_weights[:, :, 10:], torch.exp(log_probs_stacked)[:, 10:, :])
        common_deterministic_probs[common_deterministic_probs <= 1e-20] = 1e-35
        normalizing_vals = torch.sum(common_deterministic_probs, -1, keepdim=True).reciprocal()
        common_deterministic_probs = torch.bmm(normalizing_vals, common_deterministic_probs).squeeze(1)
        common_log_probs = torch.log(common_deterministic_probs).contiguous()

    elif coordinator and args.coord_type == 'contrast_attention':
        # contrastive way
        attn_weights_pos, attn_weights_neg, attn_balancer = coordinator(hidden_states_stacked)
        # attention_weights = torch.cat((attn_weights_pos, attn_weights_neg), dim=2).to(device).contiguous()

        common_deterministic_probs = torch.bmm(attn_weights_pos,torch.exp(log_probs_stacked)[:, :10, :]) - torch.bmm(attn_balancer*attn_weights_neg, torch.exp(log_probs_stacked)[:, 10:, :])
        # attention_weights = torch.cat((attn_weights_pos, attn_balancer*attn_weights_neg), dim=2).to(device).contiguous()
        attention_weights = torch.cat((attn_weights_pos, attn_weights_neg), dim=2).to(device).contiguous()
        common_deterministic_probs[common_deterministic_probs <= 1e-20] = 1e-35
        normalizing_vals = torch.sum(common_deterministic_probs, -1, keepdim=True).reciprocal()
        common_deterministic_probs = torch.bmm(normalizing_vals, common_deterministic_probs).squeeze(1)
        common_log_probs = torch.log(common_deterministic_probs).contiguous()
    elif coordinator and args.coord_type == 'contrast_attention_sim':
        # contrastive way
        attn_weights_pos, attn_weights_neg, attn_balancer = coordinator(hidden_states_stacked)

        # sim
        positive_cluster_avg_probs = torch.logsumexp(log_probs_stacked[:,:10,:], 1) - torch.log(torch.from_numpy(np.array(float(10))))
        negative_cluster_avg_probs = torch.logsumexp(log_probs_stacked[:,11:,:], 1) - torch.log(torch.from_numpy(np.array(float(10))))
        sim_val = F.cosine_similarity(torch.exp(positive_cluster_avg_probs), torch.exp(negative_cluster_avg_probs), dim=-1)

        common_deterministic_probs = torch.bmm(attn_weights_pos,torch.exp(log_probs_stacked)[:, :10, :]) - torch.bmm((1-sim_val)*attn_balancer*attn_weights_neg, torch.exp(log_probs_stacked)[:, 10:, :])
        # attention_weights = torch.cat((attn_weights_pos, (1-sim_val)*attn_balancer*attn_weights_neg), dim=2).to(device).contiguous()
        attention_weights = torch.cat((attn_weights_pos, attn_weights_neg), dim=2).to(device).contiguous()

        common_deterministic_probs[common_deterministic_probs <= 1e-20] = 1e-35
        normalizing_vals = torch.sum(common_deterministic_probs, -1, keepdim=True).reciprocal()
        common_deterministic_probs = torch.bmm(normalizing_vals, common_deterministic_probs).squeeze(1)
        common_log_probs = torch.log(common_deterministic_probs).contiguous()
    else:
        # single cluster naive averaging
        attention_weights = torch.cat((torch.FloatTensor([1.0 / 10]).repeat(batch_size,1,10), torch.FloatTensor([0.0]).repeat(batch_size,1,10)), dim=2).to(device)
        common_log_probs = torch.log(torch.bmm(attention_weights, torch.exp(log_probs_stacked))).squeeze(1)

        # contrastive way
        # [b * m1 * m2] * [b * m2 * m3]
        # positive_attention_weights = torch.FloatTensor([1.0 / 20]).repeat(batch_size, 1, 10).to(device)
        # negative_attention_weights = torch.FloatTensor([1.0 / 20]).repeat(batch_size, 1, 10).to(device)
        # attention_weights = torch.cat((positive_attention_weights, negative_attention_weights), dim=2).to(device)
        # common_deterministic_probs = (torch.bmm(positive_attention_weights,torch.exp(log_probs_stacked)[:, :10, :]) - torch.bmm(negative_attention_weights, torch.exp(log_probs_stacked)[:, 10:, :]))#.squeeze(1)
        # common_deterministic_probs[common_deterministic_probs <= 1e-20] = 1e-35
        # normalizing_vals = torch.sum(common_deterministic_probs, -1, keepdim=True).reciprocal()
        # common_deterministic_probs = torch.bmm(normalizing_vals, common_deterministic_probs).squeeze(1)
        # common_log_probs = torch.log(common_deterministic_probs)



    if sample:
        prev = torch.multinomial(torch.exp(common_log_probs), num_samples=1)
        individual_prev = []
        for indiv_log_prob_element in indiv_log_probs_stacked:
            individual_prev.append(torch.multinomial(torch.exp(indiv_log_prob_element), num_samples=1))
        individual_prev = torch.stack(individual_prev)
        return prev, common_log_probs[0][prev], common_log_probs, log_probs_stacked, individual_prev, attention_weights
    else:
        log_probs_sel, prev = torch.topk(common_log_probs, k=1, dim=-1)
        _, individual_prev = torch.topk(indiv_log_probs_stacked, k=1, dim=-1)
        return prev, log_probs_sel, common_log_probs, log_probs_stacked, individual_prev, attention_weights  # , hidden_states_list



def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

def cut_seq_to_eos_cnt(sentence, remove_id=[-1]):
    cnt = 0
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            cnt+=1
        else:
            break
    return cnt


def torch_vec_to_str(x, tokenizer):
    xx = x.cpu().numpy()
    decode_str = [tokenizer.decode(cut_seq_to_eos(s)).encode('utf-8').decode('utf-8') for s in xx]
    return decode_str


###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.cum_cost = parent.cum_cost + cost if parent else cost
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras  # can hold, for example, attention weights
        self._sequence = None

    def __repr__(self):
        return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'


def beam_search_naive(model, input_ids, position_ids=None, token_type_ids=None, length=20, beam_width=3, device='cuda'):
    """
    currently it does NOT support batch parabllel
    """

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    all_decode, all_decode_losses = [], []
    for b in range(input_ids.shape[0]):
        next_fringe = [Node(parent=None, state=None, value=-1, cost=0.0, extras=input_ids[b:b+1])]
        results = []
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().to(device), [], []
            for n in next_fringe:
                if n.value == EOS_ID:
                    results.append(n)
                else:
                    fringe.extend([n]*beam_width)

                if not fringe:
                    break
                token_type_sliced = None if token_type_ids is None else token_type_ids[b:b+1]
                position_sliced = None if position_ids is None else position_ids[b:b+1]
                prev, probs, past = generate_next_token(model, input_ids[b:b+1], position_sliced, token_type_sliced,
                                                        n.extras, 1, beam_width, False, n.state)

                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))

            next_fringe = []
            for prev, log_probs, past, n in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=n, state=past, value=prev.item(), cost=log_probs, extras=prev.expand(1, 1))
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost, reverse=True)[:beam_width] # may move this into loop to save memory

        results.extend(next_fringe)
        results.sort(key=lambda n : n.cum_cost, reverse=True)
        best_result = results[0]
        decode, decode_loss = [], []
        while best_result.value != -1:
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            best_result = best_result.parent
        decode, decode_loss = decode[::-1], decode_loss[::-1]
        all_decode.append(decode)
        all_decode_losses.append(decode_loss)

    output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode],
                                             batch_first=True, padding_value=EOS_ID)
    return output
