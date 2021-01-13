import torch.nn.functional as F


WEIGHTS_NAME = 'coordinator_weights.bin'
import collections
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class NaiveCoordinator(nn.Module):

    def __init__(self, ndocs = 20, hidden_dim = 768, nvocab = 50257):
        super(NaiveCoordinator, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(ndocs * hidden_dim, ndocs * hidden_dim // 2)# // 2)
        self.fc = nn.Linear(ndocs * hidden_dim // 2, ndocs) # // 2

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc(x), dim=-1)
        return x


class AttnCoordinator(nn.Module):

    def __init__(self, config, lm_head):
        super(AttnCoordinator, self).__init__()
        self.wce = nn.Embedding(config.n_clusters, config.n_embd)

        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.initializer_range = config.initializer_range
        self.n_embd = config.n_embd

        self.apply(self.init_weights)

        self.agg_weights = nn.Linear(config.n_embd* 20, 20, bias=False)

        # cluster embeddings

    def forward(self, inputs_embeds, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        cluster_token_ids = torch.zeros(inputs_embeds.shape[:2], device=inputs_embeds.device)
        cluster_token_ids[:, 10:] = 1
        cluster_token_embedding = self.wce(cluster_token_ids.to(torch.long))

        input_shape = inputs_embeds.size()
        hidden_states = inputs_embeds + cluster_token_embedding
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)

        attn_weights = F.softmax(self.agg_weights(hidden_states.view(hidden_states.shape[0], -1)), -1)

        return attn_weights.unsqueeze(1)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class ContrastAttnCoordinator(nn.Module):
    """
        for stable coordinator training
    """

    def __init__(self, config, lm_head,damp=False):
        super(ContrastAttnCoordinator, self).__init__()
        self.wce = nn.Embedding(config.n_clusters, config.n_embd)

        block = CoordBlock(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.initializer_range = config.initializer_range
        self.n_embd = config.n_embd

        self.apply(self.init_weights)

        self.agg_weights_pos = nn.Linear(config.n_embd* 20, 10, bias=False)
        self.agg_weights_neg = nn.Linear(config.n_embd* 20, 10, bias=False)

        self.attn_balancer = nn.Linear(config.n_embd * 20, 1, bias=True)
        self.damp = damp

    def forward(self, inputs_embeds, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        cluster_token_ids = torch.zeros(inputs_embeds.shape[:2], device=inputs_embeds.device)
        cluster_token_ids[:, 10:] = 1
        cluster_token_embedding = self.wce(cluster_token_ids.to(torch.long))

        input_shape = inputs_embeds.size()
        hidden_states = inputs_embeds + cluster_token_embedding
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)

        # constraining negativity
        attn_weights_pos = F.softmax(self.agg_weights_pos(hidden_states.view(hidden_states.shape[0], -1)), -1)
        attn_weights_neg = F.softmax(self.agg_weights_neg(hidden_states.view(hidden_states.shape[0], -1)), -1)

        attn_balancer_logit = self.attn_balancer(hidden_states.view(hidden_states.shape[0], -1))
        if self.damp:
            attn_balancer = -(torch.exp(2*attn_balancer_logit) - 0.5)/(torch.exp(2*attn_balancer_logit) + 1.0)
        else:
            attn_balancer = torch.tanh(attn_balancer_logit)

        return attn_weights_pos.unsqueeze(1), attn_weights_neg.unsqueeze(1), attn_balancer #hidden_states, common_log_probs,

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class FusedCoordinator(nn.Module):

    def __init__(self, config, lm_head):
        super(FusedCoordinator, self).__init__()
        self.wce = nn.Embedding(config.n_clusters, config.n_embd)
        #
        self.lm_head_weights = nn.Parameter(lm_head, requires_grad=False)
        self.decoder = nn.Linear(lm_head.shape[1], lm_head.shape[0], bias=False)
        self.decoder.weight = self.lm_head_weights  # Tied weights

        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.initializer_range = config.initializer_range
        self.n_embd = config.n_embd

        self.apply(self.init_weights)

        self.agg_weights = nn.Linear(config.n_embd * 20, 20, bias=False)

        self.common_h_proj_1 = nn.Linear(config.n_embd * 20, config.n_embd, bias=True)
        self.common_h_proj_2 = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.common_h_weight = nn.Linear(config.n_embd * 20, 1, bias=True)

    def forward(self, inputs_embeds, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        cluster_token_ids = torch.zeros(inputs_embeds.shape[:2], device=inputs_embeds.device)
        cluster_token_ids[:, 10:] = 1
        cluster_token_embedding = self.wce(cluster_token_ids.to(torch.long))

        input_shape = inputs_embeds.size()
        hidden_states = inputs_embeds + cluster_token_embedding
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)

        attn_weights = F.softmax(self.agg_weights(hidden_states.view(hidden_states.shape[0], -1)), -1)

        common_h = self.common_h_proj_1(hidden_states.view(hidden_states.shape[0], -1))
        common_h = F.relu(common_h)
        common_h = self.common_h_proj_2(common_h)
        common_gen_log_probs = F.log_softmax(self.decoder(common_h), -1)
        common_h_weight = torch.sigmoid(self.common_h_weight(hidden_states.view(hidden_states.shape[0], -1)))

        return attn_weights.unsqueeze(1), common_h_weight, common_gen_log_probs

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class CoordBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(CoordBlock, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = CoordAttention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CoordAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(CoordAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("bias", (torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)    # point out by Yen-Chun, FP16 overflow

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)    # point out by Yen-Chun, FP16 overflow

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class AttnCoordConfig(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = AttnCoordConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPT2Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits

def load_coordinator(args, type='fused'):

    if type=='naive':
        model = NaiveCoordinator(ndocs=20, hidden_dim=768)
    if type == 'attention':
        model = AttnCoordinator(config=args, lm_head=args.gpt2_lm_head) #config=args.coord_config
    if type == 'contrast_attention' or type == 'contrast_attention_sim':
        model = ContrastAttnCoordinator(config=args, lm_head=args.gpt2_lm_head, damp=args.use_contrast_damp)  # config=args.coord_config
    if type == 'fused':
        model = FusedCoordinator(config=args, lm_head=args.gpt2_lm_head) #config=args.coord_config

    if args.coordinator_model is not None:
        model.load_state_dict(torch.load(args.coordinator_model))

    model.to(args.device)
    return model
