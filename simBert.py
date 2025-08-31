import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for Key, value, Query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # dropout on attention probs
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # project hidden_state to multi-heads
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)  # [bs, heads, seq, head_size]
        return proj

    def attention(self, key, query, value, attention_mask):
        # scaled dot-product attention
        S = torch.matmul(query, key.transpose(-2, -1))
        S = S / math.sqrt(self.attention_head_size)
        S = S + attention_mask  # mask pads with large negative
        S = F.softmax(S, dim=-1)
        S = self.dropout(S)
        rtrn = torch.matmul(S, value)  # [bs, heads, seq, head_size]
        return rtrn

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        transformed_output = dense_layer(output)
        dropped_output = dropout(transformed_output)
        residual = input + dropped_output
        normalized_output = ln_layer(residual)
        return normalized_output

    def forward(self, hidden_states, attention_mask):
        # self-attn
        attention_out = self.self_attention(hidden_states, attention_mask)
        # concat heads back to [bs, seq, hidden]
        bs, _, seq, dh = attention_out.shape
        attention_out = attention_out.transpose(1, 2).contiguous().view(bs, seq, -1)
        attention_out = self.add_norm(
            hidden_states, attention_out,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm
        )
        # feed-forward
        interm_out = self.interm_af(self.interm_dense(attention_out))
        layer_out = self.add_norm(
            attention_out, interm_out,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm
        )
        return layer_out


class BertModel(BertPreTrainedModel):
    """
    Returns:
      {
        "last_hidden_state": [bs, seq, hidden],
        "pooler_output":     [bs, hidden]   (tanh on [CLS])
      }
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # pooler
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)  # [bs, seq, hidden]
        pos_ids = self.position_ids[:, :seq_length].expand(input_shape)  # [bs, seq]
        pos_embeds = self.pos_embedding(pos_ids)
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        embeddings = inputs_embeds + pos_embeds + tk_type_embeds
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        return embeddings

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )
        for layer_module in self.bert_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_af(self.pooler_dense(first_tk))
        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}


N_SENTIMENT_CLASSES = 5
N_ETPC_CLASSES = 7

def mean_pool(last_hidden_state, attention_mask):
    # attention_mask: [B, L] with 1 for tokens, 0 for pads
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)          # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                # [B,1]
    return summed / counts

def max_pool(last_hidden_state, attention_mask):
    mask = (attention_mask == 1).unsqueeze(-1)
    masked = last_hidden_state.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values

def pair_features(u, v):
    # SBERT classifier features: [u, v, |u-v|]
    return torch.cat([u, v, torch.abs(u - v)], dim=-1)


class SimBertMultitask(nn.Module):
    """
    Siamese (SBERT-style) multitask model with a weight-tied encoder.
    Uses the local BertModel defined above as the shared encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel(config)  # tie weights by reusing this single instance
        for p in self.encoder.parameters():
            p.requires_grad = (config.option == "finetune")

        self.hidden_size = config.hidden_size
        self.sim_pool = getattr(config, "sim_pool", "mean").lower()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Heads
        self.sentiment_classifier = nn.Linear(self.hidden_size, N_SENTIMENT_CLASSES)
        self.qqp_head  = nn.Linear(3 * self.hidden_size, 1)              # binary paraphrase
        self.etpc_head = nn.Linear(3 * self.hidden_size, N_ETPC_CLASSES) # 7-label multilabel

    def sentence_embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        if self.sim_pool == "cls":
            emb = out["pooler_output"]  # [B,H]
        elif self.sim_pool == "max":
            emb = max_pool(out["last_hidden_state"], attention_mask)  # [B,H]
        else:
            emb = mean_pool(out["last_hidden_state"], attention_mask) # [B,H]
        return emb

    def predict_sentiment(self, input_ids, attention_mask):
        u = self.sentence_embed(input_ids, attention_mask)  # [B,H]
        u = self.dropout(u)
        logits = self.sentiment_classifier(u)               # [B,5]
        return logits

    def predict_paraphrase(self, ids1, m1, ids2, m2):
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        z = pair_features(u, v)
        z = self.dropout(z)
        logit = self.qqp_head(z).squeeze(-1)                # [B]
        return logit  # unnormalized (use BCE-with-logits)

    def predict_similarity(self, ids1, m1, ids2, m2):
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        cos = F.cosine_similarity(u, v)                     # [-1,1]
        sim = (cos + 1.0) * 2.5                             # [0,5]
        return sim                                          # [B]

    def predict_paraphrase_types(self, ids1, m1, ids2, m2):
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        z = pair_features(u, v)
        z = self.dropout(z)
        logits = self.etpc_head(z)                          # [B,7]
        return logits
