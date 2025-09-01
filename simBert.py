import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    """
    Multi-head self-attention block used inside BERT.

    This module projects the input hidden states into query, key, and value
    tensors, computes scaled dot-product attention with an additive mask, and
    returns the attended values per head.
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration object with at least the attributes
                - ``hidden_size`` (int): Model hidden dimension.
                - ``num_attention_heads`` (int): Number of attention heads.
                - ``attention_probs_dropout_prob`` (float): Dropout prob for
                  attention probabilities.
        """
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        """
        Project `x` with the provided linear layer and reshape it into
        multi-head format.

        Args:
            x (torch.Tensor): Input tensor of shape ``[batch_size, seq_len, hidden_size]``.
            Linear_layer (nn.Linear): One of ``self.query``, ``self.key``, or ``self.value``.

        Returns:
            torch.Tensor: Projected tensor of shape
            `[batch_size, num_heads, seq_len, head_size]`.
        """
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        """
        Compute scaled dot-product attention.

        Note:
            The argument order here is ``(key, query, value, attention_mask)`` to
            match the existing call sites in this codebase.

        Args:
            key (torch.Tensor): Key tensor of shape
                ``[batch_size, num_heads, seq_len, head_size]``.
            query (torch.Tensor): Query tensor of shape
                ``[batch_size, num_heads, seq_len, head_size]``.
            value (torch.Tensor): Value tensor of shape
                ``[batch_size, num_heads, seq_len, head_size]``.
            attention_mask (torch.Tensor): Additive mask broadcastable to
                ``[batch_size, num_heads, seq_len, seq_len]`` where masked
                positions contain a large negative value (e.g., ``-1e4``).

        Returns:
            torch.Tensor: Context tensor of shape
            ``[batch_size, num_heads, seq_len, head_size]``.
        """
        S = torch.matmul(query, key.transpose(-2, -1))
        S = S / math.sqrt(self.attention_head_size)
        S = S + attention_mask
        S = F.softmax(S, dim=-1)
        S = self.dropout(S)
        rtrn = torch.matmul(S, value)
        return rtrn

    def forward(self, hidden_states, attention_mask):
        """
        Run multi-head self-attention over `hidden_states`.

        Args:
            hidden_states (torch.Tensor): Input embeddings of shape
                ``[batch_size, seq_len, hidden_size]``.
            Attention_mask (torch.Tensor): Additive mask broadcastable to
                ``[batch_size, num_heads, seq_len, seq_len]`` (see ``attention``).

        Returns:
            torch.Tensor: Attention output per head of shape
            ``[batch_size, num_heads, seq_len, head_size]``.
        """
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value

class BertLayer(nn.Module):
    """
    Single Transformer encoder layer consisting of:
    - Multi-head self-attention with Add & Norm.
    - Position-wise feed-forward network with Add & Norm.
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration object with attributes used by BERT layers,
                including ``hidden_size``, ``intermediate_size``,
                ``hidden_dropout_prob``, and ``layer_norm_eps``.
        """
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        # add-norm (post-attention)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed-forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # add-norm (post-FFN)
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        Apply the sequence: Dense → Dropout → Residual Add → LayerNorm.

        Args:
            input (torch.Tensor): Residual/skip connection input of shape
                ``[batch_size, seq_len, hidden_size]``.
            output (torch.Tensor): Sub-layer output to be transformed of shape
                ``[batch_size, seq_len, hidden_size]``.
            dense_layer (nn.Linear): Linear layer to apply to ``output``.
            dropout (nn.Dropout): Dropout module applied after the linear layer.
            ln_layer (nn.LayerNorm): LayerNorm applied after residual addition.

        Returns:
            torch.Tensor: Normalized tensor of shape ``[batch_size, seq_len, hidden_size]``.
        """
        transformed_output = dense_layer(output)
        dropped_output = dropout(transformed_output)
        residual = input + dropped_output
        normalized_output = ln_layer(residual)
        return normalized_output

    def forward(self, hidden_states, attention_mask):
        """
        Transformer encoder layer forward pass.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape
                ``[batch_size, seq_len, hidden_size]``.
            attention_mask (torch.Tensor): Additive mask broadcastable to
                ``[batch_size, num_heads, seq_len, seq_len]``.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, hidden_size]``.
        """
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
    Minimal BERT encoder with embeddings, stacked Transformer layers, and a pooler.

    The forward pass returns a dictionary with:

    - `last_hidden_state`: `[batch_size, seq_len, hidden_size]`
    - `pooler_output`: `[batch_size, hidden_size]` — tanh of the [CLS] token
    """

    def __init__(self, config):
        """
        Args:
            config: A configuration object with standard BERT fields such as
                ``vocab_size``, ``hidden_size``, ``pad_token_id``,
                ``max_position_embeddings``, ``type_vocab_size``,
                ``num_hidden_layers``, ``layer_norm_eps``, and ``hidden_dropout_prob``.
        """
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
        """
        Build token, position, and token-type embeddings, then apply LayerNorm
        and dropout.

        Args:
            input_ids (torch.Tensor): Token IDs of shape ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Embedded inputs of shape ``[batch_size, seq_len, hidden_size]``.
        """
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
        """
        Run the stacked Transformer encoder layers.

        Args:
            hidden_states (torch.Tensor): Embedded inputs of shape
                ``[batch_size, seq_len, hidden_size]``.
            attention_mask (torch.Tensor): Attention mask with ``1`` for tokens
                and ``0`` for padding of shape ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Final hidden states of shape
            ``[batch_size, seq_len, hidden_size]``.
        """
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )
        for layer_module in self.bert_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        Full encoder forward pass.

        Args:
            input_ids (torch.Tensor): Token IDs of shape ``[batch_size, seq_len]``.
            attention_mask (torch.Tensor): Mask with ``1`` for tokens and ``0`` for pads
                of shape ``[batch_size, seq_len]``.

        Returns:
            Dict[str, torch.Tensor]:
                - ``last_hidden_state``: ``[batch_size, seq_len, hidden_size]``
                - ``pooler_output``:     ``[batch_size, hidden_size]``
        """
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_af(self.pooler_dense(first_tk))
        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}


N_SENTIMENT_CLASSES = 5
N_ETPC_CLASSES = 7


def mean_pool(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings over non-padding positions.

    Args:
        last_hidden_state (torch.Tensor): Hidden states of shape
            ``[batch_size, seq_len, hidden_size]``.
        attention_mask (torch.Tensor): Mask with ``1`` for tokens and ``0`` for pads
            of shape ``[batch_size, seq_len]``.

    Returns:
        torch.Tensor: Sentence embedding of shape ``[batch_size, hidden_size]``.
    """
    # attention_mask: [B, L] with 1 for tokens, 0 for pads
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)          # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                # [B,1]
    return summed / counts


def max_pool(last_hidden_state, attention_mask):
    """
    Max-pool token embeddings over non-padding positions.

    Args:
        last_hidden_state (torch.Tensor): ``[batch_size, seq_len, hidden_size]``.
        attention_mask (torch.Tensor): ``[batch_size, seq_len]`` with ``1`` for tokens.

    Returns:
        torch.Tensor: Sentence embedding of shape ``[batch_size, hidden_size]``.
    """
    mask = (attention_mask == 1).unsqueeze(-1)
    masked = last_hidden_state.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def pair_features(u, v):
    """
    Build  pair features.

    Args:
        u (torch.Tensor): First sentence embedding ``[batch_size, hidden_size]``.
        v (torch.Tensor): Second sentence embedding ``[batch_size, hidden_size]``.

    Returns:
        torch.Tensor: Concatenation ``[u, v, |u - v|]`` of shape
        ``[batch_size, 3 * hidden_size]``.
    """
    return torch.cat([u, v, torch.abs(u - v)], dim=-1)


class SimBertMultitask(nn.Module):
    """
    Siamese multitask model with a weight-tied encoder.

    Tasks supported:
        1. Sentence-level sentiment classification (5 classes).
        2. Paraphrase detection (binary; single logit with BCE-with-logits loss).
        3. Paraphrase type classification (7 labels; typically multi-label).

    Notes:
        - The encoder weights are shared by using a single ``BertModel`` instance.
        - If ``config.option == 'finetune'``, encoder parameters are unfrozen.
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration object with ``hidden_size``, ``hidden_dropout_prob``,
                and task-specific options like ``sim_pool`` ("mean" | "max" | "cls") and
                ``option`` ("finetune" to train the encoder, anything else to freeze).
        """
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
        """
        Encode a sentence and pool token embeddings into a fixed-size vector.

        Args:
            input_ids (torch.Tensor): ``[batch_size, seq_len]``.
            attention_mask (torch.Tensor): ``[batch_size, seq_len]`` mask (1 = token).

        Returns:
            torch.Tensor: Pooled sentence embeddings ``[batch_size, hidden_size]``.
        """
        out = self.encoder(input_ids, attention_mask)
        if self.sim_pool == "cls":
            emb = out["pooler_output"]  # [B,H]
        elif self.sim_pool == "max":
            emb = max_pool(out["last_hidden_state"], attention_mask)  # [B,H]
        else:
            emb = mean_pool(out["last_hidden_state"], attention_mask) # [B,H]
        return emb

    def predict_sentiment(self, input_ids, attention_mask):
        """
        Predict 5-way sentiment logits for a batch of sentences.

        Args:
            input_ids (torch.Tensor): ``[batch_size, seq_len]``.
            attention_mask (torch.Tensor): ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Logits of shape ``[batch_size, 5]``.
        """
        u = self.sentence_embed(input_ids, attention_mask)  # [B,H]
        u = self.dropout(u)
        logits = self.sentiment_classifier(u)               # [B,5]
        return logits

    def predict_paraphrase(self, ids1, m1, ids2, m2):
        """
        Predict a single logit per pair for paraphrase detection.

        Args:
            ids1 (torch.Tensor): First sentence IDs ``[batch_size, seq_len]``.
            m1 (torch.Tensor): First sentence mask ``[batch_size, seq_len]``.
            ids2 (torch.Tensor): Second sentence IDs ``[batch_size, seq_len]``.
            m2 (torch.Tensor): Second sentence mask ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Logits ``[batch_size]`` (use with BCE-with-logits).
        """
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        z = pair_features(u, v)
        z = self.dropout(z)
        logit = self.qqp_head(z).squeeze(-1)                # [B]
        return logit  # unnormalized (use BCE-with-logits)

    def predict_similarity(self, ids1, m1, ids2, m2):
        """
        Predict cosine-similarity-based scores in the range ``[0, 5]``.

        Args:
            ids1 (torch.Tensor): First sentence IDs ``[batch_size, seq_len]``.
            m1 (torch.Tensor): First sentence mask ``[batch_size, seq_len]``.
            ids2 (torch.Tensor): Second sentence IDs ``[batch_size, seq_len]``.
            m2 (torch.Tensor): Second sentence mask ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Similarity scores ``[batch_size]``.
        """
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        cos = F.cosine_similarity(u, v)                     # [-1,1]
        sim = (cos + 1.0) * 2.5                             # [0,5]
        return sim                                          # [B]

    def predict_paraphrase_types(self, ids1, m1, ids2, m2):
        """
        Predict logits for 7 paraphrase type labels given a sentence pair.

        Args:
            ids1 (torch.Tensor): First sentence IDs ``[batch_size, seq_len]``.
            m1 (torch.Tensor): First sentence mask ``[batch_size, seq_len]``.
            ids2 (torch.Tensor): Second sentence IDs ``[batch_size, seq_len]``.
            m2 (torch.Tensor): Second sentence mask ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Logits of shape ``[batch_size, 7]``.
        """
        u = self.sentence_embed(ids1, m1)
        v = self.sentence_embed(ids2, m2)
        z = pair_features(u, v)
        z = self.dropout(z)
        logits = self.etpc_head(z)                          # [B,7]
        return logits
