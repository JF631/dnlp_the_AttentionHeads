# convBert.py
# Implements ConvBERT-style blocks on top of a BERT skeleton, plus a hybrid loader
# that imports as many weights as possible from a HuggingFace BERT checkpoint.
#
# Key terms used in comments to match your terminology:
# - Embed Layer
# - BERT Layer (here: ConvBertLayer or BertLayer)
# - Linear Transformation Layer
# - Layer Flow

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# These imports are assumed to exist in your project (as in your original BERT file).
from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear Transformation Layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout applied to attention probabilities
        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.1))

    def transform(self, x, linear_layer):
        # Project hidden states then split into heads
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)  # [bs, seq_len, all_head_size]
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)  # [bs, heads, seq, head_dim]
        return proj

    def attention(self, key, query, value, attention_mask):
        # Scaled dot-product attention with mask
        # key, query, value: [bs, heads, seq, head_dim]
        S = torch.matmul(query, key.transpose(-2, -1))  # [bs, heads, seq, seq]
        scale = math.sqrt(self.attention_head_size)
        S = S / scale
        # attention_mask expected: [bs, 1, 1, seq]; 0 for tokens, large negative for pads
        S = S + attention_mask
        S = F.softmax(S, dim=-1)
        S = self.dropout(S)
        attn_value = torch.matmul(S, value)  # [bs, heads, seq, head_dim]
        return attn_value

    def forward(self, hidden_states, attention_mask):
        # Compute multi-head attention
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Self-Attention block
        self.self_attention = BertSelfAttention(config)

        # Add & Norm after attention
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)  # Linear Transformation Layer
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.attention_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # Feed-Forward block
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)  # Linear Transformation Layer
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)    # Linear Transformation Layer
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.out_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

    def _merge_heads(self, attn_value, bs, seq_len, hidden_size):
        # [bs, heads, seq, head_dim] -> [bs, seq, hidden]
        return attn_value.transpose(1, 2).contiguous().view(bs, seq_len, hidden_size)

    def add_norm(self, residual_input, sublayer_out, dense_layer, dropout, ln_layer):
        # Generic: Linear Transformation Layer -> Dropout -> Residual -> LayerNorm
        transformed = dense_layer(sublayer_out)
        dropped = dropout(transformed)
        return ln_layer(residual_input + dropped)

    def forward(self, hidden_states, attention_mask):
        """
        Layer Flow (BERT Layer):
        Input -> Self-Attention -> (Linear Transformation Layer + Dropout + Add&Norm) ->
                Feed-Forward -> (Linear Transformation Layer + Dropout + Add&Norm)
        """
        bs, seq_len, hidden_size = hidden_states.size()

        # Self-Attention
        attn_value = self.self_attention(hidden_states, attention_mask)                     # [bs, heads, seq, head_dim]
        attn_value = self._merge_heads(attn_value, bs, seq_len, hidden_size)               # [bs, seq, hidden]
        attention_out = self.add_norm(hidden_states, attn_value,
                                      self.attention_dense, self.attention_dropout, self.attention_layer_norm)

        # Feed-Forward
        interm = self.interm_af(self.interm_dense(attention_out))
        layer_out = self.add_norm(attention_out, interm, self.out_dense, self.out_dropout, self.out_layer_norm)
        return layer_out

class SpanConvBlock(nn.Module):
    """
    Convolution Block used inside a ConvBERT Layer.

    Layer Flow (Convolution Block):
    Input -> Depthwise Conv1d -> Pointwise Conv1d -> GELU -> Pointwise Conv1d -> Dropout -> Output
    """
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        k = getattr(config, "conv_kernel_size", 7)
        exp = getattr(config, "conv_expansion", 2)
        p = k // 2
        groups = getattr(config, "conv_groups", hidden)  # depthwise by default

        # Depthwise + Pointwise (1x1) convs
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=k, padding=p, groups=groups, bias=True)
        self.pw1 = nn.Conv1d(hidden, hidden * exp, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(hidden * exp, hidden, kernel_size=1)
        self.dropout = nn.Dropout(getattr(config, "conv_dropout_prob", getattr(config, "hidden_dropout_prob", 0.1)))

    @staticmethod
    def _binary_seq_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts attention_mask to [bs, seq] with 1 for tokens, 0 for padding.
        Accepts masks in shapes [bs, seq] (1/0) or extended [bs,1,1,seq] (0 or -inf-like for pads).
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask.to(dtype=torch.long)
        # extended mask: tokens -> 0, pads -> negative large
        m = (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.long)
        return m

    def forward(self, x, attention_mask=None):
        # x: [bs, seq, hidden]
        mask_1d = self._binary_seq_mask(attention_mask)  # [bs, seq] with 1 where token, 0 where pad
        if mask_1d is not None:
            mask_3d = mask_1d.unsqueeze(-1)  # [bs, seq, 1]
            x = x * mask_3d

        y = x.transpose(1, 2)          # [bs, hidden, seq]
        y = self.dw(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = y.transpose(1, 2)          # [bs, seq, hidden]
        y = self.dropout(y)

        if mask_1d is not None:
            y = y * mask_3d
        return y


class ConvBertLayer(nn.Module):
    """
    ConvBERT Layer.

    Layer Flow (ConvBERT Layer):
    Input -> Self-Attention Block -> Convolution Block -> Fusion (sum or concat+Linear) ->
            Feed-Forward -> (Linear Transformation Layer + Dropout + Add&Norm)
    """
    def __init__(self, config):
        super().__init__()
        # Self-Attention block
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)  # Linear Transformation Layer
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.attention_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # Convolution block
        self.span_conv = SpanConvBlock(config)

        # Fusion
        fuse = getattr(config, "conv_fuse", "sum")
        self.fuse_mode = fuse
        if fuse == "concat":
            self.fuse_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)  # Linear Transformation Layer

        # Feed-Forward block
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)  # Linear Transformation Layer
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)    # Linear Transformation Layer
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.out_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

    def _merge_heads(self, attn_value, bs, seq_len, hidden_size):
        return attn_value.transpose(1, 2).contiguous().view(bs, seq_len, hidden_size)

    def add_norm(self, residual_input, sublayer_out, dense_layer, dropout, ln_layer):
        transformed = dense_layer(sublayer_out)
        dropped = dropout(transformed)
        return ln_layer(residual_input + dropped)

    def forward(self, hidden_states, attention_mask):
        bs, seq_len, hidden_size = hidden_states.size()

        # --- Self-Attention Block (with Add&Norm) ---
        attn_value = self.self_attention(hidden_states, attention_mask)                     # [bs, heads, seq, head_dim]
        attn_value = self._merge_heads(attn_value, bs, seq_len, hidden_size)               # [bs, seq, hidden]
        attn_out = self.add_norm(hidden_states, attn_value,
                                 self.attention_dense, self.attention_dropout, self.attention_layer_norm)

        # --- Convolution Block ---
        conv_out = self.span_conv(hidden_states, attention_mask)                            # [bs, seq, hidden]

        # --- Fusion ---
        if self.fuse_mode == "concat":
            mixed = torch.cat([attn_out, conv_out], dim=-1)                                 # [bs, seq, 2*hidden]
            mixed = self.fuse_linear(mixed)                                                 # Linear Transformation Layer
        else:
            mixed = attn_out + conv_out                                                     # sum

        # --- Feed-Forward + Add&Norm ---
        interm = self.interm_af(self.interm_dense(mixed))
        layer_out = self.add_norm(mixed, interm, self.out_dense, self.out_dropout, self.out_layer_norm)
        return layer_out

class BertModel(BertPreTrainedModel):
    """
    The (Conv)BERT model returns final embeddings for each token in a sentence.

    Layer Flow (Model):
    Embed Layer -> a stack of N BERT Layers (BertLayer or ConvBertLayer) -> Pooler (Linear Transformation Layer + Tanh).
    """
    def __init__(self, config):
        if not hasattr(config, "name_or_path"):
            setattr(config, "name_or_path", "convbert-initialized")
        super().__init__(config)
        self.config = config

        # ---- Embed Layer ----
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.embed_dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # ---- Encoder: stack of BERT Layers ----
        use_conv = getattr(config, "use_convbert", True)  # default to True for this module
        LayerCls = ConvBertLayer if use_conv else BertLayer
        self.bert_layers = nn.ModuleList([LayerCls(config) for _ in range(config.num_hidden_layers)])

        # ---- Pooler for [CLS] token ----
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)  # Linear Transformation Layer
        self.pooler_af = nn.Tanh()

        self.init_weights()

    # ---- Embed Layer ----
    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # word embeddings
        inputs_embeds = self.word_embedding(input_ids)  # [bs, seq, hidden]

        # positional embeddings
        pos_ids = self.position_ids[:, :seq_length].expand(input_shape)  # [bs, seq]
        pos_embeds = self.pos_embedding(pos_ids)  # [bs, seq, hidden]

        # token type embeddings (set to zeros if caller doesn't supply)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        tok_type_embeds = self.tk_type_embedding(token_type_ids)

        # sum & norm & dropout
        embeddings = inputs_embeds + pos_embeds + tok_type_embeds
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        return embeddings

    # ---- Encoder ----
    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: output from Embed Layer [bs, seq, hidden]
        attention_mask: [bs, seq] with 1 for tokens, 0 for pads
        """
        # Extend attention mask for self-attention: [bs, 1, 1, seq]
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        # pass through BERT Layers
        for _, layer_module in enumerate(self.bert_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    # ---- Forward ----
    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] with 1 for tokens, 0 for pads
        """
        # Embed Layer
        embedding_output = self.embed(input_ids=input_ids)

        # Encoder (stack of BERT Layers / ConvBERT Layers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # Pooler (Linear Transformation Layer + Tanh) applied to [CLS]
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}


# For convenience, keep a ConvBertModel alias that always uses ConvBertLayer
class ConvBertModel(BertModel):
    def __init__(self, config):
        # Force conv usage
        if not hasattr(config, "use_convbert"):
            setattr(config, "use_convbert", True)
        else:
            config.use_convbert = True
        super().__init__(config)


__all__ = [
    "BertModel",
    "ConvBertModel",
    "BertLayer",
    "ConvBertLayer",
    "SpanConvBlock",
    "BertSelfAttention",
]

from types import SimpleNamespace

def _build_convbert_config_from_hf(hf_config, override: dict = None):
    """Create a SimpleNamespace config for our ConvBERT from an HF BertConfig."""
    override = override or {}
    cfg = SimpleNamespace(
        vocab_size=getattr(hf_config, "vocab_size", 30522),
        hidden_size=getattr(hf_config, "hidden_size", 768),
        num_attention_heads=getattr(hf_config, "num_attention_heads", 12),
        intermediate_size=getattr(hf_config, "intermediate_size", 3072),
        num_hidden_layers=getattr(hf_config, "num_hidden_layers", 12),
        max_position_embeddings=getattr(hf_config, "max_position_embeddings", 512),
        type_vocab_size=getattr(hf_config, "type_vocab_size", 2),
        pad_token_id=getattr(hf_config, "pad_token_id", 0),
        layer_norm_eps=getattr(hf_config, "layer_norm_eps", 1e-12),
        hidden_dropout_prob=getattr(hf_config, "hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=getattr(hf_config, "attention_probs_dropout_prob", 0.1),
        # ConvBERT specifics (random init):
        initializer_range=getattr(hf_config, "initializer_range", 0.02),
        use_convbert=True,
        conv_kernel_size=override.get("conv_kernel_size", 7),
        conv_groups=override.get("conv_groups", getattr(hf_config, "hidden_size", 768)),
        conv_expansion=override.get("conv_expansion", 2),
        conv_dropout_prob=override.get("conv_dropout_prob", getattr(hf_config, "hidden_dropout_prob", 0.1)),
        conv_fuse=override.get("conv_fuse", "sum"),
        name_or_path=getattr(hf_config, "name_or_path", None) or "bert-from-pretrained"
    )
    return cfg


def _copy_module_params(dst_mod: nn.Module, src_w: torch.Tensor | None, src_b: torch.Tensor | None):
    with torch.no_grad():
        if src_w is not None and hasattr(dst_mod, "weight") and dst_mod.weight is not None:
            dst_mod.weight.copy_(src_w)
        if src_b is not None and hasattr(dst_mod, "bias") and getattr(dst_mod, "bias") is not None:
            dst_mod.bias.copy_(src_b)


def _copy_layer_norm(dst_ln: nn.LayerNorm, src_weight: torch.Tensor, src_bias: torch.Tensor):
    with torch.no_grad():
        dst_ln.weight.copy_(src_weight)
        dst_ln.bias.copy_(src_bias)


def _load_hf_into_convbert(model: 'BertModel', hf_model: 'transformers.BertModel'):
    """Copy as many parameters as possible from HF BERT into our ConvBERT model.
    Embeddings, encoder attention/ffn/ln, and pooler are ported. Convolution blocks remain randomly initialized.
    """
    sd = hf_model.state_dict()

    # --- Embeddings ---
    _copy_module_params(model.word_embedding, sd["embeddings.word_embeddings.weight"], None)
    _copy_module_params(model.pos_embedding, sd["embeddings.position_embeddings.weight"], None)
    if "embeddings.token_type_embeddings.weight" in sd:
        _copy_module_params(model.word_embedding, sd.get("embeddings.word_embeddings.weight"), None)
        _copy_module_params(model.pos_embedding, sd.get("embeddings.position_embeddings.weight"), None)
        _copy_module_params(model.tk_type_embedding, sd.get("embeddings.token_type_embeddings.weight"), None)
        _copy_layer_norm(model.embed_layer_norm,
                         sd["embeddings.LayerNorm.weight"],
                         sd["embeddings.LayerNorm.bias"])

    # --- Encoder layers ---
    for i, layer in enumerate(model.bert_layers):
        # Self-Attention (Q,K,V)
        _copy_module_params(layer.self_attention.query,
                            sd[f"encoder.layer.{i}.attention.self.query.weight"],
                            sd[f"encoder.layer.{i}.attention.self.query.bias"])
        _copy_module_params(layer.self_attention.key,
                            sd[f"encoder.layer.{i}.attention.self.key.weight"],
                            sd[f"encoder.layer.{i}.attention.self.key.bias"])
        _copy_module_params(layer.self_attention.value,
                            sd[f"encoder.layer.{i}.attention.self.value.weight"],
                            sd[f"encoder.layer.{i}.attention.self.value.bias"])

        # Attention output dense + LayerNorm
        _copy_module_params(layer.attention_dense,
                            sd[f"encoder.layer.{i}.attention.output.dense.weight"],
                            sd[f"encoder.layer.{i}.attention.output.dense.bias"])
        _copy_layer_norm(layer.attention_layer_norm,
                         sd[f"encoder.layer.{i}.attention.output.LayerNorm.weight"],
                         sd[f"encoder.layer.{i}.attention.output.LayerNorm.bias"])

        # Feed-Forward
        _copy_module_params(layer.interm_dense,
                            sd[f"encoder.layer.{i}.intermediate.dense.weight"],
                            sd[f"encoder.layer.{i}.intermediate.dense.bias"])
        _copy_module_params(layer.out_dense,
                            sd[f"encoder.layer.{i}.output.dense.weight"],
                            sd[f"encoder.layer.{i}.output.dense.bias"])
        _copy_layer_norm(layer.out_layer_norm,
                         sd[f"encoder.layer.{i}.output.LayerNorm.weight"],
                         sd[f"encoder.layer.{i}.output.LayerNorm.bias"])

    # --- Pooler ---
    if "pooler.dense.weight" in sd:
        _copy_module_params(model.pooler_dense, sd["pooler.dense.weight"], sd["pooler.dense.bias"])


# Attach a user-friendly classmethod on BertModel
def _attach_from_pretrained_on_BertModel():
    def from_pretrained(cls, name_or_path: str = "bert-base-uncased", local_files_only: bool = False, **kwargs):
        from transformers import BertModel as HFBertModel  # HF is only needed at load time
        # 1) load HF BERT
        hf_model = HFBertModel.from_pretrained(name_or_path, local_files_only=local_files_only, **kwargs)
        hf_config = hf_model.config
        # 2) build our ConvBERT config
        override = dict(
            conv_kernel_size=kwargs.pop("conv_kernel_size", 7),
            conv_groups=kwargs.pop("conv_groups", getattr(hf_config, "hidden_size", 768)),
            conv_expansion=kwargs.pop("conv_expansion", 2),
            conv_dropout_prob=kwargs.pop("conv_dropout_prob", getattr(hf_config, "hidden_dropout_prob", 0.1)),
            conv_fuse=kwargs.pop("conv_fuse", "sum"),
        )
        conv_cfg = _build_convbert_config_from_hf(hf_config, override=override)
        conv_cfg.name_or_path = name_or_path
        # 3) init our model (this creates ConvBertLayers, SpanConvBlocks get random init)
        model = cls(conv_cfg)
        # 4) copy common weights from HF BERT
        _load_hf_into_convbert(model, hf_model)
        return model

    setattr(BertModel, "from_pretrained", classmethod(from_pretrained))

# execute the attachment at import-time
_attach_from_pretrained_on_BertModel()
