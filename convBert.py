import torch
import torch.nn as nn
from transformers import BertTokenizer
import math
import torch.nn.functional as F

class ConvBERTConfig:
	def __init__(self, vocab_size=12000,
                 hidden_size=128, num_hidden_layers=2,
                 num_attention_heads=4,
                 num_conv_heads=1,
                 intermediate_size=256,
                 max_position_embeddings=256,
                 type_vocab_size=2,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 conv_kernel_size=9,
                 conv_glu=True,
                 pad_token_id=0):
				self.vocab_size=vocab_size
				self.hidden_size=hidden_size
				self.num_hidden_layers=num_hidden_layers
				self.num_attention_heads=num_attention_heads
				self.num_conv_heads=num_conv_heads
				self.intermediate_size=intermediate_size
				self.max_position_embeddings=max_position_embeddings
				self.type_vocab_size=type_vocab_size
				self.hidden_dropout_prob=hidden_dropout_prob
				self.attention_probs_dropout_prob=attention_probs_dropout_prob
				self.layer_norm_eps=layer_norm_eps
				self.conv_kernel_size=conv_kernel_size
				self.conv_glu=conv_glu
				self.pad_token_id=pad_token_id
				print("Configuration data loaded.")
				assert self.num_attention_heads > self.num_conv_heads >= 1
				assert self.hidden_size % self.num_attention_heads == 0
				assert self.conv_kernel_size % 2 == 1

class Embeddings(nn.Module):
    def __init__(self, config: ConvBERTConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding  = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, input_ids, token_type_ids=None):
        b, t = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(b, t, dtype=torch.long, device=input_ids.device)
        pos_ids = self.position_ids[:, :t].to(input_ids.device)
        x = ( self.word_embedding(input_ids)
            + self.pos_embedding(pos_ids)
            + self.tk_type_embedding(token_type_ids) )
        return self.dropout(self.ln(x))

class BertSelfAttention(nn.Module):
	def __init__(self, config: ConvBERTConfig, num_heads_override=None):
		super().__init__()
		self.num_heads = num_heads_override or config.num_attention_heads
		self.head_dim = config.hidden_size // self.num_heads
		self.hidden_size = self.head_dim * self.num_heads
		self.q = nn.Linear(config.hidden_size, config.hidden_size) # y = x*w^T+b
		self.k = nn.Linear(config.hidden_size, config.hidden_size)
		self.v = nn.Linear(config.hidden_size, config.hidden_size)
		self.out = nn.Linear(config.hidden_size, config.hidden_size)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, x, attention_mask=None):
		B, T, H = x.shape

		# From "all heads in one big vector" -> "batch of multiple smaller head vectors"
		def shape_proj(w): # w is a linear layer
			y = w(x)  # (B -> Batch, T -> sequence length, H -> Hidden Size)
			y = y.view(B, T, self.num_heads, self.head_dim)  # (B, T, nh -> number of heads, hd -> head dim)
			return y.transpose(1, 2)  # (B, nh, T, hd) -> transpose because  it's easier to batch over heads.

		q = shape_proj(self.q)
		k = shape_proj(self.k)
		v = shape_proj(self.v)

		scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
		if attention_mask is not None:
			scores = scores + attention_mask
		probs = F.softmax(scores, dim=-1)
		print("probs: ", probs)
		probs = self.dropout(probs)
		ctx = torch.matmul(probs, v) # applying attention scores to values
		ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim) # merge heads back
		return self.out(ctx)

class ConvBranch(nn.Module):
	def __init__(self, config: ConvBERTConfig, out_size: int):
		super().__init__()
		k = config.conv_kernel_size
		pad = k // 2
		self.depthwise = nn.Conv1d(H, H, kernel_size=k, padding=pad, groups=H)
		self.glu = config.conv_glu
		self.pointwise = nn.Conv1d(H, 2*out_size if self.glu else out_size, kernel_size=1)
		self.act = nn.SiLU()
		self.ln = nn.LayerNorm(out_size)

	def forward(self, x, pad_mask=None):
		y = x.transpose(1, 2)  # [B,H,T]
		if pad_mask is not None:
			m = (pad_mask == 0).float().squeeze(1).squeeze(1)  # [B,T]
			y = y * m.unsqueeze(1)
		y = self.depthwise(y)
		y = self.pointwise(y)
		if self.glu:
			a, g = y.chunk(2, dim=1)
			y = a * torch.sigmoid(g)
		y = self.act(y)
		y = y.transpose(1, 2)  # [B,T,C]
		return self.ln(y)

class TestSmoke():
	def testEmbeddings(self):
		def smokeTestEmbedding():
			vocab_size = 100  # means valid token IDs are 0..99
			pad_token_id = 0  # reserve 0 for [PAD]
			batch_size = 2
			seq_len = 5

			input_ids = torch.randint(
				low=1, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long
			)
			input_ids[0, -1] = pad_token_id
			token_type_ids = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)
			print("input_ids:\n", input_ids)
			print("token_type_ids:\n", token_type_ids)
			cfg = ConvBERTConfig()
			emb = Embeddings(cfg)
			out = emb.forward(input_ids, token_type_ids)
			print(out)

	def testBertSelfAttention(self):
		cfg = ConvBERTConfig()
		batch_size = 2
		seq_len = 5
		hidden_size =  cfg.hidden_size
		num_heads = cfg.num_attention_heads

		# Fake "token embeddings"
		x = torch.randn(batch_size, seq_len, hidden_size)

		bertSelfAttention = BertSelfAttention(cfg)
		bertSelfAttention.forward(x)

if __name__ == "__main__":
	x = torch.randn(4, 3)


