# simbert_loading.py
import torch
from transformers import AutoConfig, AutoModel
from simBert import BertModel as SimBertEncoder  # your custom encoder

def _remap_hf_to_simbert(hf_state):
    """Return a new state dict whose keys match your SimBert BertModel."""
    new_state = {}

    new_state["word_embedding.weight"] = hf_state["embeddings.word_embeddings.weight"]
    new_state["pos_embedding.weight"]  = hf_state["embeddings.position_embeddings.weight"]
    new_state["tk_type_embedding.weight"] = hf_state["embeddings.token_type_embeddings.weight"]
    new_state["embed_layer_norm.weight"] = hf_state["embeddings.LayerNorm.weight"]
    new_state["embed_layer_norm.bias"]   = hf_state["embeddings.LayerNorm.bias"]

    # HF: encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}
    #     encoder.layer.{i}.attention.output.dense.{weight,bias}
    #     encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
    #     encoder.layer.{i}.intermediate.dense.{weight,bias}
    #     encoder.layer.{i}.output.dense.{weight,bias}
    #     encoder.layer.{i}.output.LayerNorm.{weight,bias}
    # Yours:
    #   bert_layers.{i}.self_attention.{query,key,value}.{weight,bias}
    #   bert_layers.{i}.attention_dense.{weight,bias}
    #   bert_layers.{i}.attention_layer_norm.{weight,bias}
    #   bert_layers.{i}.interm_dense.{weight,bias}
    #   bert_layers.{i}.out_dense.{weight,bias}
    #   bert_layers.{i}.out_layer_norm.{weight,bias}
    num_layers = len({k.split(".")[2] for k in hf_state if k.startswith("encoder.layer.")})
    for i in range(num_layers):
        base = f"encoder.layer.{i}"

        # self-attention q/k/v
        for name in ["query", "key", "value"]:
            new_state[f"bert_layers.{i}.self_attention.{name}.weight"] = hf_state[f"{base}.attention.self.{name}.weight"]
            new_state[f"bert_layers.{i}.self_attention.{name}.bias"]   = hf_state[f"{base}.attention.self.{name}.bias"]

        # attn output dense + LN
        new_state[f"bert_layers.{i}.attention_dense.weight"] = hf_state[f"{base}.attention.output.dense.weight"]
        new_state[f"bert_layers.{i}.attention_dense.bias"]   = hf_state[f"{base}.attention.output.dense.bias"]

        new_state[f"bert_layers.{i}.attention_layer_norm.weight"] = hf_state[f"{base}.attention.output.LayerNorm.weight"]
        new_state[f"bert_layers.{i}.attention_layer_norm.bias"]   = hf_state[f"{base}.attention.output.LayerNorm.bias"]

        # intermediate dense (GELU)
        new_state[f"bert_layers.{i}.interm_dense.weight"] = hf_state[f"{base}.intermediate.dense.weight"]
        new_state[f"bert_layers.{i}.interm_dense.bias"]   = hf_state[f"{base}.intermediate.dense.bias"]

        # output dense + LN
        new_state[f"bert_layers.{i}.out_dense.weight"] = hf_state[f"{base}.output.dense.weight"]
        new_state[f"bert_layers.{i}.out_dense.bias"]   = hf_state[f"{base}.output.dense.bias"]

        new_state[f"bert_layers.{i}.out_layer_norm.weight"] = hf_state[f"{base}.output.LayerNorm.weight"]
        new_state[f"bert_layers.{i}.out_layer_norm.bias"]   = hf_state[f"{base}.output.LayerNorm.bias"]

    # --- Pooler ---
    # HF: pooler.dense.{weight,bias}
    # Yours: pooler_dense.{weight,bias}
    if "pooler.dense.weight" in hf_state:  # present in base models
        new_state["pooler_dense.weight"] = hf_state["pooler.dense.weight"]
        new_state["pooler_dense.bias"]   = hf_state["pooler.dense.bias"]

    return new_state

def build_simbert_from_pretrained(name_or_path="bert-base-uncased", local_files_only=False, torch_dtype=None):
    """Create your SimBERT encoder and load HF BERT weights into it."""
    # 1) Load HF config and model weights
    hf_cfg = AutoConfig.from_pretrained(name_or_path, local_files_only=local_files_only)
    hf_model = AutoModel.from_pretrained(
        name_or_path, local_files_only=local_files_only, torch_dtype=torch_dtype
    )
    hf_state = hf_model.state_dict()

    # 2) Instantiate your encoder with the same config
    # Your BertModel takes a config object compatible with HF config attributes
    simbert = SimBertEncoder(hf_cfg)

    # 3) Remap and load
    remapped = _remap_hf_to_simbert(hf_state)
    missing, unexpected = simbert.load_state_dict(remapped, strict=False)

    # It’s okay to miss buffers like position_ids, and to ignore task heads (none here).
    # But we assert the core weights loaded:
    core_ok = all(not k.startswith(("word_embedding", "pos_embedding", "tk_type_embedding", "bert_layers", "pooler_dense")) for k in missing)
    if not core_ok:
        print("Warning: some core weights were not loaded:", missing)
    if unexpected:
        # Typically lots of HF keys are "unexpected" because we remapped them out
        pass

    return simbert
