"""
Hari 1 - Eksplorasi Arsitektur Model
=====================================
Load model GPT-2 kecil, lihat susunan layer-nya (embedding, attention,
feed-forward), dan hitung total parameter.

Jalankan: python 02_model_architecture.py
Requirement: pip install transformers torch
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


def inspect_model(model_name: str = "gpt2"):
    print(f"\n{'='*60}")
    print(f"Arsitektur model: {model_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameter: {total_params:,}")

    # Breakdown beberapa komponen kunci
    config = model.config
    print(f"\nKonfigurasi kunci:")
    print(f"  vocab_size  : {config.vocab_size}")
    print(f"  d_model     : {config.n_embd}")
    print(f"  num_layers  : {config.n_layer}")
    print(f"  num_heads   : {config.n_head}")
    print(f"  max_position: {config.n_positions}")

    # Hitung kontribusi parameter embedding table
    embedding_params = config.vocab_size * config.n_embd
    print(f"\nParameter di token embedding table (wte): {embedding_params:,}")
    print(f"Proporsi dari total model: {embedding_params/total_params*100:.1f}%")


if __name__ == "__main__":
    inspect_model("gpt2")
