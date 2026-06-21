"""
Hari 1 - Forward Pass & Attention Weights
============================================
Jalankan forward pass pada model GPT-2, ambil attention weights mentah
dari salah satu layer, dan lihat bentuk (shape) tensornya.

Jalankan: python 03_attention_weights.py
Requirement: pip install transformers torch
"""

import torch
from transformers import AutoModel, AutoTokenizer


def get_attention_weights(model_name: str = "gpt2", text: str = "Hello world"):
    print(f"\n{'='*60}")
    print(f"Forward pass + attention weights: {model_name}")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager",  # wajib untuk bisa keluarkan attention weights
    )
    model.eval()

    inputs = tok(text, return_tensors="pt")
    print(f"Input text : {text}")
    print(f"Tokens     : {tok.tokenize(text)}")
    print(f"Token IDs  : {inputs['input_ids'].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # tuple, satu tensor per layer
    print(f"\nJumlah layer dengan attention: {len(attentions)}")

    first_layer_attn = attentions[0]
    print(f"Shape attention layer pertama: {first_layer_attn.shape}")
    print("  -> [batch, num_heads, seq_len, seq_len]")

    # Lihat attention weights dari head pertama, layer pertama
    head0 = first_layer_attn[0, 0]  # [seq_len, seq_len]
    print(f"\nAttention weights head-0, layer-0:\n{head0}")
    print("\nCatatan: tiap baris berjumlah 1.0 (hasil softmax per baris)")
    print(f"Jumlah baris pertama: {head0[0].sum().item():.4f}")


if __name__ == "__main__":
    get_attention_weights("gpt2", "Kucing itu duduk")
