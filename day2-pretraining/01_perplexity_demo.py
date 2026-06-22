"""
Hari 2 - Causal LM Loss & Perplexity
======================================
Demonstrasi bagaimana model pre-trained mengukur "kebingungan" (perplexity)
terhadap suatu kalimat. Kalimat yang lebih masuk akal = perplexity lebih rendah.

Ini adalah sinyal training yang sama yang dipakai saat pre-training berlangsung:
cross-entropy loss untuk memprediksi token berikutnya.

Jalankan: python 01_perplexity_demo.py
Requirement: pip install transformers torch
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_perplexity(model, tokenizer, text: str) -> float:
    """Hitung perplexity sebuah kalimat menggunakan model causal LM."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    perplexity = torch.exp(torch.tensor(loss)).item()
    return perplexity


def main():
    print("Loading model GPT-2...")
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print("\n" + "="*60)
    print("1. Loss dan perplexity dari satu kalimat")
    print("="*60)
    text = "Kucing itu duduk di atas tikar"
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Kalimat    : '{text}'")
    print(f"Loss       : {outputs.loss.item():.4f}")
    print(f"Perplexity : {torch.exp(outputs.loss).item():.2f}")
    print("\nPerplexity = e^loss. Makin rendah = model makin yakin dengan")
    print("urutan token tersebut = kalimat lebih 'masuk akal' menurut model.")

    print("\n" + "="*60)
    print("2. Bandingkan perplexity kalimat masuk akal vs tidak")
    print("="*60)
    kalimat = [
        ("Masuk akal", "The cat sat on the mat"),
        ("Masuk akal", "The dog ran through the park"),
        ("Tidak masuk akal", "The mat sat on the cat"),
        ("Sangat aneh", "Purple ideas sleep furiously yesterday"),
    ]
    for label, k in kalimat:
        ppl = get_perplexity(model, tok, k)
        bar = "█" * min(int(ppl / 20), 30)
        print(f"[{label:>16s}] PPL={ppl:7.1f}  {bar}")
        print(f"  '{k}'")

    print("\n" + "="*60)
    print("3. Ilustrasi training loop pre-training (pseudo-code)")
    print("="*60)
    print("""
    for batch in data_loader:          # triliun token, berbulan-bulan
        input_ids = batch['input_ids']
        labels    = input_ids          # target = input itu sendiri, digeser 1

        logits = model(input_ids)      # forward pass
        loss   = cross_entropy(logits, labels)  # prediksi vs aktual

        loss.backward()                # backward pass (gradien mengalir)
        optimizer.step()               # update semua weight matrix
        optimizer.zero_grad()

    # Itulah SELURUH pre-training. Sesederhana itu secara konsep,
    # tapi dijalankan pada skala yang sangat besar.
    """)


if __name__ == "__main__":
    main()
