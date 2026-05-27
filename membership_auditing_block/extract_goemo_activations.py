"""
extract_goemo_activations.py
Extracts [CLS] LayerNorm activations for all 2770 GoEmotions examples in
full_sample_all.json and saves in the same NDJSON format as activations.json.

Output:
  activations_full.json   - NDJSON, same format as existing activations.json
  membership_full.pt      - torch tensor of is_member flags (2770,)
  sample_df_full.json     - NDJSON with input_ids and attention_mask per example
"""

import json, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE    = "/Users/kikay/Documents/investigacion/Neuron-Level-Interpretability-and-Robustness-in-LLMs/data/goemotions"
MODEL   = "monologg/bert-base-cased-goemotions-original"
MAX_LEN = 128
L       = 12

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load data
with open(f"{BASE}/full_sample_all.json") as f:
    rows = [json.loads(l) for l in f if l.strip()]
print(f"Examples: {len(rows)}  Members: {sum(r['is_member'] for r in rows)}  "
      f"Non-members: {sum(1-r['is_member'] for r in rows)}")

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval().to(device)

# Hook: capture [CLS] from each LayerNorm output
layer_acts = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            layer_acts[layer_idx] = output[0, 0, :].detach().cpu().float().numpy()
    return hook

handles = []
for i in range(L):
    h = model.bert.encoder.layer[i].output.LayerNorm.register_forward_hook(make_hook(i))
    handles.append(h)

# Extract activations
print("Extracting activations...")
act_lines  = []
sdf_lines  = []
y_mem      = []

with torch.no_grad():
    for idx, row in enumerate(rows):
        enc = tokenizer(row["text"], max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        input_ids   = enc["input_ids"].to(device)
        attn_mask   = enc["attention_mask"].to(device)

        layer_acts.clear()
        _ = model(input_ids=input_ids, attention_mask=attn_mask)

        # Build activations record (same format as existing activations.json)
        layers_data = [
            {"index": li, "values": layer_acts[li].tolist()}
            for li in range(L)
        ]
        act_rec = {
            "linex_index": idx,
            "features": [{"token": "[CLS]", "layers": layers_data}]
        }
        act_lines.append(json.dumps(act_rec))

        # sample_df record (input_ids + attention_mask for task utility eval)
        sdf_lines.append(json.dumps({
            "input_ids":      input_ids[0].cpu().tolist(),
            "attention_mask": attn_mask[0].cpu().tolist(),
            "label":          row["label_id"],
        }))

        y_mem.append(row["is_member"])

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(rows)}")

for h in handles:
    h.remove()

# Save
out_acts = f"{BASE}/activations_full.json"
with open(out_acts, "w") as f:
    f.write("\n".join(act_lines) + "\n")
print(f"Saved: {out_acts}")

out_mem = f"{BASE}/membership_full.pt"
torch.save(torch.tensor(y_mem, dtype=torch.long), out_mem)
print(f"Saved: {out_mem}")

out_sdf = f"{BASE}/sample_df_full.json"
with open(out_sdf, "w") as f:
    f.write("\n".join(sdf_lines) + "\n")
print(f"Saved: {out_sdf}")

print(f"\nDone. {len(rows)} examples extracted.")
print(f"Members: {sum(y_mem)}  Non-members: {len(y_mem)-sum(y_mem)}")
