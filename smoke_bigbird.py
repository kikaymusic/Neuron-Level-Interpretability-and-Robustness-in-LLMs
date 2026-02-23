import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForSequenceClassification

MODEL_HF = "google/bigbird-roberta-base"
NUM_LABELS = 5
weights_path = "./data/BigBird/best_model_BigBird.pth"

device = "cpu"

print("loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF, num_labels=NUM_LABELS)

print("loading weights...")
ckpt = torch.load(weights_path, map_location=device)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

print("OK loaded")
print("missing:", len(missing), "unexpected:", len(unexpected))
