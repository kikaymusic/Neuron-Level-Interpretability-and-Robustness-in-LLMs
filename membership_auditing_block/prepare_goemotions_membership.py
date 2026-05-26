import pandas as pd
import numpy as np
import json, os

GOEMOTIONS_PATH = "data/goemotions"
EMOTIONS_FILE   = f"{GOEMOTIONS_PATH}/emotions.txt"
TARGET_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
OUTPUT_FILE     = f"{GOEMOTIONS_PATH}/full_sample_all.json"

with open(EMOTIONS_FILE) as f:
    id2emotion = [line.strip() for line in f.readlines()]
emotion2id = {e: i for i, e in enumerate(id2emotion)}
target_ids = [emotion2id[e] for e in TARGET_EMOTIONS]
goemo2local = {eid: i for i, eid in enumerate(target_ids)}

def load_tsv(path, is_member):
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "split"])
    df = df.dropna(subset=["labels"])
    df["label_ids"] = df["labels"].apply(lambda x: list(map(int, str(x).split(","))))
    df = df[df["label_ids"].apply(lambda ids: len(ids)==1 and ids[0] in target_ids)].copy()
    df["label_id"]  = df["label_ids"].apply(lambda ids: goemo2local[ids[0]])
    df["is_member"] = is_member
    return df[["text", "label_id", "is_member"]]

df_train = load_tsv(f"{GOEMOTIONS_PATH}/train.tsv", is_member=1)
df_test  = load_tsv(f"{GOEMOTIONS_PATH}/test.tsv",  is_member=0)

print("Members (train.tsv) por emocion:")
for i, e in enumerate(TARGET_EMOTIONS):
    print(f"  {e}: {(df_train['label_id']==i).sum()}")

print("\nNon-members (test.tsv) por emocion:")
for i, e in enumerate(TARGET_EMOTIONS):
    print(f"  {e}: {(df_test['label_id']==i).sum()}")

# Balancear para que el ratio sea aproximadamente 80/20
# Usar todos los de test como non-members y ajustar members
n_non = len(df_test)
n_mem_target = n_non * 4  # ratio 80/20

if len(df_train) > n_mem_target:
    df_train = df_train.sample(n=n_mem_target, random_state=42)
    print(f"\nMembers recortados a {n_mem_target} para mantener ratio 80/20")

df_all = pd.concat([df_train, df_test]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset final:")
print(f"  Total:       {len(df_all)}")
print(f"  Members:     {(df_all['is_member']==1).sum()}")
print(f"  Non-members: {(df_all['is_member']==0).sum()}")
print(f"  Ratio:       {(df_all['is_member']==1).mean():.2f} / {(df_all['is_member']==0).mean():.2f}")

df_all.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
print(f"\nGuardado en {OUTPUT_FILE}")
