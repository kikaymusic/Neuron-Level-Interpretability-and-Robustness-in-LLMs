# lumia_unimodal_tb_synapse.py
# Minimal-change fix: make the module import-safe by moving heavy imports
# (torch/tensorflow/datasets/tqdm) inside __main__. Keep Phase1/Phase2 helpers
# usable without importing heavy libs.

import argparse
import gc
import json
import math
import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lumia_utils import (
    load_models,
    get_activations_and_attention,
    build_classifier,
)

# ---------------------------------------------------------------------
# SYNAPSE integration helpers (minimal LUMIA operational API)
# These functions are intentionally simple and accept precomputed
# CLS activations from external pipelines (e.g., SYNAPSE).
# ---------------------------------------------------------------------


def _ensure_nlh(activations: np.ndarray) -> np.ndarray:
    """Normalize activations into shape [N, L, H]."""
    if activations.ndim == 3:
        return activations
    if activations.ndim == 4 and activations.shape[2] == 1:
        return activations[:, :, 0, :]
    if activations.ndim == 4 and activations.shape[-1] == 1:
        return activations[:, :, :, 0]
    raise ValueError(f"Unsupported activations shape: {activations.shape}")


def lumia_layerwise_auc_from_activations(
    activations: np.ndarray,
    membership_labels: np.ndarray,
    seed: int = 123,
    test_size: float = 0.2,
    out_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    LUMIA Phase 1 (layer-wise):
      - Train a linear membership probe per layer
      - Report AUC by layer
      - Select l_star = argmax(AUC)

    Inputs:
      activations: np.ndarray [N, L, H] (CLS per layer)
      membership_labels: np.ndarray [N] with 0/1

    Outputs:
      df_auc: DataFrame with columns [layer_id, auc]
      l_star: int
    """
    activations = _ensure_nlh(np.asarray(activations))
    y = np.asarray(membership_labels).astype(int)

    N, L, _H = activations.shape

    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    y_train = y[idx_train]
    y_test = y[idx_test]

    auc_rows = []
    for layer_id in range(L):
        X_train = activations[idx_train, layer_id, :]
        X_test = activations[idx_test, layer_id, :]

        clf = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        auc_rows.append({"layer_id": int(layer_id), "auc": float(auc)})

    df_auc = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    l_star = int(df_auc.iloc[0]["layer_id"])

    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_auc.to_csv(out_csv, index=False)

    return df_auc, l_star


def lumia_neuron_ranking_and_mask_from_layer(
    activations: np.ndarray,
    membership_labels: np.ndarray,
    layer_id: int,
    top_p: float = 0.01,
    seed: int = 123,
    test_size: float = 0.2,
    out_scores_csv: Optional[str] = None,
    out_mask_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LUMIA Phase 2 (neuron-wise on a single layer):
      - Train membership probe on a selected layer
      - Rank neurons by |w|
      - Build mask_M (global indices: layer_id*H + neuron_id)

    Inputs:
      activations: np.ndarray [N, L, H]
      membership_labels: np.ndarray [N] 0/1
      layer_id: int (l_star)
      top_p: fraction (e.g. 0.005, 0.01, 0.05)

    Outputs:
      dict with:
        layer_id, hidden_dim, top_p, top_k, indices_global
    """
    activations = _ensure_nlh(np.asarray(activations))
    y = np.asarray(membership_labels).astype(int)

    N, L, H = activations.shape
    if not (0 <= layer_id < L):
        raise ValueError(f"layer_id={layer_id} out of range [0, {L-1}]")

    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    X_train = activations[idx_train, layer_id, :]
    y_train = y[idx_train]

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    w = clf.coef_.reshape(-1)  # [H] for binary
    scores = np.abs(w)

    k = max(1, int(round(H * float(top_p))))
    rank_idx = np.argsort(scores)[::-1]
    top_local = rank_idx[:k].astype(int)

    indices_global = [int(layer_id * H + j) for j in top_local.tolist()]

    if out_scores_csv is not None:
        out_dir = os.path.dirname(out_scores_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_scores = pd.DataFrame(
            {
                "layer_id": [int(layer_id)] * H,
                "neuron_id": np.arange(H, dtype=int),
                "score": scores.astype(float),
            }
        ).sort_values("score", ascending=False)
        df_scores.to_csv(out_scores_csv, index=False)

    mask_obj = {
        "layer_id": int(layer_id),
        "hidden_dim": int(H),
        "top_p": float(top_p),
        "top_k": int(k),
        "indices_global": indices_global,
    }

    if out_mask_json is not None:
        out_dir = os.path.dirname(out_mask_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_mask_json, "w", encoding="utf-8") as f:
            json.dump(mask_obj, f, indent=2)

    return mask_obj


if __name__ == "__main__":
    # Heavy imports moved here to avoid import-time deadlocks.
    import tensorflow as tf
    import torch
    from datasets import load_dataset
    from tensorflow.keras.callbacks import EarlyStopping
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Mia with hidden states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", type=str, help="dataset name")
    parser.add_argument("-m", "--model", type=str, help="model size")
    parser.add_argument("-s", "--split", type=str, help="anagram split")
    parser.add_argument("-g", "--gpu", type=str, help="gpu")
    parser.add_argument("-r", "--round", type=str, help="round")
    args = parser.parse_args()
    config = vars(args)

    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda:0")
    print(f"Training model {args.model}")
    model_name = args.model
    model_size = 0
    start_layer = 0
    token = -1
    max_token_division = 2048
    epochs = 1000
    batch_size = 256
    dataset_name = args.dataset
    if "wikimia" in dataset_name:
        split = 32
    else:
        split = args.split
    model_name_file = model_name.replace("/", "")

    file_path = f"./results/{model_name}/{model_name_file}_{dataset_name}_ngram_{split}_{args.round}.csv"

    if not os.path.exists(file_path):
        model_name_dir = model_name.replace("/", "")
        directory = f"./processed_data/{model_name_dir}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        model, tokenizer = load_models(model_name, model_size)

        if "wikimia" in dataset_name:
            wikimia = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{split}")
            inputs = wikimia["input"]
            laberu = wikimia["label"]
            members = []
            notMembers = []
            dataset_name = "wikipediamia"
            for i in range(len(laberu)):
                if laberu[i] == 1:
                    members.append(inputs[i])
                else:
                    notMembers.append(inputs[i])
            list_members = members
            list_nonmembers = notMembers

        elif "gutenberg" in dataset_name:
            from random import sample

            pg19 = load_dataset("imperial-cpg/project-gutenberg-extended")
            list_files = os.listdir("./data/pg19/train")

            nonmember = pg19["train"]["text"]
            limitnonmember = []
            for nomem in nonmember:
                if len(nomem) > 1000:
                    limitnonmember.append(nomem[0:1000])
                else:
                    limitnonmember.append(nomem)
            member = sample(list_files, len(nonmember))

            def read_member(txt):
                base_path = "./data/pg19/train/"
                with open(f"{base_path}{txt}", "r") as f:
                    text = f.read()
                if len(text) > 1000:
                    return text[0:1000]
                return text

            member_texts = []
            for members in member:
                info = read_member(members)
                member_texts.append(info)
            list_members = member_texts[0:1000]
            list_nonmembers = limitnonmember[0:1000]

        elif "arxiv-1-month" in dataset_name:
            from datetime import datetime

            threshold_date = datetime(2023, 3, 1)

            list_files = os.listdir("./data/pajama/arxiv/")
            members = []
            nonmembers = []
            for file in list_files:
                with open(f"./data/pajama/arxiv/{file}", "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if datetime.fromisoformat(data["meta"]["timestamp"]) < threshold_date:
                                members.append(data["text"])
                            else:
                                nonmembers.append(data["text"])
                        except Exception:
                            continue

            import random

            list_members = random.sample(members, len(nonmembers))
            list_nonmembers = nonmembers

        elif "arxiv_mia_cs" in dataset_name:
            data_jsons = []
            data_members = []
            data_nomembers = []
            with open("./data/arxiv_mia/arxiv_mia.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "cs" in data["field"]:
                        data_jsons.append(data)
                        if data["label"]:
                            data_members.append(data["text"])
                        else:
                            data_nomembers.append(data["text"])
            list_members = data_members
            list_nonmembers = data_nomembers

        elif "arxiv_mia_math" in dataset_name:
            data_jsons = []
            data_members = []
            data_nomembers = []
            with open("./data/arxiv_mia/arxiv_mia.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "math" in data["field"]:
                        data_jsons.append(data)
                        if data["label"]:
                            data_members.append(data["text"])
                        else:
                            data_nomembers.append(data["text"])

            list_members = data_members
            list_nonmembers = data_nomembers

        if "pythia" in model_name:
            end_layer = len(model.gpt_neox.layers)
            n_layers = len(model.gpt_neox.layers)
        else:
            end_layer = len(model.transformer.h)
            n_layers = len(model.transformer.h)

        activations_member = []
        activations_nonmember = []

        for member in tqdm(list_members, desc="Processing list_members"):
            split_into = math.floor(len(member) / max_token_division)
            partial_activations = []
            split_into = 1

            if split_into == 0:
                split_into = 1
            for _i in range(split_into):
                hidden = get_activations_and_attention(
                    model,
                    tokenizer,
                    member,
                    start_layer,
                    end_layer,
                    token,
                    max_token_division,
                )
                partial_activations.append(hidden)

            stacked_tensors = torch.stack(partial_activations, dim=0)
            mean = torch.mean(stacked_tensors, dim=0)
            activations_member.append(mean)

        activations_member = np.array(activations_member)
        emb_dim = activations_member[0].shape[2]

        for nonmember in tqdm(list_nonmembers, desc="Processing len_nonmembers"):
            split_into = math.floor(len(nonmember) / max_token_division)
            partial_activations = []
            split_into = 1
            if split_into == 0:
                split_into = 1
            for _i in range(split_into):
                try:
                    hidden = get_activations_and_attention(
                        model,
                        tokenizer,
                        nonmember,
                        start_layer,
                        end_layer,
                        token,
                        max_token_division,
                    )
                    partial_activations.append(hidden)
                except Exception:
                    continue

            stacked_tensors = torch.stack(partial_activations, dim=0)
            mean = torch.mean(stacked_tensors, dim=0)
            activations_nonmember.append(mean)

        activations_nonmember = np.array(activations_nonmember)

        if len(activations_member) > len(activations_nonmember):
            activations_member = activations_member[0: len(activations_nonmember)]
        else:
            activations_nonmember = activations_nonmember[0: len(activations_member)]

        activations_member_copy = np.array(activations_member)
        activations_nonmember_copy = np.array(activations_nonmember)

        activations_combined = np.concatenate((activations_member_copy, activations_nonmember_copy))

        labels_member = [1] * len(activations_member)
        labels_nonmember = [0] * len(activations_nonmember)
        labels_combined = np.array(labels_member + labels_nonmember)

        X_train, X_test, y_train, y_test = train_test_split(
            activations_combined,
            labels_combined,
            test_size=0.2,
            stratify=labels_combined,
        )

        max_acc = 0
        max_auc = 0
        acc_arr = []
        acc_auc = []
        layer_arr = []

        del model
        gc.collect()
        torch.cuda.empty_cache()

        with tf.device("/gpu:1"):
            for layer in range(n_layers):
                print(f"Extracting for layer {layer}")
                X_train_layer = []
                X_test_layer = []

                for sample in X_train:
                    X_train_layer.append(np.array(sample[layer][0]))
                for sample in X_test:
                    X_test_layer.append(np.array(sample[layer][0]))

                X_train_layer = np.array(X_train_layer)
                X_test_layer = np.array(X_test_layer)

                y_train = np.array(y_train)
                y_test = np.array(y_test)

                classifier = build_classifier(emb_dim)

                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    patience=10,
                    restore_best_weights=True,
                )

                history = classifier.fit(
                    x=X_train_layer,
                    y=y_train,
                    validation_data=(X_test_layer, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                val_accuracies = history.history["val_accuracy"]
                best_val_accuracy = max(val_accuracies)

                max_acc = max(best_val_accuracy, max_acc)
                acc_arr.append(best_val_accuracy)
                layer_arr.append(layer)

                y_pred_probs = classifier.predict(X_test_layer)

                if len(y_test.shape) == 1 or y_test.shape[1] == 1:
                    if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] == 2:
                        y_pred_probs = y_pred_probs[:, 1]

                    auc_roc = roc_auc_score(y_test, y_pred_probs)
                    print(f"AUC-ROC Score: {auc_roc:.4f}")
                    max_auc = max(max_auc, auc_roc)
                    print(f"MAX AUC-ROC Score: {max_auc:.4f}")
                    acc_auc.append(auc_roc)
                else:
                    print("Multi-class AUC-ROC calculation is not shown here.")

        df = pd.DataFrame()
        df["layer"] = layer_arr
        df["acc"] = acc_arr
        df["auc"] = acc_auc

        model_name_out = model_name.replace("/", "")
        os.makedirs(f"./results/{model_name_out}", exist_ok=True)
        df.to_csv(
            f"./results/{model_name_out}/{model_name_out}_{dataset_name}_{split}_{args.round}.csv",
            index=False,
        )