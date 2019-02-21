import pandas as pd
import numpy as np
import os
import torch
import csv
import re
import heapq
import logging

from copy import copy
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from agrr.tokenization import BertTokenizer


ID_TO_TAG = ("[NONE]", "cV", "cR1", "cR2", "R1", "R2")
TAG_TO_ID = {v: k for k, v in enumerate(ID_TO_TAG)}

ID_TO_MISSED = ("[NONE]", "V")
MISSED_TO_ID = {v: k for k, v in enumerate(ID_TO_MISSED)}

G = {
    0: {'label': '_start', 'to': [1, 2, 12], 'from': []},
    1: {'label': '_end', 'to': [], 'from': [0, 7, 9]},
    2: {'label': 'cR1', 'to': [3, 10], 'from': [0]},
    3: {'label': 'cV', 'to': [4, 9], 'from': [2]},
    4: {'label': 'cR2', 'to': [5], 'from': [3]},
    5: {'label': 'R1', 'to': [7], 'from': [4, 7, 11, 14]},
    7: {'label': 'R2', 'to': [1, 5], 'from': [5]},
    9: {'label': 'R1', 'to': [1], 'from': [3, 13]},
    10: {'label': 'cR2', 'to': [11], 'from': [2]},
    11: {'label': 'cV', 'to': [5], 'from': [10]},
    12: {'label': 'cV', 'to': [13], 'from': [0]},
    13: {'label': 'cR1', 'to': [9, 14], 'from': [12]},
    14: {'label': 'cR2', 'to': [5], 'from': [13]}
}


def load_csv(data_dir, filename):
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE).fillna("")
    return df.to_dict("records")


def load_graph(data_dir, filename):
    path = os.path.join(data_dir, filename)
    vertices, edges = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            vm = re.match(r"(\d+) \[label=\"(.+)\"\]", line)    
            em = re.match(r"(\d+) -> (\d+)", line)
            if vm:
                vertices.append((int(vm.group(1)), vm.group(2)))
            elif em:
                edges.append((int(em.group(1)), int(em.group(2))))
    assert len(vertices) > 0
    assert len(edges) > 0
    graph = {}
    for v, label in vertices:
        graph[v] = {"label": label, "to": [], "from": []}
    for u, v in edges:
        graph[u]["to"].append(v)
        graph[v]["from"].append(u)
    return graph


def to_examples(records, tokenizer, max_seq_length):
    examples = []
    errors = 0
    for r in records:
        try:
            text = r["text"]

            text = re.subn(
                r"(\d+),(\d+)",
                lambda m: f"{m.groups()[0]}.{m.groups()[1]}",
                text
            )[0]
            text = text.replace("—", "-")

            tokens, idxs = tokenizer.tokenize(text)

            tags_orig = []
            for tag in ID_TO_TAG[1:]:
                if not r[tag]:
                    continue
                for be in r[tag].split(" "):
                    b, e = list(map(int, be.split(":")))
                    tags_orig.append((tag, b, e))

            tags = []
            for token, (token_b, token_e) in zip(tokens, idxs):
                tagged = False
                for tag, tag_b, tag_e in tags_orig:
                    if tag_b <= token_b and token_e <= tag_e:
                        tags.append(tag)
                        tagged = True
                if not tagged:
                    tags.append("[NONE]")

            missed_orig = []
            if r["V"]:
                for be in r["V"].split(" "):
                    b, e = list(map(int, be.split(":")))
                    assert b == e
                    missed_orig.append(b)

            missed = ["[NONE]"] * len(tokens)
            for tag_b in missed_orig:
                finded = False
                for i, (token_b, token_e) in enumerate(idxs):
                    if token_b == tag_b and not finded:
                        missed[i] = "V"
                        finded = True
                assert finded, (r, tag_b)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
                tags = tags[:(max_seq_length - 2)]
                missed = missed[:(max_seq_length - 2)]
                idxs = idxs[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tags = ["[NONE]"] + tags + ["[NONE]"]
            missed = ["[NONE]"] + missed + ["[NONE]"]
            idxs = [(-1, -1)] + idxs + [(-1, -1)]

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tag_ids = [TAG_TO_ID[tag] for tag in tags]
            missed_ids = [MISSED_TO_ID[tag] for tag in missed]

            mask = [1] * len(token_ids)

            padding = [0] * (max_seq_length - len(token_ids))
            token_ids += padding
            mask += padding
            tag_ids += padding
            missed_ids += padding
            idxs += [(-1, -1)] * len(padding)

            assert len(token_ids) == max_seq_length
            assert len(mask) == max_seq_length
            assert len(tag_ids) == max_seq_length
            assert len(missed_ids) == max_seq_length
            assert len(idxs) == max_seq_length

            examples.append({
                "token_ids": token_ids,
                "mask": mask,
                "tag_ids": tag_ids,
                "missed_ids": missed_ids,
                "label": int(r["class"]) if r["class"] != "" else 2,
                "idxs": idxs
            })
        except:
            examples.append({
                "token_ids": [0] * max_seq_length,
                "mask": [0] * max_seq_length,
                "tag_ids": [0] * max_seq_length,
                "missed_ids": [0] * max_seq_length,
                "label": 0,
                "idxs": [(-1, -1)] * max_seq_length
            })
            errors += 1
    logging.getLogger("console").info(f"Parsing errors: {100 * errors / len(examples)}%")
    return examples


def to_tensor_data(examples):
    token_ids = torch.tensor([f["token_ids"] for f in examples], dtype=torch.long)
    masks = torch.tensor([f["mask"] for f in examples], dtype=torch.long)
    tag_ids = torch.tensor([f["tag_ids"] for f in examples], dtype=torch.long)
    missed_ids = torch.tensor([f["missed_ids"] for f in examples], dtype=torch.long)
    label_ids = torch.tensor([f["label"] for f in examples], dtype=torch.long)
    idxs = torch.tensor([f["idxs"] for f in examples], dtype=torch.long)
    return token_ids, masks, tag_ids, missed_ids, label_ids, idxs


def to_result(label_logits, missed_logits, annot_logits, idxs):
    i = 1
    while idxs[i][0] != -1:
        i += 1
    annot_logits = annot_logits[1:i]
    idxs = idxs[1:i]
    
    if len(idxs) == 0:
        return {
            "text": "", "class": "",
            "V": "", "R1": "", "R2": "", "cV": "", "cR1": "", "cR2": ""
        }

    idx_to_annot_logits = {}
    for a, i in zip(annot_logits, idxs):
        i = tuple(i)
        if i not in idx_to_annot_logits:
            idx_to_annot_logits[i] = (0.0,) * 6
        idx_to_annot_logits[i] += a

    idxs = sorted(idx_to_annot_logits.keys())
    PQ = {(0, True, -1): (0.0, 0, True, -1)}
    frm = {(0, True, -1): (None, None, None)}
    while True:
        logit, u, none, i = max(PQ.values())
        if G[u]["label"] == "_end":
            break
        del PQ[u, none, i]

        if i != len(idxs) - 1:
            delta = idx_to_annot_logits[idxs[i + 1]][TAG_TO_ID["[NONE]"]]
            if (u, True, i + 1) not in PQ:
                PQ[(u, True, i + 1)] = (-np.inf, u, True, i + 1)
            if PQ[(u, True, i + 1)][0] < logit + delta:
                PQ[(u, True, i + 1)] = logit + delta, u, True, i + 1
                frm[(u, True, i + 1)] = (u, none, i)

        if not none and i != len(idxs) - 1:
            delta = idx_to_annot_logits[idxs[i + 1]][TAG_TO_ID[G[u]["label"]]]
            if (u, False, i + 1) not in PQ:
                PQ[(u, False, i + 1)] = (-np.inf, u, False, i + 1)
            if PQ[(u, False, i + 1)][0] < logit + delta:
                PQ[(u, False, i + 1)] = logit + delta, u, False, i + 1
                frm[(u, False, i + 1)] = (u, none, i)

        for v in G[u]["to"]:
            if G[v]["label"] != "_end" and i != len(idxs) - 1:
                delta = idx_to_annot_logits[idxs[i + 1]][TAG_TO_ID[G[v]["label"]]]
                if (v, False, i + 1) not in PQ:
                    PQ[(v, False, i + 1)] = (-np.inf, v, False, i + 1)
                if PQ[(v, False, i + 1)][0] < logit + delta:
                    PQ[(v, False, i + 1)] = logit + delta, v, False, i + 1
                    frm[(v, False, i + 1)] = (u, none, i)
            elif G[v]["label"] == "_end" and i == len(idxs) - 1:
                PQ[(v, none, i + 1)] = (logit, v, none, i + 1)
                frm[(v, none, i + 1)] = (u, none, i)

    idx_to_annot = {}
    u, none, i = frm[(u, none, i)]
    while u is not None:
        idx_to_annot[idxs[i]] = TAG_TO_ID["[NONE]"] if none else TAG_TO_ID[G[u]["label"]]
        u, none, i = frm[(u, none, i)]

    result = {
        "text": "", "class": str(label_logits.argmax()),
        "V": [], "R1": [], "R2": [], "cV": [], "cR1": [], "cR2": []
    }

    tag = "[NONE]"
    s, e = -1, -1
    idxs = sorted(idx_to_annot.keys())
    for i in range(len(idxs)):
        idx = idxs[i]
        a = ID_TO_TAG[idx_to_annot[idx]]
        if tag == "[NONE]":
            tag = a
            s, e = idx
        elif tag == a:
            e = idx[1]
        else:
            result[tag].append(f"{s}:{e}")
            tag = a
            s, e = idx

    if len(result["R2"]) > 0:
        for be in result["R2"]:
            b = int(be.split(":")[0])
            result["V"].append(f"{b}:{b}")
    elif len(result["R1"]) > 0:
        for be in result["R1"]:
            b = int(be.split(":")[0])
            result["V"].append(f"{b}:{b}")

    for k in ["V", "R1", "R2", "cV", "cR1", "cR2"]:
        result[k] = " ".join(result[k])

    if result["V"] == "":
        result["class"] = "0"

    return result
