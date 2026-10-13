#!/usr/bin/env python
"""Convert MME / POPE / MMStar subsets into JSONL on stdout for the neuron
VLM eval. The caller redirects stdout to a data file; no file APIs in
this script.

Usage:
    python prep_datasets.py --bench pope > data/pope.jsonl
"""
import argparse, base64, io, json, os, secrets, sys
from datasets import load_dataset
from PIL import Image


def log(msg):
    print(msg, file=sys.stderr)


def crypto_shuffle(idxs):
    return sorted(idxs, key=lambda _: secrets.randbits(64))


def crypto_sample(idxs, k: int):
    return crypto_shuffle(idxs)[:k]


def img_to_b64(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def emit(rows):
    for r in rows:
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


def gen_mme(per_category: int):
    log("loading lmms-lab/MME ...")
    ds = load_dataset("lmms-lab/MME", split="test")
    by_cat = {}
    for i, r in enumerate(ds):
        by_cat.setdefault(r["category"], []).append(i)
    rows = []
    for cat in sorted(by_cat):
        for i in crypto_sample(by_cat[cat], min(per_category, len(by_cat[cat]))):
            r = ds[i]
            q = r["question"].strip()
            if "yes or no" not in q.lower():
                q += " Please answer yes or no."
            rows.append({
                "id": f"mme-{r['question_id']}",
                "bench": "mme",
                "image_b64": img_to_b64(r["image"]),
                "question": q,
                "gt": r["answer"].strip().lower(),
                "category": cat,
            })
    return rows


def gen_pope(per_setting: int):
    log("loading lmms-lab/POPE ...")
    ds = load_dataset("lmms-lab/POPE", split="test")
    by_cat = {}
    for i, r in enumerate(ds):
        by_cat.setdefault(r["category"], []).append(i)
    rows = []
    for cat in sorted(by_cat):
        for i in crypto_sample(by_cat[cat], min(per_setting, len(by_cat[cat]))):
            r = ds[i]
            q = r["question"].strip()
            if "yes or no" not in q.lower():
                q += " Please answer yes or no."
            rows.append({
                "id": f"pope-{r['question_id']}-{cat}",
                "bench": "pope",
                "image_b64": img_to_b64(r["image"]),
                "question": q,
                "gt": r["answer"].strip().lower(),
                "category": cat,
            })
    return rows


def gen_mmstar(n: int):
    log("loading Lin-Chen/MMStar ...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    by_cat = {}
    for i, r in enumerate(ds):
        by_cat.setdefault(r["l2_category"], []).append(i)
    per_cat = max(1, n // len(by_cat))
    rows = []
    for cat in sorted(by_cat, key=str):
        for i in crypto_sample(by_cat[cat], min(per_cat, len(by_cat[cat]))):
            r = ds[i]
            q = r["question"].strip()
            if "option" not in q.lower() or "letter" not in q.lower():
                q += "\nAnswer with the option's letter from the given choices directly."
            rows.append({
                "id": f"mmstar-{r['index']}",
                "bench": "mmstar",
                "image_b64": img_to_b64(r["image"]),
                "question": q,
                "gt": str(r["answer"]).strip().upper()[:1],
                "category": str(cat),
            })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["mme", "pope", "mmstar"])
    ap.add_argument("--mme-per-category", type=int, default=20)
    ap.add_argument("--pope-per-setting", type=int, default=150)
    ap.add_argument("--mmstar", type=int, default=160)
    a = ap.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if a.bench == "mme":
        emit(gen_mme(a.mme_per_category))
    elif a.bench == "pope":
        emit(gen_pope(a.pope_per_setting))
    else:
        emit(gen_mmstar(a.mmstar))
