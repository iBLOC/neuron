#!/usr/bin/env python
"""Run MME / POPE / MMStar subsets against a neuron Mage-VL server.

Scratch tooling under scripts/vlm-bench/. Sends each JSONL row as an
OpenAI chat completion with a data-URI image, parses yes/no or A-D,
scores per benchmark convention, writes a markdown report + raw answers.

Usage:
    python eval_vlm.py --base-url http://127.0.0.1:8083 --bench all \
        --data-dir .\\data --out .\\results [--limit N]
"""
import argparse, base64, json, re, sys, time
from collections import defaultdict
from pathlib import Path
import requests

YES_RE = re.compile(r"^\W*(yes|yeah|y|是)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\W*(no|n|不是|否)\b", re.IGNORECASE)
LETTER_RE = re.compile(r"(?<!\w)([ABCD])(?!\w)", re.IGNORECASE)


def parse_yesno(text: str) -> str:
    """First yes/no token wins; anything else counts as wrong."""
    t = text.strip()
    m_yes, m_no = YES_RE.search(t), NO_RE.search(t)
    if m_yes and (not m_no or m_yes.start() <= m_no.start()):
        return "yes"
    if m_no:
        return "no"
    return "unparsed"


def parse_letter(text: str):
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else None


def send(client, url, model, row, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + row["image_b64"]}},
                {"type": "text", "text": row["question"]},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = client.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    try:
        text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        text = ""
    return text, dt


def score_yesno(rows, answers):
    """Official-style MME/POPE binary scoring per category.

    MME:  score = (acc+ + acc-) / 2 * 100
    POPE: precision / recall / F1 / accuracy / yes-rate
    """
    per_cat = defaultdict(list)
    for row, ans in zip(rows, answers):
        per_cat[row["category"]].append((row["gt"], ans))
    out = {}
    for cat, pairs in sorted(per_cat.items()):
        yes = [(g, a) for g, a in pairs if g == "yes"]
        no = [(g, a) for g, a in pairs if g == "no"]
        acc_yes = sum(a == "yes" for _, a in yes) / len(yes) if yes else None
        acc_no = sum(a == "no" for _, a in no) / len(no) if no else None
        tp = sum(a == "yes" and g == "yes" for g, a in pairs)
        fp = sum(a == "yes" and g == "no" for g, a in pairs)
        fn = sum(a == "no" and g == "yes" for g, a in pairs)
        tn = sum(a == "no" and g == "no" for g, a in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(pairs) if pairs else 0.0
        out[cat] = {
            "n": len(pairs), "yes_rate": (tp + fp) / len(pairs),
            "precision": prec, "recall": rec, "f1": f1, "acc": acc,
            "acc_yes": acc_yes, "acc_no": acc_no,
            "mme_score": (None if acc_yes is None or acc_no is None
                          else round((acc_yes + acc_no) / 2 * 100, 1)),
        }
    return out


def score_mc(rows, answers):
    per_cat = defaultdict(lambda: [0, 0])
    for row, ans in zip(rows, answers):
        c = per_cat[row["category"]]
        c[1] += 1
        if ans is not None and ans == row["gt"]:
            c[0] += 1
    return {cat: {"n": tot, "acc": ok / tot, "correct": ok}
            for cat, (ok, tot) in sorted(per_cat.items())}


def render_report(bench, rows, answers, times, scores, model):
    unparsed = sum(
        1 for row, a in zip(rows, answers)
        if (row["gt"] in ("yes", "no") and a == "unparsed")
        or (row["gt"] not in ("yes", "no") and a is None))
    total_dt = sum(times)
    lines = [f"## {bench.upper()} — {model}", ""]
    if bench == "mmstar":
        overall_ok = sum(s.get("correct", 0) for s in scores.values())
        overall_n = sum(s["n"] for s in scores.values())
        lines += [f"overall: **{overall_ok}/{overall_n} = {overall_ok/overall_n:.1%}**"
                  f" (unparsed answers: {unparsed})", "",
                  "| category | n | correct | acc |", "|---|---|---|---|"]
        for cat, s in scores.items():
            lines.append(f"| {cat} | {s['n']} | {s.get('correct',0)} | {s['acc']:.1%} |")
    else:
        w = "mme_score" if bench == "mme" else "f1"
        lines += ["| category | n | " + ("MME score" if bench == "mme" else "F1")
                  + " | acc | yes-rate | acc+(yes) | acc-(no) |",
                  "|---|---|---|---|---|---|---|"]
        for cat, s in scores.items():
            yes_s = f"{s['acc_yes']:.1%}" if s["acc_yes"] is not None else "-"
            no_s = f"{s['acc_no']:.1%}" if s["acc_no"] is not None else "-"
            main = f"{s[w]}" if isinstance(s[w], float) and w == "f1" else (
                f"{s[w]:.1f}" if w == "mme_score" and s[w] is not None else
                (f"{s[w]:.1%}" if isinstance(s[w], float) else "-"))
            lines.append(f"| {cat} | {s['n']} | {main} | {s['acc']:.1%} "
                         f"| {s['yes_rate']:.1%} | {yes_s} | {no_s} |")
        if bench == "mme":
            vals = [s["mme_score"] for s in scores.values() if s["mme_score"] is not None]
            lines += ["", f"**MME overall: {sum(vals):.1f}** "
                      f"(perception sum over sampled subtasks)"]
        else:
            allp = [(r["gt"], a) for r, a in zip(rows, answers)]
            tp = sum(a == "yes" and g == "yes" for g, a in allp)
            fp = sum(a == "yes" and g == "no" for g, a in allp)
            fn = sum(a == "no" and g == "yes" for g, a in allp)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            acc = (tp + sum(a == "no" and g == "no" for g, a in allp)) / len(allp)
            lines += ["", f"**POPE overall: F1 {f1:.3f}, acc {acc:.1%}, "
                      f"precision {prec:.3f}, recall {rec:.3f}** "
                      f"(yes-rate {(tp+fp)/len(allp):.1%}, unparse {unparsed})"]
    lines += ["", f"latency: mean {total_dt/len(times):.1f}s/question, "
              f"total {total_dt/60:.1f} min over {len(times)} questions"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8083")
    ap.add_argument("--model", default="microsoft/Mage-VL")
    ap.add_argument("--bench", default="all",
                    choices=["mme", "pope", "mmstar", "all"])
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--limit", type=int, default=0, help="cap rows per bench (0=all)")
    ap.add_argument("--max-tokens-yn", type=int, default=8)
    ap.add_argument("--max-tokens-mc", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()

    data_dir, out_dir = Path(a.data_dir), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = requests.Session()
    url = a.base_url.rstrip("/") + "/v1/chat/completions"
    benches = ["mme", "pope", "mmstar"] if a.bench == "all" else [a.bench]

    for bench in benches:
        path = data_dir / f"{bench}.jsonl"
        if not path.exists():
            print(f"skip {bench}: {path} missing", file=sys.stderr)
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.limit:
            rows = rows[: a.limit]
        max_tokens = a.max_tokens_mc if bench == "mmstar" else a.max_tokens_yn
        print(f"[{bench}] {len(rows)} questions ...", file=sys.stderr)
        answers, times = [], []
        raw_f = (out_dir / f"{bench}-answers.jsonl").open("w", encoding="utf-8")
        for i, row in enumerate(rows):
            try:
                text, dt = send(client, url, a.model, row, max_tokens, a.timeout)
            except Exception as e:
                text, dt = f"__ERROR__: {e}", 0.0
            answers.append(parse_letter(text) if bench == "mmstar" else parse_yesno(text))
            times.append(dt)
            raw_f.write(json.dumps({
                "id": row["id"], "category": row["category"], "gt": row["gt"],
                "response": text[:500], "parsed": answers[-1], "sec": round(dt, 2),
            }, ensure_ascii=False) + "\n")
            raw_f.flush()
            if (i + 1) % 10 == 0:
                done_dt = sum(times)
                print(f"  {i+1}/{len(rows)}  {done_dt/60:.1f} min elapsed",
                      file=sys.stderr)
        raw_f.close()
        if bench == "mmstar":
            scores = score_mc(rows, answers)
        else:
            scores = score_yesno(rows, answers)
        report = render_report(bench, rows, answers, times, scores, a.model)
        (out_dir / f"{bench}-report.md").write_text(report, encoding="utf-8")
        print(report, file=sys.stderr)


if __name__ == "__main__":
    main()
