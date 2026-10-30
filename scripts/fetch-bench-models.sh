#!/bin/bash
# Fetch the model weights the per-model benchmark (`benches/model_bench.rs`)
# needs, into the layout the bench searches:
#
#   $NEURON_MODEL_DIRS/<capability>/<model_id>/     (llm, asr, rerank, embedding, …)
#   $NEURON_MODEL_DIRS/<engine>/<model_id>/         (burn, candle, …)
#
# Default root: ~/.neuron/models
#
# Usage:
#   ./scripts/fetch-bench-models.sh [group ...]
#
# Groups: llm asr rerank embedding ocr chronos2 decision all (default: all)
#
# Requires `hf` / `huggingface-cli` (pip install -U huggingface_hub).
# ModelScope mirrors work too — swap the `hf download` calls for
# `modelscope download --model <repo> --local_dir <dir>` when the HF CDN
# is slow from this host.

set -uo pipefail

ROOT="${NEURON_MODEL_DIRS:-$HOME/.neuron/models}"
HF="${HF_BIN:-hf}"
# The hf CLI's xet backend stalls on large files from some hosts; the plain
# HTTPS path (with resume) is the reliable fallback.
# huggingface.co is the default; hf-mirror is the fallback for hosts
# where the HF CDN is unreachable (it is usually SLOWER, not faster).
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DL="${NEURON_BENCH_DL:-curl}"

SELECTED=("$@")
if [ ${#SELECTED[@]} -eq 0 ]; then
  SELECTED=(llm asr rerank embedding ocr chronos2 decision)
fi

want() {
  local g="$1" sel
  for sel in "${SELECTED[@]}"; do
    if [ "$sel" = "$g" ] || [ "$sel" = "all" ]; then
      return 0
    fi
  done
  return 1
}

have() { [ -d "$1" ]; }

# List the files of a repo as "path<TAB>size" through the HF API.
#
# NOTE: python on Windows emits CRLF, so the \r has to be stripped *after*
# python (not before) or bash's `read` keeps it and the URL comes out
# malformed.
repo_files() { # repo_files <repo>
  "$DL" -sSL "$ENDPOINT/api/models/$1/tree/main?recursive=true" 2>/dev/null \
    | python -c "
import json,sys
try:
    for f in json.load(sys.stdin):
        if f.get('type') == 'file':
            sys.stdout.write(f['path'] + chr(9) + str(f.get('size') or 0) + chr(10))
except Exception:
    pass
" \
    | tr -d '\r'
}

# fetch <repo> <dest-dir> [--include GLOB ...] [--exclude GLOB ...]
#
# Without --include every file is fetched; --include switches to
# allow-list mode. --exclude always wins.
fetch() {
  local repo="$1" dest="$2"; shift 2
  local include=() exclude=()
  local mode=exclude
  while [ $# -gt 0 ]; do
    case "$1" in
      --include) mode=include; shift ;;
      --exclude) shift ;;
      *)
        if [ "$mode" = "include" ]; then include+=("$1"); else exclude+=("$1"); fi
        shift
        ;;
    esac
  done
  echo "[fetch] $repo -> $dest"
  mkdir -p "$dest"
  while IFS=$'\t' read -r path size; do
    [ -n "$path" ] || continue
    if [ ${#include[@]} -gt 0 ]; then
      local keep=0
      for pat in "${include[@]}"; do
        case "$path" in $pat) keep=1 ;; esac
      done
      [ "$keep" -eq 1 ] || continue
    fi
    local drop=0
    for pat in "${exclude[@]}"; do
      case "$path" in $pat) drop=1 ;; esac
    done
    [ "$drop" -eq 0 ] || continue
    local out="$dest/$path"
    mkdir -p "$(dirname "$out")"
    # Skip only when the local file is already the expected size — a
    # truncated download must be resumed, not accepted.
    if [ -f "$out" ]; then
      local have_size
      have_size=$(wc -c < "$out" 2>/dev/null | tr -d ' ')
      if [ "$have_size" = "$size" ]; then
        continue
      fi
      echo "[fetch]   $path — resuming ($have_size of $size bytes)"
    else
      echo "[fetch]   $path ($size bytes)"
    fi
    "$DL" -sSL --fail --retry 5 --retry-delay 2 -C - -o "$out" \
      "$ENDPOINT/$repo/resolve/main/$path" || echo "[fetch]   FAILED $path"
  done < <(repo_files "$repo")
}

# ---- LLM (candle) ----------------------------------------------------------
# MiniCPM5-1B / 2B are LlamaForCausalLM checkpoints — the candle arch
# detector routes them to the Llama path.
if want llm; then
  fetch openbmb/MiniCPM5-1B "$ROOT/llm/minicpm5-1b" \
    --exclude "*.gguf" "original/*" "*.pth"
  fetch openbmb/MiniCPM5-2B "$ROOT/llm/minicpm5-2b" \
    --exclude "*.gguf" "original/*" "*.pth"
fi

# ---- ASR --------------------------------------------------------------------
# Whisper: the candle runtime also wants mel_filters.safetensors, which
# lives in the lmz/candle-whisper space, not in the model repos.
if want asr; then
  for size in tiny small; do
    fetch "openai/whisper-$size" "$ROOT/asr/whisper-$size" \
      --include "config.json" "tokenizer.json" "preprocessor_config.json" \
                "model.safetensors" "generation_config.json" "vocab.json" "merges.txt"
  done
  if [ ! -f "$ROOT/asr/whisper-tiny/mel_filters.safetensors" ]; then
    echo "[fetch] mel_filters.safetensors (lmz/candle-whisper space)"
    "$DL" -sSL --fail -o "$ROOT/asr/whisper-tiny/mel_filters.safetensors" \
      "$ENDPOINT/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors" \
      || echo "[fetch]   mel_filters download failed"
    for size in small; do
      cp "$ROOT/asr/whisper-tiny/mel_filters.safetensors" \
         "$ROOT/asr/whisper-$size/mel_filters.safetensors" 2>/dev/null || true
    done
  fi
  # SenseVoiceSmall in the sherpa-onnx packaging (model.int8.onnx + tokens.txt)
  # plus the two test WAVs the ASR benchmark uses.
  fetch csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09 \
    "$ROOT/asr/sensevoice-small" \
    --include "model.int8.onnx" "tokens.txt" "test_wavs/zh.wav" "test_wavs/en.wav"
  for wav in zh en; do
    if [ -f "$ROOT/asr/sensevoice-small/test_wavs/$wav.wav" ]; then
      mv -f "$ROOT/asr/sensevoice-small/test_wavs/$wav.wav" \
            "$ROOT/asr/sensevoice-small/$wav.wav" 2>/dev/null || true
    fi
  done
fi

# ---- Rerank ----------------------------------------------------------------
if want rerank; then
  fetch BAAI/bge-reranker-base "$ROOT/rerank/bge-reranker-base" \
    --include "config.json" "tokenizer.json" "tokenizer_config.json" \
              "special_tokens_map.json" "vocab.txt" "model.safetensors" \
              "sentencepiece.bpe.model"
  fetch mixedbread-ai/mxbai-rerank-base-v2 "$ROOT/rerank/mxbai-rerank-base-v2" \
    --include "config.json" "tokenizer.json" "tokenizer_config.json" \
              "special_tokens_map.json" "vocab.txt" "model.safetensors" \
              "sentencepiece.bpe.model"
  fetch jinaai/jina-reranker-v2-base-multilingual "$ROOT/rerank/jina-reranker-base-v2" \
    --include "config.json" "tokenizer.json" "tokenizer_config.json" \
              "special_tokens_map.json" "model.safetensors"
fi

# ---- Embedding -------------------------------------------------------------
if want embedding; then
  for spec in \
    "sentence-transformers/all-MiniLM-L6-v2:all-minilm-l6-v2" \
    "BAAI/bge-base-en-v1.5:bge-base-en-v1.5" \
    "intfloat/e5-base-v2:e5-base-v2" \
    "thenlper/gte-base:gte-base"; do
    repo="${spec%%:*}"; id="${spec##*:}"
    fetch "$repo" "$ROOT/embedding/$id" \
      --include "config.json" "tokenizer.json" "tokenizer_config.json" \
                "special_tokens_map.json" "vocab.txt" "model.safetensors" \
                "sentencepiece.bpe.model" "1_Pooling/config.json"
  done
fi

# ---- OCR -------------------------------------------------------------------
# The OCR runtime wants ONE directory holding det.prepped.onnx +
# rec.prepped.onnx + dict.txt. The upstream exports live in two repos
# (`inference.onnx` each) and still need the metadata rewrite from
# scripts/prep-ppocr-onnx.py. The CTC vocabulary (18 383 entries) is the
# `PostProcess.character_dict` list inside the recognition repo's
# inference.yml — one token per line, exactly the dict.txt shape.
if want ocr; then
  DEST="$ROOT/ocr/ppocrv5-mobile"
  if [ ! -f "$DEST/det.prepped.onnx" ]; then
    mkdir -p "$DEST/det" "$DEST/rec"
    fetch PaddlePaddle/PP-OCRv5_mobile_det_onnx "$DEST/det" \
      --include "inference.onnx"
    fetch PaddlePaddle/PP-OCRv5_mobile_rec_onnx "$DEST/rec" \
      --include "inference.onnx" "inference.yml"

    if [ -f "$DEST/rec/inference.yml" ]; then
      echo "[fetch] extracting the CTC vocabulary from inference.yml"
      python - "$DEST/rec/inference.yml" "$DEST/dict.txt" <<'PY'
import sys
import yaml
yml, out = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(yml, encoding="utf-8"))
chars = data["PostProcess"]["character_dict"]
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for c in chars:
        fh.write(c.replace("\n", "") + "\n")
print(f"[fetch] wrote {len(chars)} dict entries -> {out}")
PY
    fi

    if [ -f "$DEST/det/inference.onnx" ] && [ -f "$DEST/rec/inference.onnx" ]; then
      echo "[prep] scripts/prep-ppocr-onnx.py"
      python scripts/prep-ppocr-onnx.py "$DEST/det/inference.onnx" "$DEST/det.prepped.onnx" || {
        echo "[fetch] det prep failed — the OCR row will be reported as skipped"; }
      python scripts/prep-ppocr-onnx.py "$DEST/rec/inference.onnx" "$DEST/rec.prepped.onnx" || {
        echo "[fetch] rec prep failed — the OCR row will be reported as skipped"; }
    fi
  fi
fi

# ---- Chronos2 --------------------------------------------------------------
if want chronos2; then
  fetch amazon/chronos-t5-small "$ROOT/chronos2/chronos-t5-small" \
    --exclude "*.bin" "*.msgpack" "*.h5" "*.ot"
fi

# ---- Decision (Laya) -------------------------------------------------------
# The upstream repo ships safetensors only — the ONNX graph comes from
# scripts/prep-laya-onnx.py (needs the upstream export script too).
if want decision; then
  DEST="$ROOT/decision/laya"
  if [ ! -f "$DEST/model.onnx" ]; then
    mkdir -p "$DEST"
    echo "[fetch] convaiinnovations/laya -> $DEST"
    "$HF" download convaiinnovations/laya --local-dir "$DEST" || {
      echo "[fetch] laya download failed — the decision row will be reported as skipped"; }
  fi
fi

echo "[fetch] done. Root: $ROOT"
du -sh "$ROOT"/* 2>/dev/null | sort -h | tail -20
