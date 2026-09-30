---
title: License
---

# License

`neuron` is released under the **Apache License 2.0**.

The full license text is in the [`LICENSE`](https://github.com/ibloc/neuron/blob/main/LICENSE) file at the repository root. You can read the [Apache 2.0 license on apache.org](https://www.apache.org/licenses/LICENSE-2.0).

## In short

You are free to:
- **Use** neuron for any purpose (commercial, research, personal, etc.)
- **Modify** the source code
- **Distribute** unmodified or modified copies
- **Sublicense** the code under different terms (e.g. as part of a larger product)

Provided that you:
- **Include the LICENSE file** in any substantial redistribution
- **State changes** you've made to the source
- **Don't use the project's name** (or contributors' names) to endorse derivative products without permission

There is **no warranty** — use at your own risk.

## Third-party attributions

neuron embeds or links against the following projects. Their respective licenses are retained where required by upstream terms; none of these licenses impose copyleft obligations on the neuron source tree.

| Component | Upstream | License |
|---|---|---|
| candle (candle-core / candle-nn / candle-transformers / candle-flash-attn) | https://github.com/huggingface/candle | Apache-2.0 / MIT |
| burn / burn-store | https://github.com/tracel-ai/burn | Apache-2.0 / MIT |
| mistral.rs | https://github.com/EricLBuehler/mistral.rs | MIT |
| litert-lm | https://github.com/maceip/litert-lm-rs | Apache-2.0 |
| axum | https://github.com/tokio-rs/axum | MIT |
| tokio | https://github.com/tokio-rs/tokio | MIT |
| tower / tower-http | https://github.com/tower-rs/tower | MIT |
| hyper | https://github.com/hyperium/hyper | MIT |
| reqwest | https://github.com/seanmonstar/reqwest | Apache-2.0 / MIT |
| serde / serde_json | https://github.com/serde-rs/serde | Apache-2.0 / MIT |
| tokenizers | https://github.com/huggingface/tokenizers | Apache-2.0 |
| thiserror | https://github.com/dtolnay/thiserror | Apache-2.0 / MIT |
| tracing | https://github.com/tokio-rs/tracing | Apache-2.0 / MIT |
| async-trait | https://github.com/dtolnay/async-trait | Apache-2.0 / MIT |
| image | https://github.com/image-rs/image | Apache-2.0 / MIT |
| symphonia (audio decode) | https://github.com/pdeljanov/symphonia | MPL-2.0 |
| criterion (dev-dep benches) | https://github.com/bheisler/criterion.rs | Apache-2.0 / MIT |
| VoxCpm (vendored under `src/tts/voxcpm_burn/`) | in-house (originally self-published at https://github.com/madushan1000/voxcpm_rs) | Apache-2.0 (relicensed to match neuron on 2026-07-19; no 3rd-party copyleft involved) |
| YOLOv8 architecture (clean-room reimpl in `src/vision/yolov8_arch.rs`) | architecture pattern only; clean-room implementation, no code vendored from Ultralytics | Apache-2.0 |

## Model weights

neuron does not redistribute model weights. Weights are loaded at runtime from the user's chosen model source (HuggingFace, hf-mirror, ModelScope, or a local directory). Each weight set is governed by its own license:

- **Qwen / Llama / Phi / Mistral / ChatGLM (LLM)**: Apache-2.0 / MIT (per-model)
- **Whisper (ASR)**: MIT
- **VoxCpm (TTS)**: Apache-2.0
- **MiniLM-L6-v2 (embedding)**: Apache-2.0
- **BGE / mxbai / jina (rerank)**: MIT
- **Cohere (rerank hosted)**: governed by Cohere's API terms
- **YOLOv8 / RT-DETR / DETR / RF-DETR / Florence-2 (vision)**: per-model (most are Apache-2.0 / MIT). Historical note: a Grounding DINO entry was previously listed here under an `ultralytics` Python-subprocess backend; that backend has been removed (neuron is pure Rust), and Grounding DINO went with it.

Users are responsible for downloading weights from their respective sources and complying with each model's license terms. neuron does not enforce weight licensing — it loads whatever weights the user points it at.

## Why Apache-2.0?

Apache-2.0 was chosen for neuron because:
- It's a permissive license (no copyleft) that allows downstream use in any project, including commercial closed-source products
- It includes an explicit patent grant (no surprise patent assertions against users)
- It's widely understood and accepted in the Rust ecosystem
- It plays well with both permissive (MIT, BSD) and weak copyleft (MPL, LGPL) dependencies

Compared to GPL-family licenses (GPL-2.0, GPL-3.0, AGPL-3.0), Apache-2.0 imposes no obligation on derivative works to be open-source. This matters for embedding neuron into commercial products (Tauri desktop apps, server-side processes, mobile binaries).

## Trademark

"neuron" is the project name. The neuron contributors do not claim trademark rights over the name; you may use it freely in your derived products, provided you don't claim official endorsement.

## Contributing

By submitting a pull request to the neuron repository, you agree to license your contribution under the same Apache-2.0 terms as the rest of the project. The project follows the standard [Developer Certificate of Origin (DCO)](https://developercertificate.org/) — sign off your commits with `git commit -s` to certify that you have the right to submit the contribution.

See the [Contributors' Guide](https://github.com/ibloc/neuron/blob/main/CONTRIBUTING.md) for the full process.
