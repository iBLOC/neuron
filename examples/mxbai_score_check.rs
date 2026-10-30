//! One-off quality check for the mxbai yes/no scorer: loads the backbone
//! directly (same steps as `MxbaiRerankRuntime::load`), then
//! 1. greedy-generates after the relevance prompt — coherent output
//!    proves the forward path; garbage means the forward is broken,
//! 2. prints `logit["1"] − logit["0"]` per doc — the ordering proves the
//!    scorer's direction.
//!
//! ```bash
//! cargo run --release --example mxbai_score_check --features candle
//! ```

use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use tokenizers::Tokenizer;

fn model_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("MXBAI_CHECK_MODEL_DIR") {
        return std::path::PathBuf::from(p);
    }
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    std::path::PathBuf::from(home).join(".neuron/models/rerank/mxbai-rerank-base-v2")
}

fn mxbai_prompt(query: &str, doc: &str) -> String {
    format!(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nquery: {query}\ndocument: {doc}\nYou are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = model_dir();
    let dev = Device::Cpu;
    let cfg: Qwen2Config =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("{e}"))?;
    let vb = unsafe {
        neuron_lib::candle_compat::from_mmaped_safetensors(
            &[&dir.join("model.safetensors")],
            DType::F32,
            &dev,
        )
    }
    .map_err(|e| format!("{e}"))?;
    let mut qwen = ModelForCausalLM::new(&cfg, vb)?;
    qwen.clear_kv_cache();

    let query = "What is the capital of France?";
    let docs = [
        "Paris is the capital and largest city of France.",
        "Bananas are a good source of potassium.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a programming language.",
    ];

    let generative_check = std::env::var("MXBAI_CHECK_MODEL_DIR").is_ok();
    // Exactly what candle_runtime's chat path renders (ChatML, no system
    // message when absent) — this shape is KNOWN to answer "Paris".
    let prompt_for = |doc: &str| {
        if generative_check {
            format!("<|im_start|>user
What is the capital of France? Answer in one word.<|im_end|>
<|im_start|>assistant
")
        } else {
            mxbai_prompt(query, doc)
        }
    };
    println!("query: {query}");
    for (i, doc) in docs.iter().enumerate() {
        let prompt = if generative_check {
            prompt_for("")
        } else {
            prompt_for(doc)
        };
        // The server encodes with add_special_tokens=true.
        let enc = tok.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();

        // 1. Greedy generation probe: is the forward healthy?
        // Official candle-qwen2 loop shape: ONE full-sequence prefill at
        // offset 0, then one token per step with a growing offset.
        qwen.clear_kv_cache();
        let mut out = String::new();
        let mut offset = ids.len();
        let mut next = {
            let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
            // ModelForCausalLM::forward returns [batch, 1, vocab] —
            // already sliced to the last position.
            let logits = qwen.forward(&input, 0)?;
            logits.i((0, 0))?.argmax(candle::D::Minus1)?.to_scalar::<u32>()?
        };
        for _ in 0..4 {
            out.push_str(&tok.id_to_token(next).unwrap_or("?".into()));
            if next == 151645 {
                break;
            }
            let input = Tensor::new(&[next], &dev)?.unsqueeze(0)?;
            let logits = qwen.forward(&input, offset)?;
            offset += 1;
            next = logits.i((0, 0))?.argmax(candle::D::Minus1)?.to_scalar::<u32>()?;
        }
        qwen.clear_kv_cache();

        // 2. Verdict logit diff at the last prompt position.
        qwen.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
        let logits = qwen.forward(&input, 0)?;
        let last = logits.i((0, 0))?;
        let l_yes = last.i(16 as usize)?.to_scalar::<f32>()?;
        let l_no = last.i(15 as usize)?.to_scalar::<f32>()?;
        let score = 1.0 / (1.0 + (-(l_yes - l_no)).exp());

        println!(
            "doc[{i}] gen={out:?}  logit_yes={l_yes:.2} logit_no={l_no:.2}  diff={:+.2} score={score:.4}  {doc}",
            l_yes - l_no
        );
    }
    Ok(())
}
