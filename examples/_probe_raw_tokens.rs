// TEMPORARY PROBE - not for commit.
// The NEXT-token distribution at the last prompt position: does the model
// want to emit an added token (<ref>, <box>, <13>, ...) or plain text?
use mistralrs::{IsqBits, ModelBuilder, MultimodalMessages, TextMessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dir = std::env::var("NEURON_WEVISONE_DIR").unwrap();
    let img_path = std::env::var("NEURON_WEVISONE_IMAGE").unwrap();
    let shadow = neuron_lib::llm::config_shape::prepare_mistralrs_model_dir(&dir)?;

    let model = ModelBuilder::new(&shadow)
        .with_logging()
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let bytes = std::fs::read(&img_path)?;
    let img = image::load_from_memory(&bytes)?.to_rgb8();

    let mut msgs = MultimodalMessages::new();
    msgs = msgs.add_multimodal_message(
        TextMessageRole::User,
        "Locate all the instances that match the following description: dog. chair. bush.",
        vec![image::DynamicImage::ImageRgb8(img)],
        vec![],
        vec![],
    );

    let (logits, _tokens) = model.send_raw_chat_request(msgs).await?;
    let tok = tokenizers::Tokenizer::from_file(format!("{shadow}/tokenizer.json"))?;

    for chunk in logits.iter() {
        let shape = chunk.dims().to_vec();
        let flat = chunk.flatten_all().map_err(|e| e.to_string())?;
        let v: Vec<f32> = flat.to_vec1().map_err(|e| e.to_string())?;
        let vocab = v.len() / shape[shape.len() - 2].max(1);
        let rows = v.len() / vocab.max(1);
        let last = &v[(rows - 1) * vocab..rows * vocab];
        let mut idx: Vec<usize> = (0..last.len()).collect();
        idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap_or(std::cmp::Ordering::Equal));
        println!("chunk shape={:?} rows={} vocab={}", shape, rows, vocab);
        println!("top-10 at the LAST position:");
        for i in idx.iter().take(10) {
            let s = tok.decode(&[*i as u32], false).unwrap_or_default();
            println!("  id={} logit={:.4} text={:?}", i, last[*i], s);
        }
        let ref_id = 152670u32;
        let pos = last[ref_id as usize];
        println!("  <ref>(152670) logit={:.4} rank={:?}", pos, idx.iter().position(|x| *x == ref_id as usize));
    }
    Ok(())
}
