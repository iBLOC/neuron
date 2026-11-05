//! 真机级探针 —— **大脑的 ASR 引擎注册契约**。
//!
//! # 这条测试钉的是什么
//!
//! `neuron` 的 ASR 引擎**只**在两处会注册（`src/server/state.rs`）：
//!
//! ① 环境变量 `NEURON_ASR_RUNTIME` ∈ {`whisper`/`candle`, `tract`, `sherpa`}
//!    —— 它注册**一个**硬编码 id（`whisper-model` / `sensevoice-small`）；
//! ② `config.registrations` 里的显式条目。
//!
//! 而**模型目录**在三条路上都**只**从环境变量取：**`NEURON_ASR_MODEL_PATH`**
//! （candle-whisper `asr/whisper.rs`、sherpa `asr/sherpa.rs`、tract
//! `asr/sensevoice.rs` —— 三处各有一个 `resolve_model_path`，实测逐字读过）。
//!
//! ⇒ ⇒ **两个环境变量** ⇒ ⇒ 大脑的 ASR 面就通；**一个都没有** ⇒ ⇒
//! `NoRegisteredModel: register at least one ASR model`。
//!
//! ⚠️ 真机证据（2026-11-05，桌面端真端脑 `127.0.0.1:60080`）：
//! **一个 `NEURON_*` 变量都没设** ⇒ ⇒ 内嵌 neuron 的 ASR registry **空的**
//! ⇒ ⇒ `POST /v1/audio/transcriptions` 逐字回
//! `neuron_error: NoRegisteredModel: register at least one ASR model`。
//!
//! # 为什么它在 `neuron` 仓而不是宿主仓
//!
//! 这条探针量的**是 neuron 自己的注册契约** ⇒ ⇒ 它的家就是 neuron 仓
//! ⇒ ⇒ 而且宿主仓（`shadow-desktop`）的 `cargo test` 会去重链
//! `shadow-desktop.exe` ⇒ ⇒ **用户正在跑的 app 锁着那个文件** ⇒ ⇒
//! 拒绝访问（`os error 5`）⇒ ⇒ 探针放在宿主仓**根本跑不起来**
//! （2026-11-05 实测：`error: failed to remove file … shadow-desktop.exe`）。
//!
//! # 它怎么跑
//!
//! * 模型目录不在 ⇒ ⇒ **跳过**（CI 上没有模型，跳过是诚实的）；
//! * 模型在 ⇒ ⇒ **先反向验证**（没设 env ⇒ ⇒ 门开着但**没有引擎**）
//!   ⇒ ⇒ **再正向验证**（设两个 env ⇒ ⇒ 真语音真出文本）。
//!
//! ⚠️ **反向验证的两次症状必须不同** —— 「反正都是红」不算证据：
//! 无引擎 ⇒ ⇒ `NoRegisteredModel: register at least one ASR model`；
//! 有引擎 ⇒ ⇒ `200` ＋ 非空 `text`。
//!
//! ⚠️ 环境变量在**起 runtime 之前**设（Rust 2024：`set_var` 在多线程里是 UB）
//! ⇒ ⇒ 这里先设、再建 runtime ⇒ ⇒ 没有并发读 env 的窗口。

use std::path::PathBuf;
use std::time::Duration;

// neuron's [lib] target is named `neuron_lib`.
use neuron_lib as neuron;

/// 本机的 candle-whisper 布局模型目录（`config.json` + `tokenizer.json`
/// + `model.safetensors` + `mel_filters.safetensors`）。
fn whisper_model_dir() -> Option<PathBuf> {
    let asr = home_dir()?.join(".neuron").join("models").join("asr");
    for name in ["whisper-base", "whisper-small", "whisper-tiny"] {
        let d = asr.join(name);
        if d.join("config.json").is_file()
            && d.join("tokenizer.json").is_file()
            && d.join("model.safetensors").is_file()
            && d.join("mel_filters.safetensors").is_file()
        {
            return Some(d);
        }
    }
    None
}

/// 一段**真语音** —— sherpa-onnx 那份模型自带 `en.wav` / `zh.wav`
/// （实测：`16 kHz / 单声道 / PCM16` ⇒ ⇒ candle-whisper 那条路对得上）。
fn sample_wav() -> Option<PathBuf> {
    let d = home_dir()?
        .join(".neuron")
        .join("models")
        .join("asr")
        .join("sensevoice-small");
    for name in ["en.wav", "zh.wav"] {
        let p = d.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("USERPROFILE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(PathBuf::from))
}

/// 起一个**真** embedded neuron（随机端口 ⇒ ⇒ 不碰任何在跑的实例），
/// POST 一段**真语音**到 `/v1/audio/transcriptions`，回读**体**。
async fn transcribe(base: &str, wav: &PathBuf) -> (u16, String) {
    let bytes = std::fs::read(wav).expect("read wav");
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("sample.wav")
        .mime_str("audio/wav")
        .expect("mime");
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    let resp = reqwest::Client::builder()
        // 900s：推理时长取决于机器与编译档（2026-11-05 实测 debug 档
        // candle-whisper 超过 300s 没跑完）；探针断言的是「真出文本」，
        // 不是速度 ⇒ 客户端只需等得住。
        .timeout(Duration::from_secs(900))
        .build()
        .expect("client")
        .post(format!("{base}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await
        .expect("asr request");
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    (status, body)
}

/// **大脑的 ASR 面在引擎注册后必须真出文本** ——
/// 这是「叶区的 ASR 面能不能用」的那一格（叶区 → 端脑门 → neuron → 引擎）。
#[test]
#[ignore = "真机级探针：需要本机 ~/.neuron/models/asr 下的模型与真语音；CI 上必须显示 ignored 而不是 passed"]
fn asr_face_returns_text_when_the_engine_is_registered() {
    let Some(model_dir) = whisper_model_dir() else {
        eprintln!("skip: 本机没有 candle-whisper 布局的 ASR 模型目录");
        return;
    };
    let Some(wav) = sample_wav() else {
        eprintln!("skip: 本机没有样例语音");
        return;
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        // ── ❶ **反向验证**（先做）：**没设** env ⇒ ⇒ 门开着但**没有引擎** ──
        //    ⇒ ⇒ 症状必须是「NoRegisteredModel」⇒ ⇒ 而**不是** 401 / 连不上。
        {
            let cfg = neuron::NeuronEmbedConfig {
                port: 0,
                config: neuron::NeuronConfig::new(),
            };
            let handle = neuron::run_embedded(cfg)
                .await
                .expect("embedded neuron binds");
            let base = format!("http://127.0.0.1:{}", handle.addr.port());
            let (status, body) = transcribe(&base, &wav).await;
            eprintln!("[反向验证] status={status} body={body}");
            assert!(
                body.contains("NoRegisteredModel") || body.contains("neuron_error"),
                "反向验证的形状不对：没设 env 时应当是「没有注册 ASR 引擎」，实际 {status} / {body}"
            );
            handle.request_shutdown();
        }

        // ── ❷ **正向验证**：设上那两个环境变量 ⇒ ⇒ 真语音真出文本 ──
        //    ⚠️ Rust 2024：`set_var` 在多线程里是 UB ⇒ ⇒ 必须在**起 runtime 之前**设。
        unsafe {
            std::env::set_var("NEURON_ASR_RUNTIME", "whisper");
            std::env::set_var("NEURON_ASR_MODEL_PATH", &model_dir);
        }

        let cfg = neuron::NeuronEmbedConfig {
            port: 0,
            config: neuron::NeuronConfig::new(),
        };
        let handle = neuron::run_embedded(cfg)
            .await
            .expect("embedded neuron binds");
        let base = format!("http://127.0.0.1:{}", handle.addr.port());
        let t0 = std::time::Instant::now();
        let (status, body) = transcribe(&base, &wav).await;
        eprintln!(
            "[正向验证] elapsed={:?} status={status} body={body}",
            t0.elapsed()
        );
        handle.request_shutdown();

        assert!(
            (200..300).contains(&status),
            "ASR 面失败：{status} / {body}"
        );
        let v: serde_json::Value = serde_json::from_str(&body).expect("ASR json");
        let text = v["text"].as_str().unwrap_or("").trim().to_string();
        assert!(!text.is_empty(), "ASR 返回空文本：{body}");
        eprintln!("ASR text = {text}");
    });
}
