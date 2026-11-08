//! **本地小模型守不守得住「意图 JSON 契约」** —— 真机探针。
//!
//! ## 这条探针钉的是什么
//!
//! 端脑的纪律是「**意图由大脑给**」（2026-11-08）：端脑不再自己用关键词猜，
//! 改向大脑要一次**结构化分类**（JSON 契约，9 种意图）⇒ ⇒ 大脑给不出可用意图
//! ⇒ ⇒ 端脑明确报错，绝不回落自猜。
//!
//! ⇒ ⇒ 这条纪律在**云端模型**上已经量过（真云 24/24 契约内、全部忠实映射）。
//! 但产品还有**另一条路**：**本地模型权重**（本机 `~/.neuron/models/llm/` 下
//! 的 qwen3-0.6b 等）⇒ ⇒ 那条路上「小模型守不守得住 JSON 契约」是**未知数**
//! ⇒ ⇒ 守不住 ⇒ ⇒ 端脑会**每一轮都明确报错**（纪律没错，但产品不可用）。
//!
//! ## 手法
//!
//! 直接用 neuron's **candle LLM 运行时**（`CandleLlmRuntime`）跑真权重
//! ⇒ ⇒ 提示词**逐字复制**端脑 `src/lobes/intent.rs` 里的那段
//! `INTENT_SYSTEM_PROMPT`（同一份契约，同一个问题集）⇒ ⇒ 量出来的就是
//! 「端脑那条路上，小模型会给什么」。
//!
//! ⚠️ 端脑侧的**映射**是纯函数（云端 24/24 证明过：大脑说 A ⇒ 端脑给 A）⇒ ⇒
//! 这里量的是**模型侧**：契约内命中率、失败形态、必填字段完整性。
//!
//! ## 怎么跑
//!
//! `cargo test --features candle --test llm_intent_contract_probe -- --ignored --nocapture`
//! （权重不在 ⇒ ⇒ **跳过**并打印原因；跳过是诚实的，不是假装通过）

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{ChatMessage, ChatRequest, LlmImpl, LlmRuntimeRegistry};
use neuron_lib::NeuronConfig;

/// 本机那个小模型的**注册名**（也是 `~/.neuron/models/llm/<id>` 的目录名）。
const LOCAL_MODEL: &str = "qwen3-0.6b";

/// 每句跑几轮（默认 1 —— 本机 CPU 上单轮就要几分钟，见下）。
const ROUNDS: usize = 1;

/// 契约内的意图名（与端脑 `Intent` 变体一一对应）。
const CONTRACT: [&str; 9] = [
    "chat",
    "task_create",
    "task_confirm",
    "task_query",
    "task_update",
    "memory_query",
    "agent_query",
    "visual_qa",
    "visual_task",
];

/// ⚠️ **逐字复制**端脑 `src/lobes/intent.rs` 的 `INTENT_SYSTEM_PROMPT`
/// ⇒ ⇒ 同一个契约 ⇒ ⇒ 两边的读数可以对照。
const INTENT_SYSTEM_PROMPT: &str = concat!(
    "你是一个意图分类器。只输出**一个 JSON 对象**，不要任何解释、不要代码围栏、不要输出思考过程。\n",
    "字段 \"intent\" 取值只能是下列之一：\n",
    "  \"chat\" —— 闲聊或大脑可以直接回答的话。\n",
    "  \"task_create\" —— 用户要建任务。字段：title(必填), description, due_time, priority, tags[], reminder{remind_before, repeat}。\n",
    "  \"task_confirm\" —— 用户在确认/否定一个待建任务。字段：task_id(必填), confirmed(bool), modifications{title, due_time, priority, reminder}。\n",
    "  \"task_query\" —— 用户要查任务。字段：query。\n",
    "  \"task_update\" —— 用户要改任务。字段：task_id(必填), title, status, due_time, priority。\n",
    "  \"memory_query\" —— 用户要查记忆。字段：query。\n",
    "  \"agent_query\" —— 用户要查有哪些智能体。字段：query。\n",
    "  \"visual_qa\" —— 用户在问摄像头当前看到了什么。字段：question(必填), source_hint(设备标签, 可空)。\n",
    "  \"visual_task\" —— 用户要持续盯着摄像头找某物。字段：question(必填), target_class(必填, 如 person/car), source_hint(可空)。\n",
    "例：{\"intent\":\"task_create\",\"title\":\"和客户开会\",\"due_time\":\"明天下午三点\"}"
);

/// 问句集 —— 前四条**清晰**，后四条**歧义**（歧义才是失败率所在）。
const QUESTIONS: [(&str, &str); 8] = [
    ("task_create", "帮我创建一个任务：明天下午三点和客户开会"),
    ("task_query", "我今天有哪些任务要处理"),
    ("chat", "你好，今天心情不错"),
    ("memory_query", "你还记得我上次跟你说过什么吗"),
    ("chat|task_query|memory_query", "帮我看看明天有什么事"),
    ("task_update|chat", "把上次那个删了"),
    ("visual_qa|chat", "你说那个视频里是什么"),
    ("task_update|task_confirm|chat", "任务完成了"),
];

/// 本次跑几道问句（默认 3）—— ⚠️ 本机 CPU 上 0.6B **单轮生成要几分钟**
/// ⇒ ⇒ 一次跑满 8 句会拖到小时级 ⇒ ⇒ 范围由环境变量控制，默认先拿**形状**。
fn question_limit() -> usize {
    std::env::var("NEURON_PROBE_QUESTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
        .min(QUESTIONS.len())
}

/// 生成上限（默认 64 token —— 一个意图 JSON 用不了那么多）。
fn max_tokens() -> u32 {
    std::env::var("NEURON_PROBE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

/// 本机权重目录（neuron 的模型库）。
fn model_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    PathBuf::from(home)
        .join(".neuron")
        .join("models")
        .join("llm")
        .join(LOCAL_MODEL)
}

/// 剥掉 ``` / ```json 围栏（与端脑 `intent.rs` 同义）。
fn strip_fence(content: &str) -> String {
    let trimmed = content.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let body = trimmed.trim_start_matches("```");
    let body = body.strip_prefix("json").unwrap_or(body).trim_start();
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim().to_string()
}

/// 这一轮的回答 ⇒ (意图名, 解析出来的字段)。
///
/// ⚠️ 解析规则**逐字照抄**端脑 `parse_intent_json` 的容错（剥围栏、
/// 取 `intent` 字段）⇒ ⇒ 两边的「能不能解析」是同一把尺子。
fn parse_reply(reply: &str) -> (Option<String>, Option<serde_json::Value>) {
    let cleaned = strip_fence(reply);
    let v: serde_json::Value = match serde_json::from_str(&cleaned) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let kind = v.get("intent").and_then(|x| x.as_str()).map(str::to_string);
    (kind, Some(v))
}

/// 回答里**带没带思考块**（`<think>` … `</think>`）。
///
/// ⚠️ 小模型（qwen3-0.6b）默认**先想一段再答** ⇒ ⇒ 而生成上限很小 ⇒ ⇒ 思考
/// 还没写完就**被截断** ⇒ ⇒ 端脑那边看到的是**一段半截的思考**，不是 JSON
/// ⇒ ⇒ 这个形状要**单独记一档**（不然读数只会显示「不是 JSON」，看不出病根）。
fn has_think(reply: &str) -> bool {
    reply.contains("<think>") || reply.contains("</think>")
}

/// 这个意图的**必填字段**齐不齐（照端脑契约里标「必填」的那些）。
fn missing_required(kind: &str, v: &serde_json::Value) -> Vec<&'static str> {
    let text = |key: &str| v.get(key).and_then(|x| x.as_str()).unwrap_or("");
    let mut missing = Vec::new();
    match kind {
        "task_create" => {
            if text("title").trim().is_empty() {
                missing.push("title");
            }
        }
        "task_confirm" | "task_update" => {
            if text("task_id").trim().is_empty() {
                missing.push("task_id");
            }
        }
        "visual_qa" => {
            if text("question").trim().is_empty() {
                missing.push("question");
            }
        }
        "visual_task" => {
            if text("question").trim().is_empty() {
                missing.push("question");
            }
            if text("target_class").trim().is_empty() {
                missing.push("target_class");
            }
        }
        _ => {}
    }
    missing
}

/// 真机探针：小模型在**端脑那条意图契约**上守不守得住。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads a real local model — run it explicitly"]
async fn the_local_small_model_keeps_the_intent_contract() {
    let dir = model_dir();
    if !dir.join("config.json").is_file() || !dir.join("tokenizer.json").is_file() {
        eprintln!("skip: local model weights are missing at {}", dir.display());
        return;
    }

    let config = Arc::new(NeuronConfig::new());
    let registry = LlmRuntimeRegistry::new(config);

    // 资源预算：生产注册用 8 GB；本机不够就逐级降档 ⇒ ⇒ 失败时是**明确的错误**，
    // 不是「假装有引擎」。
    let mut registered = false;
    for budget in [
        8_000_000_000u64,
        4_000_000_000,
        2_000_000_000,
        1_000_000_000,
    ] {
        match registry.register(
            LOCAL_MODEL.to_string(),
            LlmImpl::Candle,
            DeviceSelection::Cpu,
            budget,
        ) {
            Ok(()) => {
                eprintln!(
                    "registered {} on CPU with a {} MB budget",
                    LOCAL_MODEL,
                    budget / 1_000_000
                );
                registered = true;
                break;
            }
            Err(e) => eprintln!(
                "register with a {} MB budget failed: {}",
                budget / 1_000_000,
                e
            ),
        }
    }
    assert!(
        registered,
        "the local model must be registrable on this machine"
    );

    let runtime = registry
        .runtime_for(LOCAL_MODEL)
        .await
        .expect("the local model must load");

    let mut total = 0usize;
    let mut in_contract = 0usize;
    let mut out_of_contract = 0usize;
    let mut not_json = 0usize;
    let mut think_blocks = 0usize;
    let mut incomplete = 0usize;
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut missing_kinds: BTreeMap<String, usize> = BTreeMap::new();

    for (expected, question) in QUESTIONS.iter().take(question_limit()) {
        let mut here: BTreeMap<String, usize> = BTreeMap::new();
        for round in 1..=ROUNDS {
            let req = ChatRequest {
                model: LOCAL_MODEL.to_string(),
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: INTENT_SYSTEM_PROMPT.to_string(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: format!(
                            "请把下面这句话分类成意图。\n\n[用户这一句]\n{}",
                            question
                        ),
                    },
                ],
                temperature: Some(0.0),
                max_tokens: Some(max_tokens()),
                top_p: None,
                top_k: None,
                stream: false,
                stop: None,
            };
            let started = std::time::Instant::now();
            let resp = runtime
                .chat(req)
                .await
                .expect("the local model must answer");
            let elapsed = started.elapsed();
            let reply = resp
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();
            let (kind, parsed) = parse_reply(&reply);
            total += 1;

            let short: String = reply.chars().take(160).collect();
            let label = kind.clone().unwrap_or_else(|| "<no-intent>".to_string());
            *here.entry(label).or_insert(0) += 1;

            match (&kind, &parsed) {
                (Some(k), Some(v)) if CONTRACT.contains(&k.as_str()) => {
                    in_contract += 1;
                    let missing = missing_required(k, v);
                    if !missing.is_empty() {
                        incomplete += 1;
                        *missing_kinds
                            .entry(format!("{}({})", k, missing.join("+")))
                            .or_insert(0) += 1;
                    }
                }
                (Some(k), _) => {
                    out_of_contract += 1;
                    *kinds.entry(format!("out-of-contract({})", k)).or_insert(0) += 1;
                }
                (None, _) => {
                    not_json += 1;
                    if has_think(&reply) {
                        think_blocks += 1;
                    }
                    *kinds
                        .entry(if has_think(&reply) {
                            "not-json-think-block".to_string()
                        } else {
                            "not-json-or-no-intent-field".to_string()
                        })
                        .or_insert(0) += 1;
                }
            }

            eprintln!(
                "[{}] round {}/{} elapsed={:.1}s reply={:?}",
                expected,
                round,
                ROUNDS,
                elapsed.as_secs_f64(),
                short
            );
        }
        eprintln!("QUESTION {:<28} local distribution {:?}", expected, here);
    }

    eprintln!("--- LOCAL CONTRACT SUMMARY ---");
    eprintln!(
        "TOTAL={} IN_CONTRACT={} OUT_OF_CONTRACT={} NOT_JSON={} (THINK_BLOCK={}) INCOMPLETE_FIELDS={}",
        total, in_contract, out_of_contract, not_json, think_blocks, incomplete
    );
    eprintln!("failure kinds: {:?}", kinds);
    eprintln!("missing-field kinds: {:?}", missing_kinds);
    assert_eq!(
        in_contract + out_of_contract + not_json,
        total,
        "every round must be classified exactly once"
    );
}
