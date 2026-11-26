//! 「自动降级」的**真加载探针**（刀 12）—— 策略算出来的那一档，**真的**加载一遍。
//!
//! **它量什么**（三层，缺一层就不许下结论）：
//! 1. **真事实**：`ModelFacts::from_dir` 从**真模型目录**读出盘上权重与 dtype；
//! 2. **真参数**：`HostCapabilities::detect()` 测出的这台机器的显存／内存；
//! 3. **真加载**：`CandleLlmRuntime::load_with_dtype` 按**策略给的那一档**
//!    真的把权重物化出来，并跑一句生成。
//!
//! ⇒ ⇒ 三个「真」凑齐，才敢说「降级救了这个模型」——
//! 只有前两层，那是**策略级真机读数**（纯函数在真参数上的输出）。
//!
//! **用法**：`NEURON_DTYPE_PROBE_DIR=<模型目录> cargo run --release --features cuda --example dtype_downgrade_load_probe`
//!
//! **退出码**：0 ＝ 加载并生成成功；2 ＝ 探针跑不起来；3 ＝ 策略说装不下（没试加载）。

use neuron_lib::accel::{
    DTypeChoice, HostCapabilities, ModelFacts, StrategyRequest, materialized_bytes,
    resolve_strategy,
};
use neuron_lib::config::DeviceSelection;
use neuron_lib::llm::{CandleLlmRuntime, ChatMessage, ChatRequest, LlmRuntime};
use std::path::PathBuf;
use std::process::Command;

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// 机器真有的显存占用（nvidia-smi，MiB）—— 用来量「**真的物化了多少**」。
fn gpu_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    text.lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = match std::env::var("NEURON_DTYPE_PROBE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            println!("用法：NEURON_DTYPE_PROBE_DIR=<模型目录>（本探针要**真**模型目录）");
            println!("VERDICT: COULD_NOT_RUN");
            std::process::exit(2);
        }
    };

    // ---- ① 真事实 ----
    let facts = match ModelFacts::from_dir(&dir) {
        Ok(f) => f,
        Err(e) => {
            println!("读不到模型目录：{e}");
            println!("VERDICT: COULD_NOT_RUN");
            std::process::exit(2);
        }
    };

    // ---- ② 真参数 ----
    //
    // ⚠⚠️ **ballast：把「可用内存」压进降级窗**（刀 16）——
    //
    // ⇒ ⇒ 为什么要它：「装不下 ⇒ 无损降 bf16 ⇒ 真跑成」这一格要的是
    //   「**可用内存**落在 [bf16 物化, f32 物化) 这个窗口里」⇒ ⇒ 而现有真机
    //   没有一台落在这个窗口里（实测：yx15 可用 1.03 GiB ⇒ 连 bf16 的
    //   1.40 GiB 都装不下；SM-F946U 可用 5.11 GiB ⇒ f32 的 2.80 GiB 装得下
    //   ⇒ ⇒ **窗口是空的** ⇒ ⇒ 那一格在真机上**拿不到**）。
    //
    // ⇒ ⇒ ballast 让它**可测**：占住 N MiB **匿名**内存（不是文件！⇒ 文件是
    //   page cache ⇒ 算「可用」⇒ 压不下来——这一条是真机撞过的）⇒ ⇒ 可用
    //   内存真的降下来 ⇒ ⇒ 策略看到的池就是**压低后**的那个。
    //
    // ⚠ **压在 `detect()` 之前** —— 否则策略看到的是没压过的池 ⇒ ⇒ 这一格
    //   就量不到窗口。
    let ballast_mb: usize = std::env::var("NEURON_DTYPE_PROBE_BALLAST_MB")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let mut ballast: Vec<u8> = Vec::new();
    if ballast_mb > 0 {
        // ⚠⚠️ **必须真的分配并写进去** —— 两个坑都踩过（刀 16）：
        //   ① `reserve_exact` **只改 capacity，不改 len** ⇒ ⇒ `chunks_mut`
        //      遍历的是**空切片** ⇒ ⇒ **一页都没写** ⇒ ⇒ 内核没把它算成
        //      已用 ⇒ ⇒ 「压不下来」（真机实测：ballast 3000 MiB ⇒ 可用
        //      内存**一点没降** ⇒ ⇒ 策略仍选 f32）；
        //   ② 只 `reserve` 不写 ⇒ ⇒ 页没被触碰 ⇒ ⇒ 同样不算已用。
        //   ⇒ ⇒ 用 `resize`（真分配 ＋ 真填）⇒ ⇒ 匿名页**真的**被占住。
        ballast.resize(ballast_mb * 1024 * 1024, 0xA5);
        // ⚠⚠️ **不许让编译器把它 drop 掉** —— NLL 会在最后一次「用」之后
        //   释放 ⇒ ⇒ 若此处不再用它 ⇒ ⇒ **内存当场还给内核** ⇒ ⇒
        //   ballast 白做 ⇒ ⇒ 必须显式 `forget`（或 black_box）。
        std::hint::black_box(&ballast);
        std::mem::forget(ballast);
        println!("  ballast           = 占住 {ballast_mb} MiB 匿名内存（把可用内存压进降级窗）");
    }
    let caps = HostCapabilities::detect();
    println!("== 真事实 ＋ 真参数 ==");
    println!("  dir              = {}", dir.display());
    println!(
        "  arch={} format={:?} 盘上权重={} on_disk_bytes={:?}",
        facts.arch,
        facts.format,
        gib(facts.weight_bytes),
        facts.on_disk_bytes
    );
    println!("  device(build)    = {}", caps.device.name());
    println!(
        "  vram_bytes       = {}",
        caps.vram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    println!(
        "  ram_bytes        = {}",
        caps.ram_bytes.map(gib).unwrap_or_else(|| "没测到".into())
    );
    // ⚠⚠️ **「现在可用」这一栏必须打印** —— 容量池用的是它（不是总量）
    //   ⇒ ⇒ 不打印 ⇒ ⇒ 这一格**看不见池用哪个数** ⇒ ⇒ 真机读数缺一半。
    println!(
        "  ram_available    = {}",
        caps.ram_available_bytes
            .map(gib)
            .unwrap_or_else(|| "没测到".into())
    );

    // ---- ③ 策略 ----
    //
    // ⚠ **旋钮要能穿过来**（刀 15）：`NEURON_LLM_DTYPE` 显式指定的档位必须
    // **真的进策略请求** ⇒ ⇒ 否则这一格**拿不到反事实**（显式要 f32 时的
    // 行为永远量不到 ⇒ ⇒ 只能量到「用户没说话」那一条路）。
    let req = StrategyRequest {
        device: None,
        dtype: neuron_lib::accel::requested_dtype(),
        flash_attn: None,
    };
    let strategy = resolve_strategy(&caps, &facts, &req);
    let f32b = materialized_bytes(&facts, DTypeChoice::F32);
    let bf16b = materialized_bytes(&facts, DTypeChoice::BF16);
    println!("\n== 策略 ==");
    println!("  f32 物化         = {}", gib(f32b));
    println!("  bf16 物化        = {}", gib(bf16b));
    println!(
        "  选中的档位       = {}  ／ 设备 {:?}",
        strategy.dtype.name(),
        strategy.device
    );
    println!("  按那一档要占     = {}", gib(strategy.required_bytes));
    for r in &strategy.reasons {
        println!("  · {r}");
    }
    for u in &strategy.unavailable {
        println!("  ⚠ {u}");
    }
    println!(
        "  这一档跑得起来吗   = {}",
        if neuron_lib::accel::dtype_runnable_for_arch(
            &facts.arch,
            strategy.dtype,
            matches!(strategy.device, DeviceSelection::Nvidia),
            neuron_lib::accel::CompiledFeatures::current().flash_attn,
        ) {
            "跑得起来"
        } else {
            "**跑不起来**"
        }
    );
    if !strategy.unavailable.is_empty() {
        // ⚠ 「装得下但跑不起来」与「装不下」是**两件事** ⇒ ⇒ verdict 要分开
        // ⇒ ⇒ 否则读数会把两种失败混成一个。
        if strategy.unavailable.iter().any(|u| u.contains("跑不起来")) {
            println!("\nVERDICT: STRATEGY_SAYS_NOT_RUNNABLE — 这一档**装得下但跑不起来** ⇒ ⇒ 不试加载（不拿真机去撞）");
        } else {
            println!("\nVERDICT: STRATEGY_SAYS_NO — 策略说这一档拿不到 ⇒ ⇒ 不试加载（不拿真机去撞）");
        }
        std::process::exit(3);
    }

    // ---- ③b. 「强制那一档」模式：量**运行时兜底**那一格 ----
    //
    // ⚠ 策略层**永远**会把跑不起来的那一档换掉 ⇒ ⇒ 于是「兜底那一格」
    // （load 期拒）**够不着** ⇒ ⇒ 要量它，就得**绕开策略**、直接把那一档
    // 递给执行者 ⇒ ⇒ 这一格量的是「**服务会不会倒**」，不是「能不能跑」。
    //
    // ⚠ 用法：`NEURON_DTYPE_PROBE_FORCE_LOAD=1`（其余旋钮照旧）。
    if std::env::var("NEURON_DTYPE_PROBE_FORCE_LOAD").is_ok() {
        println!(
            "\n== 强制档（绕开策略，量运行时兜底）==\n  强制：device={:?} dtype=f32",
            strategy.device
        );
        let t0 = std::time::Instant::now();
        match CandleLlmRuntime::load_with_dtype(
            &facts.model_id,
            &strategy.device,
            DTypeChoice::F32,
        )
        .await
        {
            Ok(_) => {
                println!("  加载成功（{:.1} s）", t0.elapsed().as_secs_f64());
                // ⚠⚠️ **「加载成功」不等于「兜底没生效」** —— 真机撞出来的
                // （刀 16）：在**手机（CPU）**上，f32 **本来就跑得起来**
                // （上游分派：CPU 设备 ⇒ `forward_cpu_flash_attn` ⇒ ⇒
                // candle-nn 的 **dtype 泛型**内核 ⇒ ⇒ f32 明确在列）⇒ ⇒
                // ⇒ 强制加载 f32 **理应成功** ⇒ ⇒ 「成功」**不是**兜底失效。
                //
                // ⇒ ⇒ 判读必须先问「**这一档跑不跑得起来**」⇒ ⇒ 那才是
                //   兜底该不该拒的**前提** ⇒ ⇒ 不许拿「加载成功」直接读成
                //   「兜底没生效」（那是「失败长得像成功」的变体）。
                let runnable = neuron_lib::accel::dtype_runnable_for_arch(
                    &facts.arch,
                    DTypeChoice::F32,
                    matches!(strategy.device, DeviceSelection::Nvidia),
                    neuron_lib::accel::CompiledFeatures::current().flash_attn,
                );
                if runnable {
                    println!("VERDICT: FORCE_LOAD_SUCCEEDED_AND_RUNNABLE — 这一档**跑得起来** ⇒ ⇒ 加载成功是**对的** ⇒ ⇒ 兜底**没有**该拒它（不是兜底失效）");
                    std::process::exit(0);
                }
                println!("VERDICT: FORCE_LOAD_SUCCEEDED_BUT_NOT_RUNNABLE — 这一档**跑不起来**却加载成功了 ⇒ ⇒ 兜底没生效");
                std::process::exit(3);
            }
            Err(e) => {
                println!("  加载被拒：{}", e.message);
                if e.message.contains("跑不起来") {
                    println!("VERDICT: FORCE_LOAD_REFUSED — **load 期**就拒了 ⇒ ⇒ 服务不倒（不是「加载成功、生成炸」）");
                    std::process::exit(0);
                } else {
                    println!("VERDICT: FORCE_LOAD_FAILED_OTHER — 拒了，但原因不是「跑不起来」");
                    std::process::exit(3);
                }
            }
        }
    }

    // ---- ④ 真加载 ----
    println!("\n== 真加载（按策略给的那一档） ==");
    let gpu_before = gpu_used_mib();
    println!(
        "  加载前显存占用     = {}",
        gpu_before
            .map(|m| format!("{m} MiB"))
            .unwrap_or_else(|| "没测到".into())
    );
    let t0 = std::time::Instant::now();
    let runtime = CandleLlmRuntime::new();
    // ⚠ `load_with_dtype` 是 trait 的**关联函数**（没有 `self`）⇒ ⇒ 用
    // 关联函数语法调它，不是方法语法。
    let loaded =
        CandleLlmRuntime::load_with_dtype(&facts.model_id, &strategy.device, strategy.dtype).await;
    let rt = match loaded {
        Ok(rt) => rt,
        Err(e) => {
            println!("  加载失败：{}", e.message);
            // ⚠ **load 期拒**与「加载失败」是两件事 ⇒ ⇒ verdict 要分开。
            if e.message.contains("跑不起来") {
                println!("VERDICT: LOAD_REFUSED_NOT_RUNNABLE — **load 期**就拒了 ⇒ ⇒ 服务不倒（不是「加载成功、生成炸」）");
            } else {
                println!("VERDICT: LOAD_FAILED");
            }
            std::process::exit(3);
        }
    };
    println!("  加载耗时         = {:.1} s", t0.elapsed().as_secs_f64());
    let gpu_after = gpu_used_mib();
    if let (Some(b), Some(a)) = (gpu_before, gpu_after) {
        println!(
            "  加载后显存占用     = {a} MiB（**净增 {} MiB**）",
            a.saturating_sub(b)
        );
        println!(
            "  ⇒ ⇒ 这一档若按 f32 物化会要 {} MiB ⇒ ⇒ 净增与它**不成比例** ⇒ ⇒ 真的按**{}** 物化了",
            f32b / (1024 * 1024),
            strategy.dtype.name()
        );
    }

    // ---- ⑤ 真的跑一句 ----
    let req = ChatRequest {
        model: facts.model_id.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "用一句话说明你是谁。".to_string(),
            reasoning_content: None,
        }],
        temperature: Some(0.2),
        max_tokens: Some(48),
        top_p: None,
        top_k: None,
        stream: false,
        stop: None,
    };
    let t1 = std::time::Instant::now();
    let out = rt.chat(req).await;
    match out {
        Ok(resp) => {
            println!("  生成耗时         = {:.1} s", t1.elapsed().as_secs_f64());
            let text = resp
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();
            println!("  生成结果         = {}", text.trim());
        }
        Err(e) => {
            println!("  生成失败：{}", e.message);
            println!("VERDICT: GENERATE_FAILED");
            std::process::exit(3);
        }
    }
    println!("\nVERDICT: OK — 真权重 ＋ 真参数 ＋ **真加载** ＋ 真生成");
    Ok(())
}
