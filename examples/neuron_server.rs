//! Minimal standalone host for neuron.
//!
//! neuron is a library — it has no server binary of its own. This
//! example is the smallest possible host: bind a port, serve until
//! Ctrl-C, drain gracefully. It exists for two reasons:
//!
//! 1. It is the process `scripts/run-llm-speed-compare.sh` spawns so
//!    that neuron can be measured as a *separate process* alongside
//!    ollama / llama.cpp / vLLM. Measuring RSS from outside is the
//!    only way the numbers are symmetric — an in-process host would
//!    fold reqwest/tokio into neuron's footprint.
//! 2. It doubles as the reference host implementation: ~40 lines is
//!    the whole embedding contract.
//!
//! Configuration is entirely environment-driven, matching the rest of
//! the platform (`docs/reference/env-vars.md`):
//!
//! ```bash
//! NEURON_LLM_RUNTIME=candle \
//! NEURON_LLM_MODEL_PATH=C:/Projects/iBLOC/neuron/models/MiniCPM5-2B \
//! NEURON_PORT=18083 \
//! cargo run --release --example neuron_server --features candle
//! ```
//!
//! On startup it prints exactly one machine-readable line to stdout
//! so a supervisor can discover the bound port without guessing:
//!
//! ```text
//! NEURON_SERVER_READY 127.0.0.1:18083 pid=12345
//! ```
//!
//! Everything else goes to stderr, so that line stays parseable.

use neuron_lib::{run_embedded, NeuronConfig, NeuronEmbedConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `from_env` honours NEURON_PORT / NEURON_HOST. A benchmark host
    // should not be world-reachable, so the bind address defaults to
    // loopback; set NEURON_HOST explicitly when you want otherwise.
    let mut config = NeuronConfig::from_env();
    if std::env::var("NEURON_HOST").is_err() {
        config.host = "127.0.0.1".to_string();
    }

    let cfg = NeuronEmbedConfig {
        port: config.port,
        config: config.clone(),
    };

    eprintln!(
        "[neuron-server] starting on {}:{} (model runtime = {})",
        config.host,
        config.port,
        std::env::var("NEURON_LLM_RUNTIME").unwrap_or_else(|_| "<unset>".into()),
    );

    let handle = run_embedded(cfg).await?;
    let addr = handle.addr;

    // Single parseable line on stdout; the driver script greps this.
    println!("NEURON_SERVER_READY {addr} pid={}", std::process::id());
    use std::io::Write as _;
    std::io::stdout().flush()?;

    tokio::signal::ctrl_c().await?;
    eprintln!("[neuron-server] ctrl-c received, draining...");

    // The NeuronHandle owns the server lifetime. Skipping the drain
    // leaks the bound port (see AGENTS.md rule 3).
    handle.request_shutdown();
    handle.join().await;
    eprintln!("[neuron-server] stopped");
    Ok(())
}
