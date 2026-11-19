//! Standalone HTTP harness for the chronos2 (time-series foundation model)
//! surface — the same axum router the production hosts embed, served on its
//! own port so another process (e.g. the parietal lobe) can talk to it.
//!
//! ```bash
//! cargo run --release --example chronos2_http_server --features chronos2
//! # NEURON_CHRONOS2_HTTP_PORT=8083 (default)
//! ```
//!
//! The model directory is resolved exactly like in production: `$HOME/.neuron/models/chronos2/<id>`.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = std::env::var("NEURON_CHRONOS2_HTTP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8083);

    println!(
        "chronos2 discovery: {:?}",
        neuron_lib::chronos2::env_discovery()
    );

    let app = neuron_lib::server::create_router(neuron_lib::config::NeuronConfig::default());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("neuron chronos2 http listening on http://127.0.0.1:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
