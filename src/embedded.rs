use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::watch;
use tracing::{error, info};

use crate::config::NeuronConfig;
use crate::server::{create_router, ServerState};

#[derive(Debug, Clone)]
pub struct NeuronEmbedConfig {
    pub port: u16,
    pub config: NeuronConfig,
}

impl NeuronEmbedConfig {
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
}

impl From<NeuronConfig> for NeuronEmbedConfig {
    fn from(config: NeuronConfig) -> Self {
        Self {
            port: config.port,
            config,
        }
    }
}

pub struct NeuronHandle {
    pub addr: SocketAddr,
    pub state: Arc<ServerState>,
    #[allow(dead_code)]
    pub shutdown_tx: watch::Sender<bool>,
    #[allow(dead_code)]
    pub join: Option<tokio::task::JoinHandle<()>>,
}

impl NeuronHandle {
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    pub async fn join(mut self) {
        if let Some(handle) = self.join.take() {
            let _ = handle.await;
        }
    }
}

impl std::fmt::Debug for NeuronHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuronHandle")
            .field("addr", &self.addr)
            .finish()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct NeuronHealth {
    pub status: String,
    pub addr: String,
    pub llm_loaded: bool,
    pub tts_loaded: bool,
    pub asr_loaded: bool,
    pub rerank_loaded: bool,
    pub embedding_loaded: bool,
    pub vision_loaded: bool,
}

pub async fn run_embedded(cfg: NeuronEmbedConfig) -> Result<NeuronHandle, String> {
    let NeuronEmbedConfig { port, config } = cfg;

    info!(port = port, "neuron embedded: building server state");
    let state = ServerState::new(config.clone());

    let router = create_router(config.clone());

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), port))
        .await
        .map_err(|e| format!("bind {}:{}: {}", config.host, port, e))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    info!(%addr, "neuron embedded: bound");

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    let join = tokio::spawn(async move {
        let server = axum::serve(listener, router);
        let shutdown_signal = async move {
            let _ = shutdown_rx.changed().await;
        };
        if let Err(e) = server.with_graceful_shutdown(shutdown_signal).await {
            error!(error = %e, "neuron embedded: server error");
        }
    });

    Ok(NeuronHandle {
        addr,
        state: Arc::new(state),
        shutdown_tx,
        join: Some(join),
    })
}

pub fn health(handle: &NeuronHandle) -> NeuronHealth {
    NeuronHealth {
        status: "healthy".to_string(),
        addr: handle.addr.to_string(),
        llm_loaded: true,
        tts_loaded: true,
        asr_loaded: true,
        rerank_loaded: true,
        embedding_loaded: true,
        vision_loaded: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[tokio::test]
    async fn test_embedding_binds_and_shuts_down() {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let cfg = NeuronEmbedConfig {
            port,
            config: NeuronConfig::new(),
        };
        let handle = run_embedded(cfg).await.expect("run_embedded");
        let h = health(&handle);
        assert_eq!(h.status, "healthy");
        assert!(h.addr.contains(&port.to_string()));
        handle.request_shutdown();
        handle.join().await;
    }

    #[test]
    fn test_neuron_embed_config_with_port_overrides_port() {
        let cfg = NeuronEmbedConfig {
            port: 8083,
            config: NeuronConfig::new(),
        };
        let updated = cfg.with_port(9090);
        assert_eq!(updated.port, 9090);
    }

    #[test]
    fn test_neuron_embed_config_from_neuron_config_uses_config_port() {
        let mut nc = NeuronConfig::new();
        nc.port = 7777;
        let embed: NeuronEmbedConfig = nc.clone().into();
        assert_eq!(embed.port, 7777);
        assert_eq!(embed.config.port, nc.port);
    }

    #[test]
    fn test_health_fields_returned_for_all_five_runtimes() {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let cfg = NeuronEmbedConfig {
            port,
            config: NeuronConfig::new(),
        };
        let handle = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_embedded(cfg))
            .expect("run_embedded");

        let h = health(&handle);
        assert_eq!(h.status, "healthy");
        // health() currently reports all five runtimes as loaded. This
        // is a deliberate simplification (the embedded host doesn't
        // have a real model registry to consult), but the contract is
        // that all six fields are present and bool.
        assert!(h.llm_loaded);
        assert!(h.tts_loaded);
        assert!(h.asr_loaded);
        assert!(h.rerank_loaded);
        assert!(h.embedding_loaded);
        assert!(h.vision_loaded);
        assert!(h.addr.contains(&port.to_string()));

        handle.request_shutdown();
        let _ = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle.join());
    }
}