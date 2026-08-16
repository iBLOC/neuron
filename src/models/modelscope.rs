use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

use super::source::{DownloadProgress, ParsedReference, Source};

/// ModelScope source implementation.
///
/// Uses the `modelscope-cli` command-line tool for downloading models.
/// Falls back to HTTP API if CLI is not available.
pub struct ModelScopeSource {
    cli_path: PathBuf,
    endpoint: Option<String>,
}

impl ModelScopeSource {
    pub fn new(cli_path: PathBuf) -> Self {
        Self { cli_path, endpoint: None }
    }

    pub fn with_endpoint(cli_path: PathBuf, endpoint: Option<String>) -> Self {
        Self { cli_path, endpoint }
    }
}

#[async_trait]
impl Source for ModelScopeSource {
    fn id(&self) -> &str { "modelscope" }
    fn name(&self) -> &str { "ModelScope" }
    fn base_url(&self) -> Option<&str> { self.endpoint.as_deref() }

    fn parse_reference(&self, reference: &str) -> Result<ParsedReference, String> {
        // ModelScope references are in format: owner/model_name[@revision]
        let (model_id, revision) = if let Some((id, rev)) = reference.split_once('@') {
            (id.to_string(), Some(rev.to_string()))
        } else {
            (reference.to_string(), None)
        };
        if model_id.is_empty() {
            return Err("model_id cannot be empty".to_string());
        }
        Ok(ParsedReference { model_id, revision })
    }

    async fn download(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String> {
        // Try modelscope-cli first
        if self.cli_path.exists() || which(&self.cli_path.to_string_lossy()).is_some() {
            return self.download_via_cli(reference, dest_dir, progress_tx).await;
        }

        // Fallback: download via HTTP API
        self.download_via_http(reference, dest_dir, progress_tx).await
    }
}

impl ModelScopeSource {
    async fn download_via_cli(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String> {
        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("download")
            .arg(&reference.model_id)
            .arg("--local_dir").arg(dest_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(rev) = &reference.revision {
            cmd.arg("--revision").arg(rev);
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", self.cli_path.display()))?;
        let stdout = child.stdout.take()
            .ok_or("no stdout captured from modelscope-cli")?;

        let tx = progress_tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Some(progress) = parse_progress_line(&line) {
                    let _ = tx.send(progress);
                }
            }
        });

        let status = child.wait().await
            .map_err(|e| format!("waiting on modelscope-cli failed: {e}"))?;
        if !status.success() {
            return Err(format!("modelscope-cli exited with status {status}"));
        }
        Ok(())
    }

    async fn download_via_http(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String> {
        let base_url = self.endpoint.as_deref()
            .unwrap_or("https://modelscope.cn/api/v1");

        // Create destination directory
        tokio::fs::create_dir_all(dest_dir).await
            .map_err(|e| format!("failed to create dir: {e}"))?;

        // Get the file list directly from the repo/files endpoint.
        let files_url = format!(
            "{}/models/{}/repo/files?Recursive=True&Revision={}",
            base_url,
            reference.model_id,
            reference.revision.as_deref().unwrap_or("master"),
        );
        let mut builder = reqwest::Client::builder()
            // Default user-agent; ModelScope's CDN rejects requests
            // without one (returns 401 just like HuggingFace).
            .user_agent("curl/7.88.1");
        for var in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
            if let Ok(url) = std::env::var(var)
                && !url.is_empty()
                    && let Ok(p) = reqwest::Proxy::all(&url) {
                        builder = builder.proxy(p);
                    }
        }

        // Attach the ModelScope access token if the user has
        // configured one (via the `MODELSCOPE_TOKEN` env var). The
        // token also un-gates `modelscope-cli` so the env var serves
        // both code paths. Private/gated repos require this header;
        // public models don't care. Empty / unset is fine — reqwest
        // simply omits the header.
        let ms_token = std::env::var("MODELSCOPE_TOKEN")
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(token) = ms_token {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {token}")
                    .parse()
                    .map_err(|e| format!("invalid MODELSCOPE_TOKEN: {e}"))?,
            );
            builder = builder.default_headers(headers);
        }

        let client = builder.build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;
        let resp = client.get(&files_url)
            .send()
            .await
            .map_err(|e| format!("failed to fetch file list: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("model not found: {}", reference.model_id));
        }

        let body: serde_json::Value = resp.json().await
            .map_err(|e| format!("failed to parse file list: {e}"))?;

        // The response wraps files under `Data.Files` (not a top-level `Files`).
        let files = body["Data"]["Files"].as_array()
            .ok_or_else(|| format!("no files found in response: {}", body))?;

        let total_bytes: u64 = files.iter()
            .filter_map(|f| f["Size"].as_u64())
            .sum();

        let mut bytes_downloaded: u64 = 0;

        for file in files {
            let filename = file["Path"].as_str()
                .ok_or("file path not found")?;

            // Skip directory entries (Type == "tree") — the listing
            // returns both files and folders; folders have Size 0
            // and would otherwise cause us to attempt to write a
            // file at a path that's already a directory.
            let is_dir = file["Type"].as_str() == Some("tree");
            if is_dir {
                continue;
            }

            let size = file["Size"].as_u64().unwrap_or(0);

            let download_url = format!(
                "{}/models/{}/repo?Revision={}&FilePath={}",
                base_url,
                reference.model_id,
                reference.revision.as_deref().unwrap_or("master"),
                filename
            );

            let dest_path = dest_dir.join(filename);
            if let Some(parent) = dest_path.parent() {
                // Skip empty parents (files at the root of dest_dir).
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await
                        .map_err(|e| format!("failed to create parent dir: {e}"))?;
                }
            }

            let _ = progress_tx.send(DownloadProgress {
                bytes_downloaded,
                total_bytes: Some(total_bytes),
                current_file: filename.to_string(),
            });

            let resp = client.get(&download_url)
                .send()
                .await
                .map_err(|e| format!("failed to download {}: {e}", filename))?;

            let bytes = resp.bytes().await
                .map_err(|e| format!("failed to read response: {e}"))?;

            tokio::fs::write(&dest_path, &bytes).await
                .map_err(|e| format!("failed to write file: {e}"))?;

            bytes_downloaded += size;
        }

        let _ = progress_tx.send(DownloadProgress {
            bytes_downloaded: total_bytes,
            total_bytes: Some(total_bytes),
            current_file: "done".to_string(),
        });

        Ok(())
    }
}

/// Parse progress output from modelscope-cli
fn parse_progress_line(line: &str) -> Option<DownloadProgress> {
    // ModelScope CLI outputs progress in various formats
    // Try to parse common patterns
    if line.contains("%") && line.contains("/") {
        // Try to extract bytes info
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 3 {
            let progress_part = parts[1].trim();
            if let Some((done, total)) = parse_bytes(progress_part) {
                return Some(DownloadProgress {
                    bytes_downloaded: done,
                    total_bytes: Some(total),
                    current_file: parts[0].trim().to_string(),
                });
            }
        }
    }
    None
}

fn parse_bytes(s: &str) -> Option<(u64, u64)> {
    // Parse "1.2M/3.4G" style strings
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let done = parse_size(parts[0].trim())?;
    let total = parse_size(parts[1].trim())?;
    Some((done, total))
}

fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num_str, multiplier) = if s.ends_with("G") || s.ends_with("g") {
        (&s[..s.len()-1], 1024 * 1024 * 1024)
    } else if s.ends_with("M") || s.ends_with("m") {
        (&s[..s.len()-1], 1024 * 1024)
    } else if s.ends_with("K") || s.ends_with("k") {
        (&s[..s.len()-1], 1024)
    } else {
        (s, 1)
    };
    let num: f64 = num_str.parse().ok()?;
    Some((num * multiplier as f64) as u64)
}

fn which(cmd: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let full = PathBuf::from(dir).join(cmd);
            if full.exists() {
                return Some(full);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_reference_simple() {
        let source = ModelScopeSource::new(PathBuf::from("modelscope-cli"));
        let result = source.parse_reference("Qwen/Qwen2.5-0.5B-Instruct").unwrap();
        assert_eq!(result.model_id, "Qwen/Qwen2.5-0.5B-Instruct");
        assert!(result.revision.is_none());
    }

    #[test]
    fn test_parse_reference_with_revision() {
        let source = ModelScopeSource::new(PathBuf::from("modelscope-cli"));
        let result = source.parse_reference("Qwen/Qwen2.5-0.5B-Instruct@v1.0").unwrap();
        assert_eq!(result.model_id, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(result.revision.unwrap(), "v1.0");
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("1.5G"), Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64));
        assert_eq!(parse_size("500M"), Some(500 * 1024 * 1024));
        assert_eq!(parse_size("100K"), Some(100 * 1024));
        assert_eq!(parse_size("1024"), Some(1024));
    }
}
