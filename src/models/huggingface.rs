use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

use super::source::{DownloadProgress, ParsedReference, Source};

pub struct HuggingFaceSource {
    cli_path: PathBuf,
    endpoint: Option<String>,
    /// Override the registry id (e.g. "hf-mirror" for the China
    /// CDN). Defaults to `"huggingface"`.
    source_id: String,
    /// Override the human-readable display name. Defaults to
    /// `"HuggingFace Hub"`.
    source_name: String,
}

impl HuggingFaceSource {
    pub fn new(cli_path: PathBuf) -> Self {
        Self {
            cli_path,
            endpoint: None,
            source_id: "huggingface".to_string(),
            source_name: "HuggingFace Hub".to_string(),
        }
    }

    /// Construct a HuggingFaceSource with a custom endpoint
    /// (e.g. `https://hf-mirror.com` for users behind the GFW, or
    /// a private hub URL). The endpoint is passed to
    /// `huggingface-cli` via the `HF_ENDPOINT` env var on every
    /// download. When `huggingface-cli` is not on disk the
    /// download falls back to direct HTTP calls against this
    /// endpoint (or `https://huggingface.co` by default).
    pub fn with_endpoint(cli_path: PathBuf, endpoint: Option<String>) -> Self {
        Self {
            cli_path,
            endpoint,
            source_id: "huggingface".to_string(),
            source_name: "HuggingFace Hub".to_string(),
        }
    }

    /// Construct a HuggingFaceSource with a custom id + display
    /// name on top of `with_endpoint`. Used to register multiple
    /// distinct sources that all speak the HF protocol (e.g.
    /// `huggingface` vs `hf-mirror`).
    pub fn with_identity(
        cli_path: PathBuf,
        endpoint: Option<String>,
        source_id: impl Into<String>,
        source_name: impl Into<String>,
    ) -> Self {
        Self {
            cli_path,
            endpoint,
            source_id: source_id.into(),
            source_name: source_name.into(),
        }
    }

    /// True when the configured CLI binary exists on disk (no
    /// PATH lookup — we honour the exact path the caller gave us).
    fn cli_is_available(&self) -> bool {
        // An absolute or relative-to-cwd path that resolves to a file.
        if self.cli_path.is_file() {
            return true;
        }
        // A bare command name — search PATH manually. This mirrors
        // the `which` semantics the model-scope source already uses.
        if let Some(file_name) = self.cli_path.file_name()
            && file_name == self.cli_path.as_os_str()
                && let Ok(path) = std::env::var("PATH") {
                    for dir in path.split(':') {
                        let candidate = PathBuf::from(dir).join(file_name);
                        if candidate.is_file() {
                            return true;
                        }
                    }
                }
        false
    }
}

#[async_trait]
impl Source for HuggingFaceSource {
    fn id(&self) -> &str { &self.source_id }
    fn name(&self) -> &str { &self.source_name }
    fn base_url(&self) -> Option<&str> { self.endpoint.as_deref() }

    fn parse_reference(&self, reference: &str) -> Result<ParsedReference, String> {
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
        // Prefer the CLI when it's available — it handles auth,
        // retries, and ETag caching for us. Fall back to direct
        // HTTP if the CLI binary isn't on disk.
        if self.cli_is_available() {
            return self.download_via_cli(reference, dest_dir, progress_tx).await;
        }
        self.download_via_http(reference, dest_dir, progress_tx).await
    }
}

impl HuggingFaceSource {
    async fn download_via_cli(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String> {
        let mut cmd = tokio::process::Command::new(&self.cli_path);
        cmd.arg("download")
            .arg(&reference.model_id)
            .arg("--local-dir").arg(dest_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(rev) = &reference.revision {
            cmd.arg("--revision").arg(rev);
        }
        if let Some(endpoint) = &self.endpoint {
            cmd.env("HF_ENDPOINT", endpoint);
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", self.cli_path.display()))?;
        let stdout = child.stdout.take()
            .ok_or("no stdout captured from huggingface-cli")?;

        let tx = progress_tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Some(progress) = parse_hf_progress_line(&line) {
                    let _ = tx.send(progress);
                }
            }
        });

        let status = child.wait().await
            .map_err(|e| format!("waiting on huggingface-cli failed: {e}"))?;
        if !status.success() {
            return Err(format!("huggingface-cli exited with status {status}"));
        }
        Ok(())
    }

/// Fallback path used when `huggingface-cli` is not installed.
/// Walks the HF model tree via the REST API and downloads each
/// LFS blob directly with `reqwest`. No auth is sent — gated
/// repos will fail with 401/403 and surface as a clear error.
    async fn download_via_http(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String> {
        // The puller passes the raw reference (`huggingface:owner/name`),
        // but the HF REST API expects just `owner/name`. Strip the
        // source prefix if present so the model_id is correct.
        // Accept any of `huggingface:` / `hf-mirror:` so the
        // download_via_http path stays correct no matter which
        // source the user picked in the dialog.
        let prefixes = ["huggingface:", "hf-mirror:"];
        let stripped = prefixes
            .iter()
            .find_map(|p| reference.model_id.strip_prefix(p));
        let model_id = stripped.unwrap_or(&reference.model_id);
        let base_url = self.endpoint.as_deref()
            .unwrap_or("https://huggingface.co");
        let api_base = format!("{}/api", base_url);
        let revision = reference.revision.as_deref().unwrap_or("main");

        tokio::fs::create_dir_all(dest_dir).await
            .map_err(|e| format!("failed to create dest dir: {e}"))?;

        // HF API requires a User-Agent or it returns 401. Construct
        // a dedicated client with the standard curl-like identity.
        // `reqwest` doesn't honour system proxy env vars by default,
        // so we read them and build a `reqwest::Proxy` from the
        // standard `HTTPS_PROXY` / `HTTP_PROXY` variables — otherwise
        // requests on proxied machines fail with TLS errors or 401.
        // `ALL_PROXY` is intentionally skipped: it can be a SOCKS URL
        // which `reqwest` can't handle without the `socks` feature.
        let mut builder = reqwest::Client::builder()
            .user_agent("curl/7.88.1");
        for var in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
            if let Ok(url) = std::env::var(var)
                && !url.is_empty()
                    && let Ok(p) = reqwest::Proxy::all(&url) {
                        builder = builder.proxy(p);
                    }
        }
        // Attach an HF auth token if the user has configured one
        // (via the `HF_TOKEN` env var). The same token also
        // un-gates `huggingface-cli` so the env var serves both
        // code paths. Empty / unset is fine — reqwest simply
        // omits the header.
        let hf_token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(token) = hf_token {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {token}")
                    .parse()
                    .map_err(|e| format!("invalid HF_TOKEN: {e}"))?,
            );
            builder = builder.default_headers(headers);
        }
        let client = builder
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;

        // List the repo tree via the HF REST API.
        let tree_url = format!(
            "{}/models/{}/tree/{}?recursive=false",
            api_base, model_id, revision,
        );
        let resp = client.get(&tree_url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| format!("failed to fetch tree: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!(
                "model not found: {} (HTTP {} — gated repos require `huggingface-cli` for auth)",
                model_id, resp.status(),
            ));
        }

        let body: serde_json::Value = resp.json().await
            .map_err(|e| format!("failed to parse tree: {e}"))?;

        let entries = body.as_array()
            .ok_or_else(|| format!("unexpected tree response: {body}"))?;

        // Resolve each entry's LFS pointer, then download the raw blob.
        // `path` field is what we want; `type` is "file" | "directory".
        let mut bytes_downloaded: u64 = 0;

        for entry in entries {
            let path = entry["path"].as_str()
                .ok_or_else(|| format!("entry missing path: {entry}"))?;
            let kind = entry["type"].as_str().unwrap_or("file");
            if kind != "file" {
                continue;
            }

            let parent = dest_dir.join(path).parent()
                .map(|p| p.to_path_buf());
            if let Some(p) = parent
                && !p.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(&p).await
                        .map_err(|e| format!("failed to create parent dir: {e}"))?;
                }
            let dest_path = dest_dir.join(path);

            // Build the resolve URL — works for both LFS (redirects to the
            // actual blob) and small non-LFS files (served inline).
            let download_url = format!(
                "{}/{}/resolve/{}/{}",
                base_url, model_id, revision, path,
            );

            let _ = progress_tx.send(DownloadProgress {
                bytes_downloaded,
                total_bytes: None,
                current_file: path.to_string(),
            });

            let file_resp = client.get(&download_url)
                .send()
                .await
                .map_err(|e| format!("download failed for {path}: {e}"))?;

            if !file_resp.status().is_success() {
                return Err(format!(
                    "download failed for {path}: HTTP {} — gated repos require `huggingface-cli` for auth",
                    file_resp.status(),
                ));
            }

            // Stream the response body to disk in 64 KiB chunks.
            // The previous implementation called `resp.bytes().await`
            // which buffered the entire file in memory — fine for a
            // 30 MB config but fatal for an 8 GB model.safetensors.
            // Streaming keeps peak memory bounded to one chunk and
            // pushes intermediate progress updates so the UI
            // progress bar actually moves.
            let expected_size = file_resp.content_length();
            let mut file = tokio::fs::File::create(&dest_path).await
                .map_err(|e| format!("create {path} failed: {e}"))?;
            let mut downloaded_in_file: u64 = 0;
            let mut last_published_at: u64 = 0;
            const PROGRESS_PUBLISH_BYTES: u64 = 1 << 20; // 1 MiB

            let mut stream = file_resp.bytes_stream();
            use futures_util::StreamExt;
            use tokio::io::AsyncWriteExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk
                    .map_err(|e| format!("read body failed: {e}"))?;
                file.write_all(&chunk).await
                    .map_err(|e| format!("write {path} failed: {e}"))?;
                downloaded_in_file += chunk.len() as u64;
                // Throttle progress publishes — the `UnboundedSender`
                // is non-blocking, but the consumer still walks a
                // mutex on every send, so 1 MiB granularity keeps the
                // log + UI responsive without losing detail.
                if downloaded_in_file - last_published_at >= PROGRESS_PUBLISH_BYTES {
                    last_published_at = downloaded_in_file;
                    let total_so_far = bytes_downloaded + downloaded_in_file;
                    let total_for_job = match (expected_size, bytes_downloaded) {
                        (Some(es), 0) => Some(es),
                        // Best-effort: when we know this file's size
                        // we can show the running total for *this*
                        // job; once we move to the next file we
                        // stop reporting a global total to avoid
                        // confusing the user with a decreasing
                        // total_bytes number.
                        _ => None,
                    };
                    let _ = progress_tx.send(DownloadProgress {
                        bytes_downloaded: total_so_far,
                        total_bytes: total_for_job,
                        current_file: path.to_string(),
                    });
                }
            }

            bytes_downloaded += downloaded_in_file;
        }

        let _ = progress_tx.send(DownloadProgress {
            bytes_downloaded,
            total_bytes: Some(bytes_downloaded),
            current_file: "done".to_string(),
        });

        Ok(())
    }
}

/// Parse a single line of huggingface-cli output into a
/// `DownloadProgress`. Handles the three tqdm-shaped lines the
/// CLI actually emits:
///
/// 1. `Downloading <file>: <pct>%|...| <done>/<total> [t<...]
/// 2. `Fetching N files: <pct>%|...| N/N [t<...`
/// 3. `<file>: <pct>%|...| <done>/<total> [t<...`  (no "Downloading " prefix)
///
/// Returns `None` for non-progress lines (warnings, paths, blank).
fn parse_hf_progress_line(line: &str) -> Option<DownloadProgress> {
    let (current_file, body) = split_file_and_body(line)?;

    // tqdm format: <percent>%|<bar>| <done>/<total> [<time>, <speed>]
    let last_pipe = body.rfind('|')?;
    let fraction = body[last_pipe + 1..]
        .split('[')
        .next()?
        .trim();
    let (downloaded, total) = parse_size_pair(fraction)?;

    Some(DownloadProgress {
        bytes_downloaded: downloaded,
        total_bytes: Some(total),
        current_file,
    })
}

/// Split a progress line into `(current_file, rest_after_colon)`.
/// Handles both `Downloading <file>: ...` and `Fetching N files: ...`
/// and bare `<file>: ...` forms. Returns `None` if the line is not
/// a recognizable progress line.
fn split_file_and_body(line: &str) -> Option<(String, &str)> {
    if let Some(rest) = line.strip_prefix("Downloading ") {
        let (file, body) = rest.split_once(':')?;
        return Some((file.trim().to_string(), body));
    }
    if line.starts_with("Fetching ") {
        // `Fetching N files: <bar>` — surface the N count as a
        // pseudo-filename so the API still reports current_file.
        let (head, body) = line.split_once(':')?;
        return Some((head.trim().to_string(), body));
    }
    // Bare `<file>: <bar>` (no "Downloading " prefix).
    if let Some((file, body)) = line.split_once(':')
        && body.contains('|') && body.contains('[') {
            return Some((file.trim().to_string(), body));
        }
    None
}

/// Parse a `<downloaded>/<total>` string with optional size suffix
/// into `(downloaded_u64, total_u64)`. Supports B / kB / MB / GB / TB
/// (decimal) and KiB / MiB / GiB / TiB (binary). Returns `None` if
/// either side is missing or unparseable.
fn parse_size_pair(s: &str) -> Option<(u64, u64)> {
    let (done, total) = s.split_once('/')?;
    let downloaded = parse_size(done.trim())?;
    let total = parse_size(total.trim())?;
    Some((downloaded, total))
}

/// Parse a size string like `"1.2k"`, `"512"`, `"4MiB"`. Recognised
/// suffixes: B, kB, MB, GB, TB (10^n) and KiB, MiB, GiB, TiB (2^10n).
fn parse_size(s: &str) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    // Walk from the right to find where the unit suffix starts.
    // Units can be 1 to 3 characters; the numeric prefix can be a
    // float ("1.2") or integer ("512").
    let (num_part, unit_part) = split_number_unit(s);
    let multiplier: u64 = match unit_part {
        "" | "B" => 1,
        "k" | "kB" => 1_000,
        "M" | "MB" => 1_000_000,
        "G" | "GB" => 1_000_000_000,
        "T" | "TB" => 1_000_000_000_000,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        "TiB" => 1 << 40,
        _ => return None,
    };
    let n: f64 = num_part.parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    Some((n * multiplier as f64) as u64)
}

/// Split `"1.2kB"` into `("1.2", "kB")` and `"512"` into
/// `("512", "")`. Units are matched longest-first so "MiB" wins
/// over "B" and "MB" wins over "B".
fn split_number_unit(s: &str) -> (&str, &str) {
    let bytes = s.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        let c = bytes[i - 1];
        if c.is_ascii_digit() || c == b'.' {
            break;
        }
        i -= 1;
    }
    (&s[..i], &s[i..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_reference_bare() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        let p = s.parse_reference("Qwen/Qwen2.5-0.5B").unwrap();
        assert_eq!(p.model_id, "Qwen/Qwen2.5-0.5B");
        assert_eq!(p.revision, None);
    }

    #[test]
    fn test_parse_reference_with_revision() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        let p = s.parse_reference("Qwen/Qwen2.5-0.5B@main").unwrap();
        assert_eq!(p.model_id, "Qwen/Qwen2.5-0.5B");
        assert_eq!(p.revision, Some("main".to_string()));
    }

    #[test]
    fn test_parse_reference_with_revision_tag() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        let p = s.parse_reference("Qwen/X@v1.0.0").unwrap();
        assert_eq!(p.revision, Some("v1.0.0".to_string()));
    }

    #[test]
    fn test_parse_reference_empty_returns_error() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        assert!(s.parse_reference("").is_err());
    }

    #[test]
    fn test_id_and_name() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        assert_eq!(s.id(), "huggingface");
        assert_eq!(s.name(), "HuggingFace Hub");
    }

    #[test]
    fn test_with_endpoint_sets_base_url() {
        let s = HuggingFaceSource::with_endpoint(
            PathBuf::from("hf"),
            Some("https://hf-mirror.com".to_string()),
        );
        assert_eq!(s.base_url(), Some("https://hf-mirror.com"));
    }

    #[test]
    fn test_default_constructor_has_no_base_url() {
        let s = HuggingFaceSource::new(PathBuf::from("hf"));
        assert_eq!(s.base_url(), None);
    }

    // -- progress-line parsing --

    #[test]
    fn test_parse_hf_progress_line_decimal_units() {
        // The classic "Downloading X: 100%|██████████| 1.2k/1.2k [..]" line.
        let p = parse_hf_progress_line(
            "Downloading README.md: 100%|██████████| 1.2k/1.2k [00:00<00:00, 2.5kB/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "README.md");
        assert_eq!(p.bytes_downloaded, 1_200);
        assert_eq!(p.total_bytes, Some(1_200));
    }

    #[test]
    fn test_parse_hf_progress_line_binary_units() {
        let p = parse_hf_progress_line(
            "Downloading model.safetensors:  42%|#########  | 210MiB/500MiB [00:12<00:16, 16.7MiB/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "model.safetensors");
        assert_eq!(p.bytes_downloaded, 210 * (1u64 << 20));
        assert_eq!(p.total_bytes, Some(500 * (1u64 << 20)));
    }

    #[test]
    fn test_parse_hf_progress_line_floating_point_decimal() {
        // "4.5MB/10MB" — fraction is float, not integer.
        let p = parse_hf_progress_line(
            "Downloading weights.bin:  45%|########   | 4.5MB/10MB [00:01<00:01, 4.5MB/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "weights.bin");
        assert_eq!(p.bytes_downloaded, 4_500_000);
        assert_eq!(p.total_bytes, Some(10_000_000));
    }

    #[test]
    fn test_parse_hf_progress_line_no_unit() {
        let p = parse_hf_progress_line(
            "Downloading config.json: 100%|██████████| 512/512 [00:00<00:00, ?B/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "config.json");
        assert_eq!(p.bytes_downloaded, 512);
        assert_eq!(p.total_bytes, Some(512));
    }

    #[test]
    fn test_parse_hf_progress_line_fetching_aggregate() {
        // The aggregate "Fetching N files" line has no
        // "Downloading" prefix; current_file falls back to the
        // whole "Fetching N files" header. total is the file count
        // (which is small, not bytes) — we surface it as-is rather
        // than fabricate a byte count.
        let p = parse_hf_progress_line(
            "Fetching 2 files: 100%|██████████| 2/2 [00:00<00:00, 23831.27it/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "Fetching 2 files");
        assert_eq!(p.bytes_downloaded, 2);
        assert_eq!(p.total_bytes, Some(2));
    }

    #[test]
    fn test_parse_hf_progress_line_bare_file() {
        // Some CLI variants omit the "Downloading " prefix.
        let p = parse_hf_progress_line(
            "config.json: 100%|██████████| 1k/1k [00:00<00:00, 1kB/s]",
        )
        .unwrap();
        assert_eq!(p.current_file, "config.json");
        assert_eq!(p.bytes_downloaded, 1_000);
        assert_eq!(p.total_bytes, Some(1_000));
    }

    #[test]
    fn test_parse_hf_progress_line_non_progress_returns_none() {
        // Plain log lines, paths, and warnings must not be
        // mistaken for progress.
        assert!(parse_hf_progress_line("").is_none());
        assert!(parse_hf_progress_line(
            "/home/user/.neure/models/llm/qwen3"
        )
        .is_none());
        assert!(parse_hf_progress_line(
            "Some warning: the file already exists"
        )
        .is_none());
    }

    #[test]
    fn test_parse_size_unit_table() {
        // Pin the unit table so an accidental rename doesn't
        // silently shift a "MB" suffix to mean binary MiB.
        assert_eq!(parse_size("100").unwrap(), 100);
        assert_eq!(parse_size("100B").unwrap(), 100);
        assert_eq!(parse_size("1kB").unwrap(), 1_000);
        assert_eq!(parse_size("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_size("1GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_size("1KiB").unwrap(), 1_024);
        assert_eq!(parse_size("1MiB").unwrap(), 1u64 << 20);
        assert_eq!(parse_size("1GiB").unwrap(), 1u64 << 30);
        // Longest-suffix match: "MiB" wins over trailing "B".
        assert_eq!(parse_size("4MiB").unwrap(), 4 * (1u64 << 20));
    }
}