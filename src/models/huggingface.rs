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

    /// True when a usable CLI binary exists on disk.
    fn cli_is_available(&self) -> bool {
        self.resolve_cli().is_some()
    }

    /// The CLI binary to spawn: prefer the new `hf` CLI, then the
    /// configured path as-is.
    ///
    /// ⚠️ `hf` comes first on purpose: since huggingface_hub 1.x the
    /// legacy `huggingface-cli` is a **deprecated shim that no longer
    /// works** — it prints a usage banner and exits instead of
    /// downloading (verified on a real install, 2026-11-14). Machines
    /// with only the old hub have no `hf` on PATH and fall through to
    /// the configured path, which still works there.
    fn resolve_cli(&self) -> Option<PathBuf> {
        let hf = std::ffi::OsStr::new("hf");
        // A configured path that names the new CLI directly wins.
        if self.cli_path.is_file()
            && self.cli_path.file_name().map(|n| n == hf).unwrap_or(false)
        {
            return Some(self.cli_path.clone());
        }
        if let Some(found) = find_cli_on_path(hf) {
            return Some(found);
        }
        // An absolute / relative-to-cwd configured path.
        if self.cli_path.is_file() {
            return Some(self.cli_path.clone());
        }
        // A bare command name — search PATH manually. This mirrors
        // the `which` semantics the model-scope source already uses.
        if let Some(file_name) = self.cli_path.file_name()
            && file_name == self.cli_path.as_os_str()
        {
            return find_cli_on_path(file_name);
        }
        None
    }
}

/// Search `PATH` for a bare command name, honouring the platform
/// separator via `std::env::split_paths` and the Windows PATHEXT
/// convention — `huggingface-cli` installs as `huggingface-cli.exe`,
/// so the bare name only resolves if we also try the `.exe` suffix.
///
/// ⚠️ The previous implementation did `path.split(':')` — the Unix
/// separator — so on Windows the search never found anything and
/// every Windows machine silently fell back to the raw HTTP
/// downloader even with `huggingface-cli` installed (resumable,
/// token-aware) sitting right there on PATH.
fn find_cli_on_path(file_name: &std::ffi::OsStr) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
        // Bare name without extension → also try the `.exe` form.
        // Harmless on Unix (that file just won't exist) and keeps the
        // helper testable on both platforms.
        let has_no_extension = file_name
            .to_str()
            .map(|n| Path::new(n).extension().is_none())
            .unwrap_or(false);
        if has_no_extension {
            let exe = dir.join(format!("{}.exe", file_name.to_string_lossy()));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
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
        let cli = self
            .resolve_cli()
            .ok_or("no usable HuggingFace CLI found (looked for `hf`, then the configured path)")?;
        let mut cmd = tokio::process::Command::new(&cli);
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

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", cli.display()))?;
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

        // List the repo tree via the HF REST API. A couple of quick
        // retries — the tree call is small but sits on the same flaky
        // link as the blobs, and dying here fails the whole job
        // before a single byte lands.
        let tree_url = format!(
            "{}/models/{}/tree/{}?recursive=false",
            api_base, model_id, revision,
        );
        let resp = {
            const TREE_ATTEMPTS: u32 = 3;
            let mut last_err: Option<String> = None;
            let mut resp = None;
            for attempt in 1..=TREE_ATTEMPTS {
                if attempt > 1 {
                    tokio::time::sleep(std::time::Duration::from_secs(attempt as u64)).await;
                }
                match client.get(&tree_url).header("Accept", "application/json").send().await {
                    Ok(r) => {
                        resp = Some(r);
                        break;
                    }
                    Err(e) => last_err = Some(format!("failed to fetch tree: {e}")),
                }
            }
            resp.ok_or_else(|| last_err.unwrap_or_else(|| "failed to fetch tree".to_string()))?
        };

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

            let downloaded_in_file = Self::fetch_file_resumable(
                &client,
                &download_url,
                &dest_path,
                path,
                bytes_downloaded,
                &progress_tx,
            )
            .await?;

            bytes_downloaded += downloaded_in_file;
        }

        let _ = progress_tx.send(DownloadProgress {
            bytes_downloaded,
            total_bytes: Some(bytes_downloaded),
            current_file: "done".to_string(),
        });

        Ok(())
    }

    /// Download one repo file with **resume + retry** — thin wrapper:
    /// the implementation lives in [`crate::models::resumable`] and
    /// is shared with the ModelScope source. On any mid-stream
    /// failure (reset, stall, decode error) the next attempt
    /// re-issues the GET with a `Range: bytes=<n>-` header and
    /// appends to the same file handle, so a flaky link keeps making
    /// forward progress instead of restarting from zero.
    ///
    /// The failure this exists for: the first connection serves a
    /// few tens of MB and then dies (`read body failed: error
    /// decoding response body`). Without resume every retry started
    /// from byte 0 and died the same way — multi-GB models could
    /// never complete on an unstable link.
    ///
    /// Retry budget: attempts that make **zero** progress are capped
    /// (`MAX_STALLED_ATTEMPTS`); any attempt that moved the cursor
    /// resets that counter. Total attempts are also capped so a
    /// 1-byte-per-30s trickle cannot loop forever. A 30 s idle
    /// timeout converts a hung stream into a normal retry, and
    /// backoff grows 1s→30s, resetting whenever bytes flow again.
    ///
    /// `total_bytes` progress reporting is best-effort `None` here —
    /// the queue UI renders MB counts without it; a correct total
    /// would need re-deriving after every resume (206 content-length
    /// is the *remaining* bytes) and lies more often than it helps.
    async fn fetch_file_resumable(
        client: &reqwest::Client,
        url: &str,
        dest_path: &Path,
        display_name: &str,
        bytes_before_file: u64,
        progress_tx: &UnboundedSender<DownloadProgress>,
    ) -> Result<u64, String> {
        // The implementation moved to `models::resumable` (shared
        // with the ModelScope source) and gained one branch: a `200`
        // WITH a `Content-Range` header is treated as sliced — that
        // is ModelScope's CDN resume shape (probed 2026-11-18), and
        // plain HuggingFace answers proper `206`s, which the shared
        // helper handles identically.
        super::resumable::fetch_file_resumable(
            client,
            url,
            dest_path,
            display_name,
            bytes_before_file,
            progress_tx,
            "neuron/hf",
        )
        .await
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
            "/home/user/.neuron/models/llm/qwen3"
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

    // -- PATH search (find_cli_on_path) --

    #[serial_test::serial]
    #[test]
    fn test_find_cli_on_path_resolves_exe_suffix() {
        // Windows installs `huggingface-cli` as `huggingface-cli.exe`;
        // the bare name must resolve through the `.exe` fallback.
        // (The old implementation split PATH on ':' — the Unix
        // separator — so this lookup never succeeded on Windows.)
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("huggingface-cli.exe");
        std::fs::write(&exe, b"stub").unwrap();

        let old_path = std::env::var_os("PATH");
        // SAFETY: PATH mutation under the serial lock (no other test
        // reads PATH concurrently).
        unsafe {
            std::env::set_var("PATH", tmp.path());
        }
        let found = find_cli_on_path(std::ffi::OsStr::new("huggingface-cli"));
        if let Some(p) = &old_path {
            // SAFETY: same serial-locked restore.
            unsafe {
                std::env::set_var("PATH", p);
            }
        }

        let found = found.expect("exe-suffixed CLI must be found on PATH");
        assert_eq!(found, exe);
    }

    #[serial_test::serial]
    #[test]
    fn test_find_cli_on_path_honors_platform_separator() {
        // Two directories joined with the *platform* separator must
        // both be searched (the old split(':') missed every dir on
        // Windows and, conversely, would mis-split on Unix paths
        // containing ':'-free separators).
        let tmp_a = tempfile::tempdir().unwrap();
        let tmp_b = tempfile::tempdir().unwrap();
        let cli = tmp_b.path().join("my-cli-stub");
        std::fs::write(&cli, b"stub").unwrap();

        let joined = if cfg!(windows) {
            format!("{};{}", tmp_a.path().display(), tmp_b.path().display())
        } else {
            format!("{}:{}", tmp_a.path().display(), tmp_b.path().display())
        };

        let old_path = std::env::var_os("PATH");
        // SAFETY: PATH mutation under the serial lock.
        unsafe {
            std::env::set_var("PATH", joined);
        }
        let found = find_cli_on_path(std::ffi::OsStr::new("my-cli-stub"));
        if let Some(p) = &old_path {
            // SAFETY: same serial-locked restore.
            unsafe {
                std::env::set_var("PATH", p);
            }
        }

        assert_eq!(found.expect("second PATH dir must be searched"), cli);
    }

    #[serial_test::serial]
    #[test]
    fn test_resolve_cli_prefers_new_hf_cli() {
        // hub ≥ 1.x ships a working `hf` binary alongside a broken
        // `huggingface-cli` deprecation shim; the resolver must pick
        // `hf` or downloads die on the shim's usage banner.
        let tmp_hf = tempfile::tempdir().unwrap();
        let tmp_old = tempfile::tempdir().unwrap();
        let hf_bin = tmp_hf.path().join(if cfg!(windows) { "hf.exe" } else { "hf" });
        let old_bin = tmp_old
            .path()
            .join(if cfg!(windows) { "huggingface-cli.exe" } else { "huggingface-cli" });
        std::fs::write(&hf_bin, b"stub").unwrap();
        std::fs::write(&old_bin, b"stub").unwrap();

        let joined = if cfg!(windows) {
            format!("{};{}", tmp_hf.path().display(), tmp_old.path().display())
        } else {
            format!("{}:{}", tmp_hf.path().display(), tmp_old.path().display())
        };
        let old_path = std::env::var_os("PATH");
        // SAFETY: PATH mutation under the serial lock.
        unsafe {
            std::env::set_var("PATH", &joined);
        }
        let src = HuggingFaceSource::new(PathBuf::from("huggingface-cli"));
        let resolved = src.resolve_cli();
        // SAFETY: same serial-locked restore.
        unsafe {
            if let Some(p) = &old_path {
                std::env::set_var("PATH", p);
            }
        }
        assert_eq!(resolved.expect("hf must resolve"), hf_bin);
    }

    #[serial_test::serial]
    #[test]
    fn test_resolve_cli_falls_back_to_legacy_cli() {
        // Old hub installs have no `hf`; the legacy binary must still
        // be picked up (it works there).
        let tmp_old = tempfile::tempdir().unwrap();
        let old_bin = tmp_old
            .path()
            .join(if cfg!(windows) { "huggingface-cli.exe" } else { "huggingface-cli" });
        std::fs::write(&old_bin, b"stub").unwrap();

        let old_path = std::env::var_os("PATH");
        // SAFETY: PATH mutation under the serial lock.
        unsafe {
            std::env::set_var("PATH", tmp_old.path());
        }
        let src = HuggingFaceSource::new(PathBuf::from("huggingface-cli"));
        let resolved = src.resolve_cli();
        // SAFETY: same serial-locked restore.
        unsafe {
            if let Some(p) = &old_path {
                std::env::set_var("PATH", p);
            }
        }
        assert_eq!(resolved.expect("legacy CLI must resolve"), old_bin);
    }

    // -- resume + retry (fetch_file_resumable / download_via_http) --

    use std::sync::Arc;

    /// A raw-socket HF-shaped flaky server. **Raw on purpose**: the
    /// real failure this simulates is a mid-body TCP cut (proxy/GFW
    /// closing the stream), and a hyper-based harness aborts the
    /// whole response — headers included — when a body stream errors,
    /// which the client then sees as `IncompleteMessage` instead of
    /// "got some bytes, then the connection died".
    ///
    /// Behaviour:
    /// - `GET /api/models/<id>/tree/...` → static JSON array (never cut).
    /// - First non-Range GET → `200` with the **full** Content-Length,
    ///   `cut_at` bytes written, then the socket drops (the cut).
    /// - `Range: bytes=N-` GET → `206` with the tail (or, when
    ///   `ignore_range`, a full `200` — the "server ignored our Range"
    ///   negative control).
    async fn spawn_flaky_server(
        payload: Arc<Vec<u8>>,
        cut_at: usize,
        ignore_range: bool,
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let payload = payload.clone();
                tokio::spawn(async move {
                    // Read until end of request headers.
                    let mut buf = [0u8; 8192];
                    let mut req = String::new();
                    loop {
                        let n = match sock.read(&mut buf).await {
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        if n == 0 {
                            return;
                        }
                        req.push_str(&String::from_utf8_lossy(&buf[..n]));
                        if req.contains("\r\n\r\n") {
                            break;
                        }
                    }

                    let is_tree = req
                        .lines()
                        .next()
                        .map(|l| l.contains("/api/models/") && l.contains("/tree/"))
                        .unwrap_or(false);
                    let range_offset: Option<usize> = req
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                        .and_then(|l| l.split_once(':'))
                        .and_then(|(_, v)| v.trim().strip_prefix("bytes="))
                        .and_then(|v| v.split('-').next())
                        .and_then(|v| v.parse().ok());

                    // Tree request → static JSON, complete response.
                    if is_tree {
                        let tree = r#"[{"path":"weights.bin","type":"file"}]"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            tree.len(),
                            tree,
                        );
                        let _ = sock.write_all(resp.as_bytes()).await;
                        return;
                    }

                    // Resume request → 206 tail (or ignored → full 200).
                    if let Some(offset) = range_offset
                        && offset < payload.len()
                    {
                        if ignore_range {
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                payload.len(),
                            );
                            let _ = sock.write_all(resp.as_bytes()).await;
                            let _ = sock.write_all(&payload).await;
                        } else {
                            let tail = &payload[offset..];
                            let resp = format!(
                                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {offset}-{}/*\r\nConnection: close\r\n\r\n",
                                tail.len(),
                                payload.len() - 1,
                            );
                            let _ = sock.write_all(resp.as_bytes()).await;
                            let _ = sock.write_all(tail).await;
                        }
                        return;
                    }

                    // First full request → full Content-Length, partial
                    // body, then the socket drops: the mid-stream cut.
                    let head = &payload[..cut_at.min(payload.len())];
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
                        payload.len(),
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.write_all(head).await;
                    drop(sock);
                });
            }
        });
        format!("http://{addr}")
    }

    fn test_payload(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fetch_file_resumable_recovers_from_mid_stream_cut() {
        // Payload must exceed the 1 MiB progress-publish granularity
        // so the progress assertion can actually observe flow past the
        // cut point.
        let payload = Arc::new(test_payload(4 * 1024 * 1024));
        let base = spawn_flaky_server(payload.clone(), 1024 * 1024, false).await;

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.bin");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let got = HuggingFaceSource::fetch_file_resumable(
            &client,
            &format!("{base}/model.bin"),
            &dest,
            "model.bin",
            0,
            &tx,
        )
        .await
        .expect("download must survive the mid-stream cut");

        assert_eq!(got as usize, payload.len());
        let written = std::fs::read(&dest).unwrap();
        assert_eq!(written, *payload);

        // Progress kept flowing across the resume.
        let mut progress_max = 0u64;
        while let Ok(p) = rx.try_recv() {
            progress_max = progress_max.max(p.bytes_downloaded);
        }
        assert!(progress_max >= 1024 * 1024, "progress must pass the cut point");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_fetch_file_resumable_restarts_when_server_ignores_range() {
        let payload = Arc::new(test_payload(64 * 1024));
        let base = spawn_flaky_server(payload.clone(), 8 * 1024, true).await;

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.bin");

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let got = HuggingFaceSource::fetch_file_resumable(
            &client,
            &format!("{base}/model.bin"),
            &dest,
            "model.bin",
            0,
            &tx,
        )
        .await
        .expect("a 200 after Range must trigger a clean restart");

        assert_eq!(got as usize, payload.len());
        let written = std::fs::read(&dest).unwrap();
        assert_eq!(written, *payload);
    }

    #[serial_test::serial]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_download_via_http_completes_through_a_flaky_link() {
        // Full-source integration: tree fetch → per-file resume, with
        // proxy env vars cleared and PATH blanked so the loopback
        // request can't be hijacked by a developer machine's proxy
        // settings and the *HTTP fallback* is the path under test
        // (a real `hf` on the developer's PATH would otherwise take
        // over and hit the real Hub).
        let payload = Arc::new(test_payload(128 * 1024));
        let base = spawn_flaky_server(payload.clone(), 32 * 1024, false).await;

        let empty_path = tempfile::tempdir().unwrap();
        let saved_vars: Vec<(&str, Option<std::ffi::OsString>)> = [
            "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "PATH",
        ]
        .iter()
        .map(|k| (*k, std::env::var_os(k)))
        .collect();
        // SAFETY: env mutation under the serial lock; the loopback
        // client would otherwise be routed through the proxy, and a
        // real CLI on PATH would shadow the fallback under test.
        unsafe {
            for (k, _) in &saved_vars {
                std::env::remove_var(k);
            }
            std::env::set_var("PATH", empty_path.path());
        }

        let src = HuggingFaceSource::with_endpoint(
            PathBuf::from("no-such-huggingface-cli-stub"),
            Some(base.clone()),
        );
        let parsed = src.parse_reference("test/model").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let result = src.download(&parsed, &dest, tx).await;

        // SAFETY: same serial-locked restore.
        unsafe {
            for (k, v) in &saved_vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }

        result.expect("full download_via_http must survive a flaky link");
        let written = std::fs::read(dest.join("weights.bin")).unwrap();
        assert_eq!(written, *payload);
        // Progress stream was alive.
        assert!(rx.try_recv().is_ok());
    }
}