//! Shared **resumable HTTP file download** used by every source that
//! speaks plain HTTP (HuggingFace, ModelScope, ...).
//!
//! Extracted from the HuggingFace source so the ModelScope source
//! stops one-shotting multi-GB files into memory
//! (`resp.bytes()` + one `fs::write`): on an unstable link that path
//! could never finish a big model, and one interrupted retry
//! restarted from byte 0.
//!
//! Contract (same as the original HF implementation, one addition):
//!
//! - On any mid-stream failure the next attempt re-issues the GET
//!   with `Range: bytes=<n>-` and **appends** to the same file, so a
//!   flaky link keeps making forward progress.
//! - `206 Partial Content` = sliced, append.
//! - `200` **with a `Content-Range` header** = the server honored
//!   the range but answered 200 (ModelScope's CDN does exactly this,
//!   probed 2026-11-18: `Accept-Ranges: bytes` + `Content-Range` on a
//!   200) ⇒ the body is the remaining bytes ⇒ append.
//! - `200` **without** `Content-Range` = the server ignored the
//!   range ⇒ restart from the top (truncate).
//! - `416` after a resume = the previous attempt died after writing
//!   the final byte ⇒ the file is complete.
//!
//! Retry budget: zero-progress attempts are capped
//! (`MAX_STALLED_ATTEMPTS`); any attempt that moved the cursor resets
//! the counter. Total attempts are also capped so a 1-byte-per-30s
//! trickle cannot loop forever. A 30 s idle timeout converts a hung
//! stream into a normal retry, and backoff grows 1s→30s, resetting
//! whenever bytes flow again.

use std::path::Path;

use tokio::sync::mpsc::UnboundedSender;

use super::source::DownloadProgress;

const MAX_STALLED_ATTEMPTS: u32 = 8;
const MAX_TOTAL_ATTEMPTS: u32 = 100;
/// No chunk for 30 s = the stream is hung; convert to a retry.
const STREAM_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const PROGRESS_PUBLISH_BYTES: u64 = 1 << 20; // 1 MiB

/// Download one file with **resume + retry**, appending to
/// `dest_path`. Returns the total bytes present in the file when it
/// finished (i.e. the resume cursor at completion).
///
/// `bytes_before_file` is the cumulative byte count of the files
/// downloaded before this one — progress events report
/// `bytes_before_file + offset` so a queue UI can render the whole
/// job, not just this file. `total_bytes` stays `None`: a correct
/// total would need re-deriving after every resume (the sliced
/// content-length is the *remaining* bytes) and lies more often than
/// it helps.
///
/// `log_tag` prefixes the retry log lines (`neuron/hf`,
/// `neuron/ms`, ...) so the origin is still visible in the logs.
pub(crate) async fn fetch_file_resumable(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    display_name: &str,
    bytes_before_file: u64,
    progress_tx: &UnboundedSender<DownloadProgress>,
    log_tag: &str,
) -> Result<u64, String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    // Seed the cursor from an existing partial file: callers keep
    // partials on disk across job retries (the puller no longer
    // deletes them), so a retried pull resumes at the byte level
    // instead of re-fetching from 0. A server that then ignores the
    // Range (200 without Content-Range) resets below.
    let mut offset: u64 = match tokio::fs::metadata(dest_path).await {
        Ok(meta) => meta.len(),
        Err(_) => 0,
    };
    let mut stalled_attempts: u32 = 0;
    let mut total_attempts: u32 = 0;
    let mut backoff_secs: u64 = 1;
    // One handle across attempts: a resume keeps appending in
    // place (the write cursor sits exactly at `offset`), a
    // forced restart recreates (and thus truncates) the file.
    let mut file: Option<tokio::fs::File> = None;

    loop {
        total_attempts += 1;
        if stalled_attempts >= MAX_STALLED_ATTEMPTS || total_attempts > MAX_TOTAL_ATTEMPTS {
            return Err(format!(
                "download failed for {display_name}: gave up after {total_attempts} attempts (last position: byte {offset})"
            ));
        }
        if total_attempts > 1 {
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            backoff_secs = (backoff_secs * 2).min(30);
        }

        let mut req = client.get(url);
        if offset > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let file_resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                stalled_attempts += 1;
                eprintln!(
                    "[{log_tag}] {display_name}: request failed ({e}); retrying from byte {offset}"
                );
                continue;
            }
        };

        // 416 after a resume means the previous attempt died
        // after writing the final byte — the file is complete.
        if offset > 0 && file_resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            break;
        }
        let status = file_resp.status();
        if !status.is_success() {
            // Transient upstream failures (5xx / 429) are worth a
            // resume-retry; hard 4xx (401/403 gated) are not.
            if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                stalled_attempts += 1;
                eprintln!(
                    "[{log_tag}] {display_name}: HTTP {status}; retrying from byte {offset}"
                );
                continue;
            }
            return Err(format!(
                "download failed for {display_name}: HTTP {status} — the repo may be gated or the reference wrong"
            ));
        }

        // We asked for a Range. Three legal shapes:
        // - 206 = sliced ⇒ append;
        // - 200 WITH Content-Range = sliced but answered 200
        //   (ModelScope) ⇒ append;
        // - 200 WITHOUT Content-Range = the server ignored the
        //   range ⇒ restart from the top (truncate).
        if offset > 0 && status == reqwest::StatusCode::OK {
            let sliced = file_resp
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .is_some();
            if !sliced {
                offset = 0;
                file = None;
            }
        }

        if file.is_none() {
            // offset > 0 = resuming from a seeded/partial file ⇒ open
            // for append (File::create would truncate the prefix);
            // offset == 0 = fresh download ⇒ create/truncate. The
            // restart branch below resets `file` to None so a
            // truncate-on-restart still happens through this path.
            file = Some(
                if offset > 0 {
                    tokio::fs::OpenOptions::new()
                        .append(true)
                        .open(dest_path)
                        .await
                } else {
                    tokio::fs::File::create(dest_path).await
                }
                .map_err(|e| format!("create {display_name} failed: {e}"))?,
            );
        }
        let file = file.as_mut().expect("file handle just created");

        let mut stream = file_resp.bytes_stream();
        let mut last_published_at: u64 = offset;
        let mut finished = false;
        while !finished {
            let next = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()).await;
            match next {
                // Stream ended cleanly — this file is complete.
                Ok(None) => finished = true,
                // Read error mid-body (reset / decode failure) —
                // fall through to a resume retry.
                Ok(Some(Err(e))) => {
                    stalled_attempts += 1;
                    eprintln!(
                        "[{log_tag}] {display_name}: read body failed at byte {offset} ({e}); retrying with resume"
                    );
                    break;
                }
                // No chunk within the idle timeout — the stream
                // is hung; treat it like a reset.
                Err(_elapsed) => {
                    stalled_attempts += 1;
                    eprintln!(
                        "[{log_tag}] {display_name}: stream idle >{STREAM_IDLE_TIMEOUT:?} at byte {offset}; retrying with resume"
                    );
                    break;
                }
                Ok(Some(Ok(chunk))) => {
                    file.write_all(&chunk)
                        .await
                        .map_err(|e| format!("write {display_name} failed: {e}"))?;
                    offset += chunk.len() as u64;
                    // Forward progress — both retry counters and
                    // the backoff reset.
                    stalled_attempts = 0;
                    backoff_secs = 1;
                    if offset - last_published_at >= PROGRESS_PUBLISH_BYTES {
                        last_published_at = offset;
                        let _ = progress_tx.send(DownloadProgress {
                            bytes_downloaded: bytes_before_file + offset,
                            total_bytes: None,
                            current_file: display_name.to_string(),
                        });
                    }
                }
            }
        }
        if finished {
            break;
        }
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// A 200 **with** `Content-Range` after a Range request must be
    /// treated as sliced (body appended), NOT as "server ignored the
    /// range ⇒ truncate and restart". ModelScope's CDN behaves this
    /// way (probed 2026-11-18); a truncate here would corrupt every
    /// resumed byte range on that source.
    ///
    /// Drive it with a real socket: response #1 declares
    /// `Content-Length: 8` but writes only 4 bytes and closes — the
    /// client sees a mid-body failure and retries with
    /// `Range: bytes=4-`; response #2 answers `200` **with**
    /// `Content-Range: bytes 4-7/8` and the remaining 4 bytes. The
    /// file must end up as the full 8 bytes (`ABCDEFGH`); the broken
    /// truncate-on-200 behavior would leave only `EFGH`.
    #[tokio::test]
    async fn resume_accepts_200_with_content_range() {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut hit: usize = 0;
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                if hit == 0 {
                    // Lie about the length, deliver half, drop the
                    // connection: a mid-body failure for the client.
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nABCD",
                        )
                        .await;
                    let _ = sock.shutdown().await;
                } else {
                    // Sliced-but-200: the ModelScope CDN shape.
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Range: bytes 4-7/8\r\nConnection: close\r\n\r\nEFGH",
                        )
                        .await;
                    let _ = sock.shutdown().await;
                }
                hit += 1;
            }
        });

        let dir = tempfile::TempDir::new().unwrap();
        let dest = dir.path().join("f.bin");
        let (tx, mut rx) = mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        let url = format!("http://{addr}/f.bin");

        let written = fetch_file_resumable(&client, &url, &dest, "probe.bin", 0, &tx, "neuron/test")
            .await
            .expect("resumable download must survive the 200+Content-Range resume");

        let on_disk = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(on_disk, b"ABCDEFGH", "200+Content-Range must append, not truncate");
        assert_eq!(written, 8);
        // 8 bytes < PROGRESS_PUBLISH_BYTES (1 MiB): no progress event
        // is expected for a transfer this small — the channel stays
        // empty and that is correct, not a missed publish.
        assert!(rx.try_recv().is_err(), "sub-threshold transfers publish no progress");
    }

    /// The negative control for the branch above: a 200 **without**
    /// `Content-Range` after a Range request means the server ignored
    /// the range — the helper must truncate and restart, not append
    /// (appending would duplicate the first half).
    #[tokio::test]
    async fn resume_truncates_on_200_without_content_range() {
        use tokio::io::AsyncReadExt;
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut hit: usize = 0;
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                if hit == 0 {
                    // Same mid-body failure setup.
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nABCD",
                        )
                        .await;
                    let _ = sock.shutdown().await;
                } else {
                    // Ignored the Range: full 8-byte body, no
                    // Content-Range, status 200.
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nABCDEFGH",
                        )
                        .await;
                    let _ = sock.shutdown().await;
                }
                hit += 1;
            }
        });

        let dir = tempfile::TempDir::new().unwrap();
        let dest = dir.path().join("f.bin");
        let (tx, _rx) = mpsc::unbounded_channel();
        let client = reqwest::Client::new();
        let url = format!("http://{addr}/f.bin");

        fetch_file_resumable(&client, &url, &dest, "probe.bin", 0, &tx, "neuron/test")
            .await
            .expect("download completes after the restart");

        let on_disk = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(on_disk, b"ABCDEFGH", "restart must truncate, not append");
    }
}
