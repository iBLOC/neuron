# Security Policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security reports.
Use GitHub's private vulnerability reporting
(**Security → Report a vulnerability** on this repository), or contact
the maintainers directly. Reports are acknowledged within 7 days.

## Supported builds

Only the latest `main` receives security fixes. Feature flags not
enabled in your build do not compile the affected code — check the
table below before triaging an advisory.

## Dependency risk register

Reviewed 2026-10-13 against open Dependabot alerts; re-reviewed 2026-10-27
with a scripted RustSec cross-check of every locked version (see
"Scanning" below). Each group below is a known, accepted risk with a
defined trigger condition — not an oversight.

### 1. `rmcp` 0.8.5 — via `litert-lm` (feature `litert`)

Open alerts: session-table DoS (high), DNS rebinding (high), OAuth
metadata/header handling (medium ×2).

- **Pulled by**: `litert-lm 0.1.0`, compiled only under the `litert`
  feature; default builds and every other feature never compile it.
- **Exposure**: neuron exposes no MCP endpoints; the affected surfaces
  are transports inside litert-lm's own tooling.
- **Fix status**: fixed in rmcp 3.x; `litert-lm` has no release pulling
  it yet (0.1.0 is Google's only line). Revisit on any litert-lm
  update; drop or vendor the dependency if Google stops maintaining it.

### 2. `tract-onnx` / `tract-nnef` 0.21.10 — model-load parsers

Open alerts: ONNX `external_data` location path traversal (medium),
NNEF `.dat` integer overflow OOB read (medium). The NNEF fix also exists
in 0.21.16, but that patch release pins `half =2.4.1` exactly while the
rest of the tree (candle, tokenizers) needs `half 2.7` — blocked until
tract relaxes the pin.

- **Exposure**: triggered by **loading a maliciously crafted model
  file**. neuron loads operator-provisioned models (downloaded from the
  pinned sources in the catalog, or prepped locally) — treat model
  directories with the same trust as code.
- **Fix status**: the 0.22/0.23 line carries all fixes; upgrading
  0.21 → 0.22/0.23 is a tracked compatibility project (API changes
  across the ASR / decision / OCR / vision paths), not a patch bump.

### 3. `zip` / `zstd` family — via `candle-core` (default feature)

Open alerts: zip archive handling (RUSTSEC-2026-0367/368/369, high) and
a zstd decompression bounds issue (RUSTSEC-2026-0391). All are fixed
only in **major** bumps (`zip >= 9`, `zstd-safe >= 8.2`), which the
upstream `candle-core 0.11` manifest does not allow yet.

- **Pulled by**: `candle-core` (default `candle` feature). neuron's own
  code only ever opens zip archives for **weight files it downloaded
  from the pinned catalog sources** — the same trust domain as the model
  files themselves — never arbitrary user uploads.
- **Fix status**: revisit on the next candle upgrade; if candle stays on
  zip 8.x, consider a `[patch]` override as a stopgap after checking the
  zip 9 API surface.

### 4. `vite` / `esbuild` (docs site dev server)

Open alerts: `server.fs.deny` bypass on Windows (high), optimized-deps
path traversal (medium), launch-editor NTLMv2 disclosure (medium),
dev-server request exposure (medium).

- **Exposure**: **development server only** (`vitepress dev`). The
  deployed docs site is static HTML and unaffected.
- **Fix status**: requires vitepress 2.x (alpha at 2.0.0-alpha.20;
  stable 1.6.4 caps vite at 5.4.21). Revisit when 2.0 goes stable.

## Scanning

- Dependabot alerts are monitored on the repository (npm + cargo
  manifests).
- `npm audit` runs on docs dependency changes; patch-level fixes are
  applied immediately (`npm audit fix`), major bumps are evaluated.
- Since 2026-10-27 the RustSec advisory DB is cross-checked against
  every `Cargo.lock` entry with a scripted semver matcher; in-range
  fixes are applied as surgical `cargo update -p <crate> --precise`
  bumps. The 2026-10-27 pass patched `rustls` 0.23.46 (4 advisories),
  `rustls-webpki` 0.103.16, `tokio` 1.53.3, `webpki-roots` 1.0.10,
  `webpki-root-certs` 1.0.10, `ctutils` 0.4.6 and `zstd-sys` 2.1.2.
