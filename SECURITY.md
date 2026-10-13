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

Reviewed 2026-10-13 against open Dependabot alerts. Each group below is
a known, accepted risk with a defined trigger condition — not an
oversight.

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
NNEF `.dat` integer overflow OOB read (medium).

- **Exposure**: triggered by **loading a maliciously crafted model
  file**. neuron loads operator-provisioned models (downloaded from the
  pinned sources in the catalog, or prepped locally) — treat model
  directories with the same trust as code.
- **Fix status**: fixes live in newer tract lines; upgrading
  0.21 → 0.22/0.23 is a tracked compatibility project (API changes
  across the ASR / decision / OCR / vision paths), not a patch bump.

### 3. `vite` / `esbuild` (docs site dev server)

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
