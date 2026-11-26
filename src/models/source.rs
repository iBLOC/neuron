use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct ParsedReference {
    pub model_id: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub current_file: String,
}

#[async_trait]
pub trait Source: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    /// Optional base URL / endpoint of the source. Used by the
    /// `Catalog` to surface the resolved endpoint on `/v1/models`
    /// entries, and by `HuggingFaceSource` to redirect downloads
    /// to a mirror via the `HF_ENDPOINT` env var. Sources that
    /// don't have a configurable endpoint return `None`.
    fn base_url(&self) -> Option<&str> {
        None
    }
    fn parse_reference(&self, reference: &str) -> Result<ParsedReference, String>;
    async fn download(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
    ) -> Result<(), String>;

    /// 拉**这个引擎要的那套权重**（默认实现 = 全拉；真正的源覆写它）。
    ///
    /// ⚠️ **为什么要有这一格**（2026-11-26 真机读数，MIX4 arm64）：设备上
    /// 真跑一次 `hf-mirror:openai/whisper-tiny` ⇒ ⇒ 它**正在下**
    /// `flax_model.msgpack`（85 MB）⇒ ⇒ **candle 根本用不了** ⇒ ⇒ 同一
    /// 个仓里的另外两套（`pytorch_model.bin` 151 MB／`tf_model.h5`
    /// 290 MB）也一样 ⇒ ⇒ ≈ 680 MB 里**只有 151 MB 是要的**。
    ///
    /// ⚠️ **默认实现是「全拉」** ⇒ ⇒ 别的源（与测试里的替身）**一行不用
    /// 改** ⇒ ⇒ 行为不突变 ⇒ ⇒ 覆写是**逐个源自愿**的。
    async fn download_filtered(
        &self,
        reference: &ParsedReference,
        dest_dir: &Path,
        progress_tx: UnboundedSender<DownloadProgress>,
        preferred_weights: &[&str],
    ) -> Result<(), String> {
        let _ = preferred_weights;
        self.download(reference, dest_dir, progress_tx).await
    }
}

/// 引擎**要哪套权重格式**（拉的时候只留那套）。
///
/// ⚠️ **口径**：这是**引擎族**的粗口径 ⇒ ⇒ 真正的判定仍由装载面说了算
/// （装不上 ⇒ ⇒ 报「装不下」那一句真话）⇒ ⇒ 这里只管**别把三套格式
/// 都拖下来**。
pub fn preferred_weight_extensions(
    engine: crate::models::registry::EngineType,
) -> &'static [&'static str] {
    use crate::models::registry::EngineType;
    match engine {
        EngineType::Llm
        | EngineType::Tts
        | EngineType::Embedding
        | EngineType::Rerank
        | EngineType::Vision
        | EngineType::ImageGen => &["safetensors"],
        // ⚠️ ASR 有两族：candle 那族吃 safetensors、sherpa/tract 那族吃
        // onnx ⇒ ⇒ 两个都留着（真仓里只有其中一套）。
        EngineType::Asr => &["safetensors", "onnx"],
        EngineType::Ocr => &["onnx"],
        EngineType::Decision => &[],
    }
}

/// ⚠️⚠️ **同一个仓里往往同时躺着多套权重格式** —— `openai/whisper-tiny`
/// 那一档同时有 `model.safetensors`（151 MB）／`pytorch_model.bin`
/// （151 MB）／`tf_model.h5`（290 MB）／`flax_model.msgpack`（85 MB）
/// ⇒ ⇒ **全拉下来 ≈ 680 MB**，而 candle **只吃 safetensors**
/// ⇒ ⇒ **4.5 倍的带宽**（真机读数：设备上真的在拖那个 85 MB 的 msgpack）。
///
/// ⚠️ **口径**（三句话，每句都有一格判据）：
///   ❶ 仓里**有**引擎要的那套 ⇒ ⇒ **只留那套**，把**同权重的替代格式**
///      剔掉（`pytorch_model*`／`tf_model*`／`flax_model*`）
///      ⇒ ⇒ **别的 `.bin`／`.pth` 不误伤**（例：`audiovae.pth` 是
///      TTS 那条腿自己要用的）；
///   ❷ 仓里**没有**那套 ⇒ ⇒ **全留**（不许把仓拉成空目录 ⇒ ⇒ 那比多下
///      几倍更糟：用户看到「下载完成」却什么也没有）；
///   ❸ `preferred` 为空 ⇒ ⇒ **全留**（这一族没有偏好 ⇒ ⇒ 不猜）。
pub fn filter_weight_files(paths: &[String], preferred: &[&str]) -> Vec<String> {
    if preferred.is_empty() {
        return paths.to_vec();
    }
    let has_preferred = paths.iter().any(|p| {
        let ext = Path::new(p)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        preferred.iter().any(|w| *w == ext)
    });
    if !has_preferred {
        return paths.to_vec();
    }
    paths
        .iter()
        .filter(|p| !is_alternate_weight_file(p))
        .cloned()
        .collect()
}

/// 那几份**同权重的替代格式**（HF 老仓三件套）—— **只认这几个文件名**。
///
/// ⚠️ **为什么按文件名而不按扩展名**：TTS 那条腿的 `audiovae.pth`、
/// 某些仓的 `model.bin` 都是**自己要用的** ⇒ ⇒ 按扩展名一刀切会误伤
/// ⇒ ⇒ 误伤的后果是「引擎注册上了、权重找不到」⇒ ⇒ 只剔**那几份**
/// **同权重替代品**。
fn is_alternate_weight_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    const ALT_EXTS: &[&str] = &["bin", "h5", "msgpack", "ot", "pb"];
    if !ALT_EXTS.contains(&ext.as_str()) {
        return false;
    }
    const ALT_STEMS: &[&str] = &["pytorch_model", "tf_model", "flax_model"];
    let stem = name.split('.').next().unwrap_or(name);
    ALT_STEMS.iter().any(|s| stem.starts_with(s))
}

pub struct SourceRegistry {
    sources: HashMap<String, Arc<dyn Source>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self { sources: HashMap::new() }
    }

    pub fn with_defaults() -> Self {
        Self::with_endpoints(&std::collections::HashMap::new())
    }

    /// Build a `SourceRegistry` with the default HuggingFace and ModelScope
    /// sources, optionally configured with per-source base URL maps.
    ///
    /// Three HF-family sources are registered so the UI can offer a
    /// China-friendly mirror out of the box:
    /// - `huggingface`  → https://huggingface.co (default)
    /// - `hf-mirror`    → https://hf-mirror.com (China-friendly CDN)
    /// - `modelscope`   → https://modelscope.cn (Chinese repo)
    pub fn with_endpoints(
        endpoints: &std::collections::HashMap<String, String>,
    ) -> Self {
        let mut r = Self::new();

        // Primary HuggingFace — `https://huggingface.co` by default,
        // override via `endpoints["huggingface"]`.
        let hf_endpoint = endpoints.get("huggingface").cloned();
        r.register(Arc::new(super::huggingface::HuggingFaceSource::with_endpoint(
            std::path::PathBuf::from("huggingface-cli"),
            hf_endpoint,
        )));

        // HF mirror at hf-mirror.com — same protocol as HF, just a
        // CDN-fronted endpoint. Falls back to the user-supplied
        // override if present.
        let mirror_endpoint = endpoints
            .get("hf-mirror")
            .cloned()
            .or_else(|| Some("https://hf-mirror.com".to_string()));
        r.register(Arc::new(super::huggingface::HuggingFaceSource::with_identity(
            std::path::PathBuf::from("huggingface-cli"),
            mirror_endpoint,
            "hf-mirror",
            "HF Mirror",
        )));

        // ModelScope — Chinese alternative.
        let ms_endpoint = endpoints.get("modelscope").cloned();
        r.register(Arc::new(super::modelscope::ModelScopeSource::with_endpoint(
            std::path::PathBuf::from("modelscope-cli"),
            ms_endpoint,
        )));
        r
    }

    pub fn register(&mut self, source: Arc<dyn Source>) {
        self.sources.insert(source.id().to_string(), source);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Source>> {
        self.sources.get(id).cloned()
    }

    pub fn list(&self) -> Vec<Arc<dyn Source>> {
        self.sources.values().cloned().collect()
    }

    /// Return `(id, base_url)` for every registered source, sorted
    /// by id. Sources without a `base_url` carry `None`. Used by
    /// the planned `/v1/catalog/sources` endpoint and by tests that
    /// need to assert endpoint wiring (e.g. `with_endpoints`
    /// passing a HuggingFace mirror to the HuggingFaceSource).
    pub fn list_endpoints(&self) -> Vec<(String, Option<String>)> {
        let mut out: Vec<(String, Option<String>)> = self
            .sources
            .iter()
            .map(|(id, src)| (id.clone(), src.base_url().map(String::from)))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub fn parse_reference(&self, ref_str: &str) -> Result<(Arc<dyn Source>, ParsedReference), String> {
        let (source_id, rest) = if let Some((src, rest)) = ref_str.split_once(':') {
            (src, rest)
        } else {
            // Default source: ModelScope (Alibaba's open model hub).
            // Users behind the GFW can override with a `hf-mirror:`
            // or `huggingface:` prefix.
            ("modelscope", ref_str)
        };

        let source = self
            .get(source_id)
            .ok_or_else(|| format!("unknown source: {source_id}"))?;

        let parsed = source.parse_reference(rest)?;
        Ok((source, parsed))
    }

    /// Resolve which source a pull request should use, given a
    /// `CatalogConfig`. Lookup order: per-model > per-engine >
    /// default. Returns an error if the resolved source id is not
    /// registered in this registry.
    pub fn resolve_for(
        &self,
        engine: crate::models::EngineType,
        model_id: &str,
        config: &crate::models::catalog::CatalogConfig,
    ) -> Result<Arc<dyn Source>, String> {
        let id = config.resolve_source_id(engine, model_id);
        self.get(&id)
            .ok_or_else(|| format!("source not registered: {id}"))
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    struct DummySource {
        id: String,
    }

    #[async_trait]
    impl Source for DummySource {
        fn id(&self) -> &str { &self.id }
        fn name(&self) -> &str { &self.id }
        fn parse_reference(&self, r: &str) -> Result<ParsedReference, String> {
            Ok(ParsedReference { model_id: r.to_string(), revision: None })
        }
        async fn download(
            &self,
            _r: &ParsedReference,
            _d: &Path,
            _p: UnboundedSender<DownloadProgress>,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    fn dummy(id: &str) -> Arc<dyn Source> {
        Arc::new(DummySource { id: id.to_string() })
    }

    #[test]
    fn test_source_registry_register_and_get() {
        let mut r = SourceRegistry::new();
        r.register(dummy("test"));
        assert!(r.get("test").is_some());
        assert!(r.get("nonexistent").is_none());
    }

    #[test]
    fn test_source_registry_list() {
        let mut r = SourceRegistry::new();
        r.register(dummy("a"));
        r.register(dummy("b"));
        let ids: Vec<String> = r.list().iter().map(|s| s.id().to_string()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }

    #[test]
    fn test_source_registry_default_includes_huggingface() {
        let r = SourceRegistry::with_defaults();
        assert!(r.get("huggingface").is_some());
    }

    #[test]
    fn test_parse_reference_defaults_to_modelscope() {
        let r = SourceRegistry::new();
        let mut r = r;
        r.register(dummy("modelscope"));
        let (src, _parsed) = r.parse_reference("Qwen/X").unwrap();
        assert_eq!(src.id(), "modelscope");
    }

    #[test]
    fn test_parse_reference_with_source_prefix() {
        let mut r = SourceRegistry::new();
        r.register(dummy("huggingface"));
        r.register(dummy("modelscope"));
        let (src, _) = r.parse_reference("modelscope:Qwen/X").unwrap();
        assert_eq!(src.id(), "modelscope");
    }

    #[test]
    fn test_parse_reference_unknown_source_returns_error() {
        let r = SourceRegistry::new();
        let result = r.parse_reference("unknown:Qwen/X");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("unknown source"));
    }

    #[test]
    fn test_resolve_for_returns_engine_override() {
        use crate::models::catalog::CatalogConfig;
        use crate::models::EngineType;
        use std::collections::HashMap;

        let mut r = SourceRegistry::new();
        r.register(dummy("huggingface"));
        r.register(dummy("modelscope"));
        let mut per_engine = HashMap::new();
        per_engine.insert(EngineType::Llm, "modelscope".to_string());
        let cfg = CatalogConfig {
            default_source_id: "huggingface".to_string(),
            per_engine_source: per_engine,
            per_model_source: HashMap::new(),
        };
        let s = r.resolve_for(EngineType::Llm, "x", &cfg).unwrap();
        assert_eq!(s.id(), "modelscope");
    }

    #[test]
    fn test_resolve_for_returns_err_for_missing_source() {
        use crate::models::catalog::CatalogConfig;
        use crate::models::EngineType;
        use std::collections::HashMap;

        let r = SourceRegistry::new();
        let cfg = CatalogConfig {
            default_source_id: "nonexistent".to_string(),
            per_engine_source: HashMap::new(),
            per_model_source: HashMap::new(),
        };
        let err = match r.resolve_for(EngineType::Llm, "x", &cfg) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.contains("not registered"));
    }

    #[test]
    fn test_with_endpoints_passes_hf_endpoint_to_registered_source() {
        use std::collections::HashMap;
        let mut endpoints = HashMap::new();
        endpoints.insert("huggingface".to_string(), "https://hf-mirror.com".to_string());
        let r = SourceRegistry::with_endpoints(&endpoints);
        let endpoints_listed = r.list_endpoints();
        // Three sources: huggingface, hf-mirror (always has a default
        // URL), and modelscope.
        assert_eq!(endpoints_listed.len(), 3);
        // huggingface should carry the custom endpoint.
        let hf = endpoints_listed.iter().find(|(id, _)| id == "huggingface").unwrap();
        assert_eq!(hf.1.as_deref(), Some("https://hf-mirror.com"));
        // hf-mirror always has a default URL.
        let mirror = endpoints_listed.iter().find(|(id, _)| id == "hf-mirror").unwrap();
        assert_eq!(mirror.1.as_deref(), Some("https://hf-mirror.com"));
        // modelscope has no custom endpoint → None.
        let ms = endpoints_listed.iter().find(|(id, _)| id == "modelscope").unwrap();
        assert_eq!(ms.1, None);
    }

    #[test]
    fn test_list_endpoints_returns_none_for_sources_without_base_url() {
        // `DummySource` (defined in this module) does not override
        // `base_url`, so it should appear with `None`.
        let mut r = SourceRegistry::new();
        r.register(dummy("nosrc"));
        let endpoints = r.list_endpoints();
        assert_eq!(endpoints, vec![("nosrc".to_string(), None)]);
    }

    #[test]
    fn test_list_endpoints_is_sorted_by_id() {
        let mut r = SourceRegistry::new();
        r.register(dummy("zzz"));
        r.register(dummy("aaa"));
        r.register(dummy("mmm"));
        let ids: Vec<String> = r.list_endpoints().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["aaa".to_string(), "mmm".to_string(), "zzz".to_string()]);
    }

    /// ⚠️⚠️ **同一个仓里的多套权重格式 ⇒ ⇒ 只留引擎要的那套**（2026-11-26）
    /// —— 判据 F2-a／b／c。
    ///
    /// ⚠️ **真机读数**（MIX4，arm64 Android 13）：真跑一次
    /// `hf-mirror:openai/whisper-tiny` ⇒ ⇒ 它**正在下**
    /// `flax_model.msgpack`（85 MB）⇒ ⇒ **candle 根本用不了** ⇒ ⇒ 同一
    /// 个仓里 ≈ 680 MB，**只有 151 MB 是要的**。
    #[test]
    fn the_weight_filter_keeps_only_the_format_the_engine_loads() {
        let whisper_repo: Vec<String> = [
            "README.md",
            "added_tokens.json",
            "config.json",
            "flax_model.msgpack",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "tf_model.h5",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let kept = filter_weight_files(&whisper_repo, &["safetensors"]);
        // ⚠️ **那三份替代格式必须没了**（它们是**同一份权重**的别的格式）。
        for gone in ["flax_model.msgpack", "pytorch_model.bin", "tf_model.h5"] {
            assert!(
                !kept.iter().any(|k| k == gone),
                "{gone} 是同一份权重的替代格式 ⇒ ⇒ 不该被拖下来；kept={kept:?}"
            );
        }
        // ⚠️ **要用的那份与杂项都得留着**（少一个都装不上）。
        for wanted in [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
        ] {
            assert!(
                kept.iter().any(|k| k == wanted),
                "{wanted} 必须留下；kept={kept:?}"
            );
        }

        // ⚠️ **仓里没有引擎要的那套 ⇒ ⇒ 全留**（不许把仓拉成空目录 ⇒ ⇒
        // 那比多下几倍更糟：用户看到「下载完成」却什么也没有）。
        let pytorch_only: Vec<String> = ["pytorch_model.bin", "config.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = filter_weight_files(&pytorch_only, &["safetensors"]);
        assert_eq!(
            kept, pytorch_only,
            "仓里没有引擎要的那套 ⇒ ⇒ 必须全拉（不许下成空目录）"
        );

        // ⚠️ **别的 .bin／.pth 不误伤**（TTS 那条腿的 audiovae.pth 是
        // 要用的；按扩展名一刀切会把它砍掉 ⇒ ⇒ 「引擎注册上了、权重找不到」）。
        let mixed: Vec<String> = ["model.safetensors", "audiovae.pth", "pytorch_model.bin"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = filter_weight_files(&mixed, &["safetensors"]);
        assert!(
            kept.iter().any(|k| k == "audiovae.pth"),
            "audiovae.pth 不是替代格式 ⇒ ⇒ 不许被砍；kept={kept:?}"
        );
        assert!(
            !kept.iter().any(|k| k == "pytorch_model.bin"),
            "pytorch_model.bin 是替代格式 ⇒ ⇒ 该被砍；kept={kept:?}"
        );

        // ⚠️ **没有偏好 ⇒ ⇒ 全留**（这一族没有偏好 ⇒ ⇒ 不猜）。
        let kept = filter_weight_files(&whisper_repo, &[]);
        assert_eq!(kept.len(), whisper_repo.len());
    }

    /// ⚠️ **每个引擎族要哪套权重格式**（2026-11-26）—— 判据 F2-d。
    #[test]
    fn every_engine_family_declares_the_weight_format_it_loads() {
        use crate::models::registry::EngineType;
        assert_eq!(
            preferred_weight_extensions(EngineType::Llm),
            &["safetensors"]
        );
        // ⚠️ ASR 有两族：candle 吃 safetensors、sherpa/tract 吃 onnx
        // ⇒ ⇒ 两个都要留（真仓里只有其中一套）。
        let asr = preferred_weight_extensions(EngineType::Asr);
        assert!(
            asr.contains(&"safetensors") && asr.contains(&"onnx"),
            "ASR 的偏好必须同时容下两族：{asr:?}"
        );
        assert_eq!(preferred_weight_extensions(EngineType::Ocr), &["onnx"]);
    }
}
