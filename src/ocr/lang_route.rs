//! Language → recognition-pack routing for the PP-OCR pipeline.
//!
//! PP-OCRv5 ships per-script recognition models as *separate weights + dict
//! pairs* (HF `PaddlePaddle/<script>_PP-OCRv5_mobile_rec`). neuron exposes
//! them as optional `rec-<pack>/` subdirectories of the OCR model dir:
//!
//! ```text
//!   <model_dir>/
//!     det.prepped.onnx  rec.prepped.onnx  dict.txt   ← default (zh/en/ja)
//!     rec-latin/    { rec.prepped.onnx, dict.txt }   ← optional packs
//!     rec-korean/   { ... }
//! ```
//!
//! A request's `languages` field picks the pack: BCP-47 codes route through
//! the static table below (first matching language wins; `zh`/`en`/`ja` and
//! anything unrouted fall back to the default rec), and a language string
//! equal to a discovered pack name (`"latin"`, `"korean"`, …) selects it
//! directly — the escape hatch for scripts this table doesn't enumerate.

/// Normalize a user-supplied language tag: lowercase, trimmed, primary
/// subtag only (`zh-CN` → `zh`, `pt_BR` → `pt`).
pub fn normalize(lang: &str) -> String {
    lang.trim()
        .to_lowercase()
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_string()
}

/// Static BCP-47 → pack mapping (`None` = the default zh/en/ja rec).
/// Families follow PaddleOCR's multilingual model grouping.
pub fn pack_for_language(lang: &str) -> Option<&'static str> {
    match lang {
        "ko" => Some("korean"),
        "ru" | "uk" | "bg" | "be" | "mk" | "sr" | "mn" | "kk" => Some("cyrillic"),
        "fr" | "de" | "es" | "it" | "pt" | "nl" | "pl" | "cs" | "sk" | "sl" | "hr" | "hu"
        | "ro" | "da" | "sv" | "no" | "fi" | "et" | "lv" | "lt" | "ca" | "gl" | "eu" | "cy"
        | "af" | "sq" | "vi" | "id" => Some("latin"),
        "ar" | "fa" | "ur" | "ps" => Some("arabic"),
        "hi" | "mr" | "ne" => Some("devanagari"),
        "th" => Some("thai"),
        "el" => Some("greek"),
        "te" => Some("telugu"),
        "ta" => Some("tamil"),
        // zh / en / ja are covered by the default rec; anything unknown also
        // falls back (the caller warns when a requested language was unrouted).
        _ => None,
    }
}

/// The recognition pack a request should use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecPack {
    /// The model dir's top-level rec graph (default zh/en/ja coverage).
    Default,
    /// A `rec-<name>/` subdirectory pack.
    Named(String),
}

/// Pick the pack for a request's `languages` list. The first language that
/// resolves wins — mixed-script requests route to one rec model, matching
/// PaddleOCR's one-recognizer-per-request pipeline. An exact match against a
/// known pack name takes precedence over the language table.
pub fn select_pack(languages: &[String], known_packs: &[String]) -> RecPack {
    for lang in languages {
        let normalized = normalize(lang);
        if normalized == "default" {
            return RecPack::Default;
        }
        if known_packs.iter().any(|p| *p == normalized) {
            return RecPack::Named(normalized);
        }
        if let Some(pack) = pack_for_language(&normalized) {
            if known_packs.iter().any(|p| p == pack) {
                return RecPack::Named(pack.to_string());
            }
        }
    }
    RecPack::Default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_region_subtags() {
        assert_eq!(normalize("zh-CN"), "zh");
        assert_eq!(normalize("  PT_br "), "pt");
        assert_eq!(normalize("FR"), "fr");
        assert_eq!(normalize("ja"), "ja");
    }

    #[test]
    fn default_scripts_route_to_default_rec() {
        for lang in ["zh", "en", "ja", "zz"] {
            assert_eq!(pack_for_language(lang), None, "{lang}");
        }
    }

    #[test]
    fn families_route_to_their_packs() {
        assert_eq!(pack_for_language("ko"), Some("korean"));
        assert_eq!(pack_for_language("ru"), Some("cyrillic"));
        assert_eq!(pack_for_language("fr"), Some("latin"));
        assert_eq!(pack_for_language("ar"), Some("arabic"));
        assert_eq!(pack_for_language("hi"), Some("devanagari"));
        assert_eq!(pack_for_language("th"), Some("thai"));
        assert_eq!(pack_for_language("el"), Some("greek"));
        assert_eq!(pack_for_language("te"), Some("telugu"));
        assert_eq!(pack_for_language("ta"), Some("tamil"));
    }

    fn packs(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn select_first_resolvable_language_wins() {
        let known = packs(&["latin", "korean", "arabic"]);
        let langs = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        assert_eq!(select_pack(&langs(&["en", "zh"]), &known), RecPack::Default);
        assert_eq!(select_pack(&langs(&["ar", "en"]), &known), RecPack::Named("arabic".into()));
        assert_eq!(select_pack(&langs(&["de-DE", "ar"]), &known), RecPack::Named("latin".into()));
        // routed language whose pack is not present falls back to default
        assert_eq!(select_pack(&langs(&["ko"]), &[]), RecPack::Default);
        // unrouted language falls back to default
        assert_eq!(select_pack(&langs(&["zz"]), &known), RecPack::Default);
        // explicit pack name (the escape hatch)
        assert_eq!(select_pack(&langs(&["latin"]), &known), RecPack::Named("latin".into()));
        // explicit "default" keyword
        assert_eq!(select_pack(&langs(&["latin", "default"]), &known), RecPack::Named("latin".into()));
        assert_eq!(select_pack(&langs(&["default"]), &known), RecPack::Default);
        // empty list → default
        assert_eq!(select_pack(&[], &known), RecPack::Default);
    }
}
