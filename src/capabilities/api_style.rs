use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ApiStyle(pub String);

impl ApiStyle {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
    pub fn custom(s: impl Into<String>) -> Self { Self(s.into()) }

    pub fn openai_chat() -> Self { Self("openai-chat".into()) }
    pub fn openai_audio() -> Self { Self("openai-audio".into()) }
    pub fn openai_embeddings() -> Self { Self("openai-embeddings".into()) }
    pub fn openai_rerank() -> Self { Self("openai-rerank".into()) }
    pub fn openai_vision() -> Self { Self("openai-vision".into()) }
    pub fn openai_images() -> Self { Self("openai-images".into()) }
    pub fn anthropic_messages() -> Self { Self("anthropic-messages".into()) }

    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for ApiStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_helpers() {
        assert_eq!(ApiStyle::openai_chat().as_str(), "openai-chat");
        assert_eq!(ApiStyle::anthropic_messages().as_str(), "anthropic-messages");
        assert_eq!(ApiStyle::openai_audio().as_str(), "openai-audio");
        assert_eq!(ApiStyle::openai_embeddings().as_str(), "openai-embeddings");
        assert_eq!(ApiStyle::openai_rerank().as_str(), "openai-rerank");
        assert_eq!(ApiStyle::openai_images().as_str(), "openai-images");
    }

    #[test]
    fn test_custom() {
        assert_eq!(ApiStyle::custom("slack-events").as_str(), "slack-events");
        assert_eq!(ApiStyle::custom("acme-internal-v2").as_str(), "acme-internal-v2");
    }

    #[test]
    fn test_serde_round_trip() {
        let s = ApiStyle::openai_chat();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"openai-chat\"");
        let back: ApiStyle = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }
}