//! Integration tests for neure's HTTP API.
//!
//! These tests require a running neure server. Start the server with:
//!
//! ```bash
//! cargo run
//! ```

#[cfg(test)]
mod integration {
    use reqwest::Client;
    use serde_json::json;

    const BASE_URL: &str = "http://localhost:8083";

    fn client() -> Client {
        Client::new()
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let resp = client()
            .get(format!("{}/health", BASE_URL))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["service"], "neure");
    }

    #[tokio::test]
    async fn test_info_endpoint() {
        let resp = client()
            .get(format!("{}/v1/info", BASE_URL))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert_eq!(json["object"], "info");
        assert!(json["version"].is_string());
        assert!(json["capabilities"]["llm"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let resp = client()
            .get(format!("{}/v1/models", BASE_URL))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert_eq!(json["object"], "list");
        assert!(json["data"].is_array());
    }

    #[tokio::test]
    async fn test_chat_completions_empty_messages() {
        let resp = client()
            .post(format!("{}/v1/chat/completions", BASE_URL))
            .json(&json!({
                "model": "test",
                "messages": []
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 400);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert!(json["error"]["message"].as_str().unwrap().contains("messages"));
    }

    #[tokio::test]
    async fn test_forecast_empty_series() {
        let resp = client()
            .post(format!("{}/v1/forecast", BASE_URL))
            .json(&json!({
                "model": "test",
                "series": [],
                "horizon": 5
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 400);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert!(json["error"]["message"].as_str().unwrap().contains("series"));
    }

    #[tokio::test]
    async fn test_forecast_zero_horizon() {
        let resp = client()
            .post(format!("{}/v1/forecast", BASE_URL))
            .json(&json!({
                "model": "test",
                "series": [1.0, 2.0, 3.0],
                "horizon": 0
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 400);
        let json: serde_json::Value = resp.json().await.expect("json parse failed");
        assert!(json["error"]["message"].as_str().unwrap().contains("horizon"));
    }

    #[tokio::test]
    async fn test_rerank_empty_query() {
        let resp = client()
            .post(format!("{}/v1/rerank", BASE_URL))
            .json(&json!({
                "model": "test",
                "query": "",
                "documents": ["doc1"]
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn test_embedding_empty_input() {
        let resp = client()
            .post(format!("{}/v1/embeddings", BASE_URL))
            .json(&json!({
                "model": "test",
                "input": ""
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 400);
    }
}
