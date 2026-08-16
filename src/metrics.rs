//! Performance metrics collection and monitoring.
//!
//! Tracks inference latency, throughput, and resource usage
//! for all model types (LLM, TTS, ASR, Rerank, Embedding, Vision).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Global metrics collector.
static METRICS: std::sync::LazyLock<MetricsCollector> =
    std::sync::LazyLock::new(MetricsCollector::new);

/// Performance metrics for a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub count: u64,
    pub total_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub avg_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
}

/// System resource metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub cpu_usage_percent: f32,
    pub gpu_memory_used_bytes: Option<u64>,
    pub gpu_memory_total_bytes: Option<u64>,
    pub gpu_usage_percent: Option<f32>,
}

/// Comprehensive metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub operations: HashMap<String, OperationMetrics>,
    pub resources: ResourceMetrics,
    pub uptime_seconds: f64,
}

/// Latency histogram for percentile calculations.
struct LatencyHistogram {
    samples: Vec<f64>,
    max_samples: usize,
}

impl LatencyHistogram {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    fn record(&mut self, duration_ms: f64) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(duration_ms);
    }

    fn percentiles(&self) -> (f64, f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let p50 = sorted[len * 50 / 100];
        let p95 = sorted[len * 95 / 100];
        let p99 = sorted[len * 99 / 100];

        (p50, p95, p99)
    }
}

/// Metrics collector that tracks all operations.
struct MetricsCollector {
    start_time: Instant,
    operations: Mutex<HashMap<String, LatencyHistogram>>,
    counters: HashMap<String, AtomicU64>,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            operations: Mutex::new(HashMap::new()),
            counters: HashMap::new(),
        }
    }

    fn record_operation(&self, name: &str, duration: Duration) {
        let duration_ms = duration.as_secs_f64() * 1000.0;

        let mut operations = self.operations.lock().unwrap();
        let histogram = operations
            .entry(name.to_string())
            .or_insert_with(|| LatencyHistogram::new(10000));
        histogram.record(duration_ms);
    }

    fn get_snapshot(&self) -> MetricsSnapshot {
        let operations = self.operations.lock().unwrap();
        let mut op_metrics = HashMap::new();

        for (name, histogram) in operations.iter() {
            let count = histogram.samples.len() as u64;
            if count == 0 {
                continue;
            }

            let total: f64 = histogram.samples.iter().sum();
            let min = histogram.samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = histogram.samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg = total / count as f64;
            let (p50, p95, p99) = histogram.percentiles();

            op_metrics.insert(
                name.clone(),
                OperationMetrics {
                    count,
                    total_duration_ms: total,
                    min_duration_ms: min,
                    max_duration_ms: max,
                    avg_duration_ms: avg,
                    p50_duration_ms: p50,
                    p95_duration_ms: p95,
                    p99_duration_ms: p99,
                },
            );
        }

        MetricsSnapshot {
            timestamp: chrono::Utc::now(),
            operations: op_metrics,
            resources: ResourceMetrics {
                memory_used_bytes: 0,
                memory_total_bytes: 0,
                cpu_usage_percent: 0.0,
                gpu_memory_used_bytes: None,
                gpu_memory_total_bytes: None,
                gpu_usage_percent: None,
            },
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
        }
    }
}

/// Record an operation duration.
pub fn record_operation(name: &str, duration: Duration) {
    METRICS.record_operation(name, duration);
}

/// Get a snapshot of all metrics.
pub fn get_metrics() -> MetricsSnapshot {
    METRICS.get_snapshot()
}

/// Timer for measuring operation duration.
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
        }
    }

    pub fn stop(self) -> Duration {
        let duration = self.start.elapsed();
        record_operation(&self.name, duration);
        duration
    }
}

/// Convenience macro for timing operations.
#[macro_export]
macro_rules! time_operation {
    ($name:expr, $block:block) => {{
        let timer = $crate::metrics::Timer::new($name);
        let result = $block;
        let duration = timer.stop();
        (result, duration)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_records_duration() {
        let timer = Timer::new("test_op");
        std::thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_metrics_snapshot() {
        let timer = Timer::new("snapshot_test");
        std::thread::sleep(Duration::from_millis(5));
        timer.stop();

        let snapshot = get_metrics();
        assert!(snapshot.uptime_seconds >= 0.0);
    }

    #[test]
    fn test_latency_histogram() {
        let mut histogram = LatencyHistogram::new(100);
        for i in 0..100 {
            histogram.record(i as f64);
        }
        let (p50, p95, p99) = histogram.percentiles();
        assert!((p50 - 50.0).abs() < 1.0);
        assert!((p95 - 95.0).abs() < 1.0);
        assert!((p99 - 99.0).abs() < 1.0);
    }

    #[test]
    fn test_operation_metrics() {
        for _ in 0..10 {
            let timer = Timer::new("test_metrics");
            std::thread::sleep(Duration::from_millis(1));
            timer.stop();
        }

        let snapshot = get_metrics();
        let op = snapshot.operations.get("test_metrics").unwrap();
        assert_eq!(op.count, 10);
        assert!(op.avg_duration_ms >= 1.0);
    }
}
