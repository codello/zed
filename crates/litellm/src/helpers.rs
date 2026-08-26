use http_client::http;

pub fn extract_retry_after(headers: &http::HeaderMap) -> Option<std::time::Duration> {
    if let Some(reset) = headers.get("X-RateLimit-Reset") {
        if let Ok(s) = reset.to_str() {
            if let Ok(epoch_ms) = s.parse::<u64>() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                if epoch_ms > now {
                    return Some(std::time::Duration::from_millis(epoch_ms - now));
                }
            }
        }
    }
    None
}

pub fn is_none_or_empty<T: AsRef<[U]>, U>(opt: &Option<T>) -> bool {
    opt.as_ref().is_none_or(|v| v.as_ref().is_empty())
}
