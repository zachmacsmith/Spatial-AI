# API Rate Limiting - Quick Reference

## Problem
Hitting API rate limits (requests per minute), not token limits.

## Solution
Added rate limiting to `video_processor.py` to automatically space out API calls.

## Configuration

In `BatchParameters`, you can control rate limiting:

```python
# Enable/disable rate limiting
enable_rate_limiting: bool = True

# Maximum API requests per minute (adjust based on your API tier)
api_requests_per_minute: int = 10

# Safety buffer (10% = requests will be 10% slower than limit)
rate_limit_buffer: float = 0.1
```

## How It Works

1. **RateLimiter class** tracks API calls and enforces minimum time between requests
2. **Thread-safe** - works correctly with parallel processing
3. **Automatic delays** - waits before each API call if needed
4. **Statistics** - reports total API calls at end

## Example Usage

### For Free Tier (Low Limits)
```python
from video_processing.batch_parameters import BatchParameters

params = BatchParameters(
    enable_rate_limiting=True,
    api_requests_per_minute=5,  # Very conservative
    rate_limit_buffer=0.2  # 20% safety margin
)
```

### For Paid Tier (Higher Limits)
```python
params = BatchParameters(
    enable_rate_limiting=True,
    api_requests_per_minute=50,  # Higher limit
    rate_limit_buffer=0.1  # 10% safety margin
)
```

### Disable Rate Limiting (If Not Needed)
```python
params = BatchParameters(
    enable_rate_limiting=False
)
```

## How to Find Your API Limits

### Claude (Anthropic)
- Free tier: ~5 requests/min
- Tier 1: ~50 requests/min
- Tier 2: ~1000 requests/min
- Check: https://console.anthropic.com/settings/limits

### Gemini (Google)
- Free tier: ~15 requests/min
- Paid tier: ~60 requests/min
- Check: https://ai.google.dev/pricing

### OpenAI
- Free tier: ~3 requests/min
- Tier 1: ~500 requests/min
- Check: https://platform.openai.com/account/limits

## What You'll See

When rate limiting is active:
```
Rate Limiter: 10 req/min, min interval: 6.60s
Processing 5 keyframes...
  [Rate Limit] Waiting 3.45s...
Keyframe 1: using tool
  [Rate Limit] Waiting 2.12s...
Keyframe 2: idle
...

API Rate Limiting Stats:
  Total API requests: 15
  Configured limit: 10 req/min
  Min interval: 6.60s
```

## Tips

1. **Start conservative**: Use a low `api_requests_per_minute` first
2. **Adjust based on errors**: If you still get rate limit errors, lower the value
3. **Use buffer**: The `rate_limit_buffer` adds safety margin
4. **Parallel processing**: Rate limiter works with `max_workers` settings
5. **Processing time**: Lower rate limits = longer processing time

## Calculation

```
min_interval = (60 / requests_per_minute) * (1 + buffer)

Example with 10 req/min and 10% buffer:
min_interval = (60 / 10) * 1.1 = 6.6 seconds between requests
```

## Integration with Presets

All presets now have rate limiting enabled by default:
- `PRESET_BASIC`: 10 req/min
- `PRESET_OBJECTS`: 10 req/min  
- `PRESET_RELATIONSHIPS`: 10 req/min
- `PRESET_HTML_ANALYSIS`: 10 req/min
- `PRESET_FULL`: 10 req/min

You can override when creating params:
```python
params = PRESET_FULL.copy()
params.api_requests_per_minute = 20  # Increase if you have higher limits
```
