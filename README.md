# litellm-cfg

A Python script to fetch models from a LiteLLM proxy and output them in opencode.json format.

## Installation

```bash
chmod +x litellm-cfg.py
```

## Configuration Priority

Configuration values are resolved in this priority order (highest to lowest):

1. **CLI arguments** (`-a`, `-u`)
2. **Config file** (`.litellm-cfg.json` or `~/.config/litellm-cfg/config.json`)
3. **Environment variables** (`LITELLM_API_KEY`, `LITELLM_URL`)
4. **Default values** (url: `http://localhost:4000`)

## Configuration File

Create a config file at one of these locations:

- `./.litellm-cfg.json` (current directory)
- `~/.config/litellm-cfg/config.json` (user config)
- Custom path via `LITELLM_CFG_CONFIG` env var or `-c` flag

### Example Config

```json
{
  "api_key": "sk-your-litellm-api-key",
  "url": "http://litellm.example.com:4000"
}
```

## Usage Examples

### Using Config File

```bash
# With config file at .litellm-cfg.json
python3 litellm-cfg.py -m
```

### Using CLI Arguments (overrides config)

```bash
python3 litellm-cfg.py -m -a "sk-xxx" -u "http://localhost:4000"
```

### Using Environment Variables

```bash
export LITELLM_API_KEY="sk-xxx"
export LITELLM_URL="http://localhost:4000"
python3 litellm-cfg.py -m
```

### Custom Config File Path

```bash
python3 litellm-cfg.py -m -c /path/to/my-config.json
```

### Save Output to File

```bash
python3 litellm-cfg.py -m > models.json
```

### Use in opencode.json

```bash
# Create complete opencode.json with models
cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "{env:LITELLM_API_KEY}"
      },
      "models":
EOF

python3 litellm-cfg.py -m >> opencode.json

cat >> opencode.json << 'EOF'
    }
  }
}
EOF
```

## Expected Output

```json
{
  "gpt-4o": {
    "name": "gpt-4o",
    "limit": {
      "context": 128000,
      "output": 4096
    }
  },
  "claude-3-sonnet": {
    "name": "claude-3-sonnet",
    "limit": {
      "context": 200000,
      "output": 4096
    }
  }
}
```

## Token Limit Resolution

The script tries to get token limits in this order:

1. **Admin endpoint** (`/v1/model/info`) - requires admin key
2. **Internal cost map** - known models like GPT-4, Claude, etc.
3. **Skip** - models without known limits

## No External Dependencies

Uses only Python standard library (`urllib`, `json`, `argparse`, etc.).
