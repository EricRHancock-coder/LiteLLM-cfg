# litellm-cfg

A Python script to fetch models from LLM providers (LiteLLM proxy or NVIDIA NIM) and output them in opencode.json format.

## Supported Providers

- **LiteLLM** - LiteLLM proxy server
- **NVIDIA NIM** - NVIDIA's managed inference API

## Installation

```bash
chmod +x litellm-cfg.py
```

## Configuration Priority

Configuration values are resolved in this priority order (highest to lowest):

1. **CLI arguments** (`-a`, `-u`)
2. **Config file** (`.litellm-cfg.json` or `~/.config/litellm-cfg/config.json`)
3. **Environment variables** (provider-specific, see below)
4. **Default values** (see provider sections below)

## Configuration File

Create a config file at one of these locations:

- `./.litellm-cfg.json` (current directory)
- `~/.config/litellm-cfg/config.json` (user config)
- Custom path via `LITELLM_CFG_CONFIG` env var or `-c` flag

### Nested Provider Config (Recommended)

```json
{
  "providers": {
    "litellm": {
      "api_key": "sk-your-litellm-api-key",
      "url": "http://litellm.example.com:4000"
    },
    "nvidia_nim": {
      "api_key": "nvapi-your-nvidia-api-key",
      "url": "https://integrate.api.nvidia.com"
    }
  }
}
```

### Legacy Flat Config (Still Supported for LiteLLM)

```json
{
  "api_key": "sk-your-litellm-api-key",
  "url": "http://litellm.example.com:4000"
}
```

## Usage Examples

### LiteLLM Provider (Default)

#### Using Config File

```bash
# With config file at .litellm-cfg.json
python3 litellm-cfg.py -m

# Or explicitly specify provider
python3 litellm-cfg.py -m -p litellm
```

#### Using CLI Arguments (overrides config)

```bash
python3 litellm-cfg.py -m -p litellm -a "sk-xxx" -u "http://localhost:4000"
```

#### Using Environment Variables

```bash
export LITELLM_API_KEY="sk-xxx"
export LITELLM_URL="http://localhost:4000"
python3 litellm-cfg.py -m
```

### NVIDIA NIM Provider

#### Using Config File

```bash
python3 litellm-cfg.py -m -p nvidia_nim
```

#### Using CLI Arguments

```bash
python3 litellm-cfg.py -m -p nvidia_nim -a "nvapi-xxx" -u "https://integrate.api.nvidia.com"
```

#### Using Environment Variables

```bash
export NIM_API_KEY="nvapi-xxx"
# NIM_URL is optional (defaults to https://integrate.api.nvidia.com)
python3 litellm-cfg.py -m -p nvidia_nim
```

Supported environment variables for NVIDIA NIM:
- `NIM_API_KEY`, `NVIDIA_NIM_API_KEY`, or `NV_API_KEY`
- `NIM_URL`, `NVIDIA_NIM_URL`, or `NV_URL`

### Custom Config File Path

```bash
python3 litellm-cfg.py -m -p nvidia_nim -c /path/to/my-config.json
```

### Save Output to File

```bash
python3 litellm-cfg.py -m -p nvidia_nim > nim-models.json
```

### Use in opencode.json

```bash
# Create complete opencode.json with NVIDIA NIM models
cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nvidia_nim": {
      "options": {
        "baseURL": "https://integrate.api.nvidia.com/v1",
        "apiKey": "{env:NIM_API_KEY}"
      },
      "models":
EOF

python3 litellm-cfg.py -m -p nvidia_nim >> opencode.json

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

## Provider-Specific Details

### LiteLLM

- **Default URL**: `http://localhost:4000`
- **Environment Variables**: `LITELLM_API_KEY`, `LITELLM_URL`
- **Token Limits**: The script tries to get token limits in this order:
  1. **Admin endpoint** (`/v1/model/info`) - requires admin key
  2. **Default limits** - 128k context, 4k output

### NVIDIA NIM

- **Default URL**: `https://integrate.api.nvidia.com`
- **Environment Variables**: `NIM_API_KEY`, `NIM_URL`
- **Token Limits**: Uses default limits (128k context, 4k output) for all models
- **API Key**: Get your API key at [build.nvidia.com](https://build.nvidia.com/settings/api-keys)

## No External Dependencies

Uses only Python standard library (`urllib`, `json`, `argparse`, etc.).
