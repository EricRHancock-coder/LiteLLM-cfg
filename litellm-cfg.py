#!/usr/bin/env python3
"""LiteLLM configuration tool for opencode.json format.

Connects to a LiteLLM proxy instance and outputs models configuration
in the opencode.json format.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


# LiteLLM model cost map for known model token limits
# Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
LITELLM_COST_MAP = {
    "gpt-4": {"context": 8192, "output": 4096},
    "gpt-4-32k": {"context": 32768, "output": 4096},
    "gpt-4-turbo": {"context": 128000, "output": 4096},
    "gpt-4o": {"context": 128000, "output": 4096},
    "gpt-4o-mini": {"context": 128000, "output": 16384},
    "gpt-3.5-turbo": {"context": 16385, "output": 4096},
    "gpt-3.5-turbo-16k": {"context": 16385, "output": 4096},
    "claude-3-opus": {"context": 200000, "output": 4096},
    "claude-3-sonnet": {"context": 200000, "output": 4096},
    "claude-3-haiku": {"context": 200000, "output": 4096},
    "claude-3-5-sonnet": {"context": 200000, "output": 8192},
    "claude-3-5-haiku": {"context": 200000, "output": 4096},
    "gemini-pro": {"context": 32760, "output": 8192},
    "gemini-1.5-pro": {"context": 2000000, "output": 8192},
    "gemini-1.5-flash": {"context": 1000000, "output": 8192},
}


def get_config_file_path() -> str:
    """Get the path to the config file.
    
    Priority:
    1. LITELLM_CFG_CONFIG env var
    2. ./.litellm-cfg.json (current directory)
    3. ~/.config/litellm-cfg/config.json
    """
    # Check env var first
    env_path = os.environ.get("LITELLM_CFG_CONFIG")
    if env_path:
        return env_path
    
    # Check current directory
    local_config = ".litellm-cfg.json"
    if os.path.exists(local_config):
        return local_config
    
    # Check user config directory
    home = os.path.expanduser("~")
    user_config = os.path.join(home, ".config", "litellm-cfg", "config.json")
    if os.path.exists(user_config):
        return user_config
    
    return ""


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse config file {config_path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}", file=sys.stderr)
        return {}


def get_token_limits_from_cost_map(model_id: str) -> Optional[Dict[str, int]]:
    """Try to get token limits from LiteLLM cost map based on model ID patterns."""
    model_lower = model_id.lower()

    # Check for exact matches first
    if model_id in LITELLM_COST_MAP:
        return LITELLM_COST_MAP[model_id]

    # Check for pattern matches
    for pattern, limits in LITELLM_COST_MAP.items():
        if pattern in model_lower:
            return limits

    return None


def make_request(url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Make an HTTP GET request and return JSON response."""
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ConnectionError(f"Authentication failed: {e.reason}")
        elif e.code == 403:
            return None  # Admin endpoint not accessible, that's ok
        raise ConnectionError(f"HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect: {e.reason}")
    except Exception as e:
        raise ConnectionError(f"Request failed: {e}")


def fetch_models_from_litellm(
    base_url: str, api_key: str
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Fetch models from LiteLLM proxy.

    Tries /v1/model/info first for detailed info including token limits,
    falls back to /v1/models.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try /v1/model/info first (requires admin access, has token limits)
    model_info = None
    try:
        info_data = make_request(f"{base_url}/v1/model/info", headers)
        if info_data and "data" in info_data:
            model_info = info_data["data"]
    except ConnectionError:
        pass

    # Always get basic models from /v1/models
    models_data = make_request(f"{base_url}/v1/models", headers)
    if models_data and "data" in models_data:
        return models_data["data"], model_info

    raise ConnectionError("Failed to fetch models from LiteLLM")


def extract_model_name_from_id(model_id: str) -> str:
    """Extract a display name from model ID."""
    # Handle provider/model format (e.g., "openai/gpt-4")
    if "/" in model_id:
        parts = model_id.split("/")
        return parts[-1]
    return model_id


def build_models_config(
    models: List[Dict[str, Any]], model_info_list: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Build opencode.json models section from LiteLLM model data."""
    models_config = {}

    # Build a lookup from model info if available
    info_lookup = {}
    if model_info_list:
        for info in model_info_list:
            model_name = info.get("model_name", "")
            info_lookup[model_name] = info.get("model_info", {})

    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        # Use model_name from info if available, otherwise extract from id
        model_name = model_id
        if model_info_list:
            # Try to find matching model_info
            for info in model_info_list:
                if info.get("model_name") == model_id:
                    model_name = info.get("model_name", model_id)
                    break

        # Build the model entry
        model_entry = {
            "name": extract_model_name_from_id(model_id),
        }

        # Try to get token limits
        limits = None

        # First try from model_info (if we have admin access data)
        if model_name in info_lookup:
            info = info_lookup[model_name]
            max_tokens = info.get("max_tokens")
            max_input = info.get("max_input_tokens")
            max_output = info.get("max_output_tokens")

            if max_input or max_tokens:
                limits = {
                    "context": max_input or max_tokens or 4096,
                    "output": max_output or 4096,
                }

        # Fall back to cost map lookup
        if limits is None:
            limits = get_token_limits_from_cost_map(model_id)

        # Add limits if we found them
        if limits:
            model_entry["limit"] = limits

        models_config[model_name] = model_entry

    return models_config


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch models from LiteLLM and output opencode.json format"
    )
    parser.add_argument(
        "-m",
        "--models",
        action="store_true",
        required=True,
        help="Fetch and output models configuration",
    )
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        default=None,
        help="API key for LiteLLM (highest priority)",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=None,
        help="LiteLLM base URL (highest priority)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: .litellm-cfg.json or ~/.config/litellm-cfg/config.json)",
    )

    args = parser.parse_args()

    # Load config file (if exists)
    config_file_path = args.config or get_config_file_path()
    file_config = load_config_file(config_file_path) if config_file_path else {}

    # Priority order (highest to lowest):
    # 1. CLI args
    # 2. Config file
    # 3. Environment variables
    # 4. Default values
    
    # Get API key
    api_key = args.api_key  # CLI arg (highest)
    if not api_key:
        api_key = file_config.get("api_key")  # Config file
    if not api_key:
        api_key = os.environ.get("LITELLM_API_KEY")  # Environment
    
    if not api_key:
        print("Error: API key required. Use -a/--api-key, config file, or set LITELLM_API_KEY env var", file=sys.stderr)
        return 1

    # Get base URL
    base_url = args.url  # CLI arg (highest)
    if not base_url:
        base_url = file_config.get("url")  # Config file
    if not base_url:
        base_url = os.environ.get("LITELLM_URL", "http://localhost:4000")  # Environment or default
    
    # Remove trailing slash if present
    base_url = base_url.rstrip("/")

    try:
        # Fetch models from LiteLLM
        models, model_info = fetch_models_from_litellm(base_url, api_key)

        if not models:
            print("Error: No models found in LiteLLM response", file=sys.stderr)
            return 1

        # Build opencode.json models section
        models_config = build_models_config(models, model_info)

        # Output as JSON to stdout
        print(json.dumps(models_config, indent=2))
        return 0

    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
