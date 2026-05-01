#!/usr/bin/env python3
"""Configuration tool for opencode.json format.

Connects to LiteLLM or NVIDIA NIM providers and outputs models configuration
in the opencode.json format.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


# Provider defaults
DEFAULTS = {
    "litellm": {
        "url": "http://localhost:4000",
    },
    "nvidia_nim": {
        "url": "https://integrate.api.nvidia.com",
    },
}

# Default token limits for models without specific info
DEFAULT_LIMITS = {"context": 128000, "output": 4096}


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


def get_provider_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Extract provider-specific configuration from config dict.

    Supports both nested and flat config formats:
    - Nested: {"providers": {"litellm": {"api_key": "..."}}}
    - Flat: {"api_key": "...", "url": "..."} (backward compatible)
    """
    result = {}

    # Try nested format first: providers.{provider_name}
    providers = config.get("providers", {})
    if provider in providers:
        provider_config = providers[provider]
        if isinstance(provider_config, dict):
            result.update(provider_config)

    # Fall back to flat format for backward compatibility
    # (only for litellm provider to maintain existing behavior)
    if provider == "litellm" and not result:
        for key in ["api_key", "url"]:
            if key in config:
                result[key] = config[key]

    return result


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


def fetch_models_from_nim(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    """Fetch models from NVIDIA NIM API.

    NVIDIA NIM uses OpenAI-compatible /v1/models endpoint.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # NVIDIA NIM uses /v1/models endpoint (OpenAI compatible)
    models_data = make_request(f"{base_url}/v1/models", headers)
    if models_data and "data" in models_data:
        return models_data["data"]

    raise ConnectionError("Failed to fetch models from NVIDIA NIM")


def extract_model_name_from_id(model_id: str) -> str:
    """Extract a display name from model ID."""
    # Handle provider/model format (e.g., "openai/gpt-4")
    if "/" in model_id:
        parts = model_id.split("/")
        return parts[-1]
    return model_id


def build_models_config_litellm(
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

        # Fall back to default limits
        if limits is None:
            limits = DEFAULT_LIMITS.copy()

        # Add limits if we found them
        if limits:
            model_entry["limit"] = limits

        models_config[model_name] = model_entry

    return models_config


def build_models_config_nim(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build opencode.json models section from NVIDIA NIM model data."""
    models_config = {}

    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        # Build the model entry
        model_entry = {
            "name": extract_model_name_from_id(model_id),
        }

        # Use default limits for all NIM models
        model_entry["limit"] = DEFAULT_LIMITS.copy()

        models_config[model_id] = model_entry

    return models_config


def resolve_config(
    provider: str,
    cli_api_key: Optional[str],
    cli_url: Optional[str],
    file_config: Dict[str, Any],
) -> tuple[str, str]:
    """Resolve configuration values using priority order.

    Priority (highest to lowest):
    1. CLI arguments
    2. Config file (nested or flat)
    3. Environment variables
    4. Default values
    """
    # Get provider-specific config from file
    provider_config = get_provider_config(file_config, provider)

    # Environment variable names based on provider
    env_api_key_vars = {
        "litellm": "LITELLM_API_KEY",
        "nvidia_nim": ["NIM_API_KEY", "NVIDIA_NIM_API_KEY", "NV_API_KEY"],
    }
    env_url_vars = {
        "litellm": "LITELLM_URL",
        "nvidia_nim": ["NIM_URL", "NVIDIA_NIM_URL", "NV_URL"],
    }

    # Resolve API key
    api_key = cli_api_key  # CLI arg (highest priority)
    if not api_key:
        api_key = provider_config.get("api_key")  # Config file
    if not api_key:
        # Try environment variables
        env_vars = env_api_key_vars.get(provider, [f"{provider.upper()}_API_KEY"])
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        for var in env_vars:
            api_key = os.environ.get(var)
            if api_key:
                break

    # Resolve URL
    url = cli_url  # CLI arg (highest priority)
    if not url:
        url = provider_config.get("url")  # Config file
    if not url:
        # Try environment variables
        env_vars = env_url_vars.get(provider, [f"{provider.upper()}_URL"])
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        for var in env_vars:
            url = os.environ.get(var)
            if url:
                break
    if not url:
        url = DEFAULTS.get(provider, {}).get("url", "")  # Default

    # Remove trailing slash if present
    if url:
        url = url.rstrip("/")

    return api_key, url


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch models from LLM providers and output opencode.json format"
    )
    parser.add_argument(
        "-m",
        "--models",
        action="store_true",
        required=True,
        help="Fetch and output models configuration",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["litellm", "nvidia_nim"],
        default="litellm",
        help="Provider to fetch models from (default: litellm)",
    )
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        default=None,
        help="API key for the provider (highest priority)",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=None,
        help="Provider base URL (highest priority)",
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

    # Resolve configuration
    api_key, base_url = resolve_config(
        args.provider, args.api_key, args.url, file_config
    )

    if not api_key:
        provider_display = args.provider.replace("_", " ").title()
        print(
            f"Error: API key required for {provider_display}. Use -a/--api-key, config file, or set environment variable",
            file=sys.stderr,
        )
        return 1

    if not base_url:
        print(f"Error: URL required for {args.provider}", file=sys.stderr)
        return 1

    try:
        if args.provider == "litellm":
            # Fetch models from LiteLLM
            models, model_info = fetch_models_from_litellm(base_url, api_key)

            if not models:
                print("Error: No models found in LiteLLM response", file=sys.stderr)
                return 1

            # Build opencode.json models section
            models_config = build_models_config_litellm(models, model_info)

        elif args.provider == "nvidia_nim":
            # Fetch models from NVIDIA NIM
            models = fetch_models_from_nim(base_url, api_key)

            if not models:
                print("Error: No models found in NVIDIA NIM response", file=sys.stderr)
                return 1

            # Build opencode.json models section
            models_config = build_models_config_nim(models)

        else:
            print(f"Error: Unknown provider '{args.provider}'", file=sys.stderr)
            return 1

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
