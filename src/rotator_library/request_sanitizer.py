# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any


def _supports_dimensions(model: str) -> bool:
    """Models that accept an OpenAI-style `dimensions` embedding parameter."""
    if model.startswith("openai/text-embedding-3"):
        return True
    # Gemini Code Assist :embedContent maps dimensions -> outputDimensionality.
    if model.startswith("gemini_cli/") and "embedding" in model.lower():
        return True
    return False


def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and not _supports_dimensions(model):
        del payload["dimensions"]
        
    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if model not in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]:
            del payload["thinking"]
            
    return payload
