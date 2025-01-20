"""
Ollama API client for local LLM model support.
"""
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Union, Any

import requests

from ...config import config

logger = logging.getLogger(__name__)

# Model capabilities
MODEL_CAPABILITIES = {
    "llama3.3:70b": {
        "max_tokens": 4096,
        "supports_streaming": True,
        "supports_embeddings": True
    },
    "llama3.1:8b": {
        "max_tokens": 4096,
        "supports_streaming": True,
        "supports_embeddings": True
    },
    "llama3.1:70b": {
        "max_tokens": 4096,
        "supports_streaming": True,
        "supports_embeddings": True
    },
    "llama3.1:405b": {
        "max_tokens": 4096,
        "supports_streaming": True,
        "supports_embeddings": True
    },
    "phi4:14b": {
        "max_tokens": 2048,
        "supports_streaming": True,
        "supports_embeddings": False
    },
    "qwq:32b": {
        "max_tokens": 4096,
        "supports_streaming": True,
        "supports_embeddings": False
    }
}

# Supported Ollama models
SUPPORTED_OLLAMA_MODELS = {
    "llama3.3:70b",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "phi4:14b",
    "qwq:32b"
}

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: Optional[str] = None):
        """Initialize Ollama client with optional host override."""
        self.host = host or config.ollama_host
        
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and accessible."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama not accessible: {str(e)}")
            return False
            
    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get capabilities for a specific model."""
        return MODEL_CAPABILITIES.get(model, {
            "max_tokens": 2048,  # Conservative default
            "supports_streaming": False,
            "supports_embeddings": False
        })

    def generate(self, 
                prompt: str, 
                model: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs: Any) -> str:
        """
        Generate a response using the Ollama API.
        
        Args:
            prompt: The prompt string to send to the model
            model: Optional model override, defaults to config.model_name
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If Ollama is not installed or accessible
        """
        if not self._check_ollama_installed():
            raise RuntimeError(
                "Ollama is not installed or accessible. "
                "Please install Ollama first: https://ollama.ai"
            )
            
        model = model or config.model_name
        
        # Validate model and get capabilities
        if model not in SUPPORTED_OLLAMA_MODELS:
            logger.warning(f"Unsupported Ollama model: {model}")
            model = "llama3.1:8b"  # Fallback to stable model
            
        capabilities = self.get_model_capabilities(model)
        if max_tokens is None or max_tokens > capabilities["max_tokens"]:
            max_tokens = capabilities["max_tokens"]
            
        try:
            # Rate limiting: exponential backoff
            max_retries = 3
            retry_delay = 1.0
            
            response = None
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "temperature": max(0.0, min(1.0, temperature)),
                            "max_tokens": max_tokens,
                            **kwargs
                        },
                        timeout=30.0  # Add timeout
                    )
                    response.raise_for_status()
                    return response.json()["response"]
                    
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    retry_delay *= 2
                    logger.warning(f"Retrying in {retry_delay}s... ({str(e)})")
                    time.sleep(retry_delay)
            
            # If we get here, all retries failed
            return "MAX RETRIES EXCEEDED"
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Ollama API error ({error_type}): {error_msg}")
            
            if isinstance(e, requests.exceptions.Timeout):
                return "REQUEST TIMEOUT"
            elif isinstance(e, requests.exceptions.ConnectionError):
                return "CONNECTION ERROR"
            elif isinstance(e, requests.exceptions.HTTPError) and response:
                return f"HTTP ERROR {response.status_code}"
            
            return "API ERROR"
            
    def chat(self,
            messages: list[Dict[str, str]],
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000) -> str:
        """
        Chat completion using the Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override, defaults to config.model_name
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Convert chat messages to a single prompt for now
        # TODO: Update when Ollama adds native chat support
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.generate(prompt, model, temperature, max_tokens)

# Global client instance
client = OllamaClient()
