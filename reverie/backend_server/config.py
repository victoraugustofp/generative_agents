"""
Configuration module for model selection and API settings.
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union, Any

class ModelType(str, Enum):
    """Supported model types for the application."""
    # OpenAI Models
    OPENAI_GPT4O = "openai_gpt4o"
    OPENAI_GPT4O_MINI = "openai_gpt4o_mini"
    OPENAI_O1 = "openai_o1"
    OPENAI_O1_MINI = "openai_o1_mini"
    
    # Ollama Models
    OLLAMA_LLAMA33_70B = "ollama_llama33_70b"
    OLLAMA_LLAMA31_8B = "ollama_llama31_8b"
    OLLAMA_LLAMA31_70B = "ollama_llama31_70b"
    OLLAMA_LLAMA31_405B = "ollama_llama31_405b"
    OLLAMA_PHI4 = "ollama_phi4"
    OLLAMA_QWQ = "ollama_qwq"

@dataclass
class ModelCapabilities:
    """Model capabilities and constraints."""
    max_tokens: int
    context_window: int
    supports_vision: bool = False
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_embeddings: bool = False
    embedding_model: str = "text-embedding-ada-002"

class ModelConfig:
    def __init__(self):
        model_type = os.getenv("MODEL_TYPE", ModelType.OPENAI_GPT4O.value)
        try:
            self._model_type = ModelType(model_type)
        except ValueError:
            self._model_type = ModelType.OPENAI_GPT4O
            
        self._api_key: str = os.getenv("OPENAI_API_KEY", "")
        self._ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Model-specific parameters
        self._temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        max_tokens_str = os.getenv("MODEL_MAX_TOKENS")
        self._max_tokens: Optional[int] = (
            int(max_tokens_str) if max_tokens_str is not None else None
        )
        
    @property
    def model_type(self) -> ModelType:
        return self._model_type
        
    @property
    def model_name(self) -> str:
        """Get the actual model name based on the model type."""
        model_names = {
            # OpenAI Models
            ModelType.OPENAI_GPT4O: "gpt-4o-2024-11-20",
            ModelType.OPENAI_GPT4O_MINI: "gpt-4o-mini-2024-11-20",
            ModelType.OPENAI_O1: "o1-2024-11-20",
            ModelType.OPENAI_O1_MINI: "o1-mini-2024-11-20",
            
            # Ollama Models
            ModelType.OLLAMA_LLAMA33_70B: "llama3.3:70b",
            ModelType.OLLAMA_LLAMA31_8B: "llama3.1:8b",
            ModelType.OLLAMA_LLAMA31_70B: "llama3.1:70b",
            ModelType.OLLAMA_LLAMA31_405B: "llama3.1:405b",
            ModelType.OLLAMA_PHI4: "phi4:14b",
            ModelType.OLLAMA_QWQ: "qwq:32b"
        }
        return model_names[self.model_type]
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get capabilities for the current model."""
        caps = {
            # OpenAI Models
            ModelType.OPENAI_GPT4O: ModelCapabilities(
                max_tokens=16384,
                context_window=128000,
                supports_vision=True,
                supports_streaming=True,
                supports_function_calling=True,
                supports_embeddings=True,
                embedding_model="text-embedding-3-large"
            ),
            ModelType.OPENAI_GPT4O_MINI: ModelCapabilities(
                max_tokens=4096,
                context_window=128000,
                supports_vision=True,
                supports_streaming=True,
                supports_function_calling=True,
                supports_embeddings=True,
                embedding_model="text-embedding-3-small"
            ),
            ModelType.OPENAI_O1: ModelCapabilities(
                max_tokens=32768,
                context_window=128000,
                supports_vision=True,
                supports_streaming=True,
                supports_function_calling=True,
                supports_embeddings=True,
                embedding_model="text-embedding-3-large"
            ),
            ModelType.OPENAI_O1_MINI: ModelCapabilities(
                max_tokens=4096,
                context_window=32768,
                supports_vision=False,
                supports_streaming=True,
                supports_function_calling=True,
                supports_embeddings=True,
                embedding_model="text-embedding-3-small"
            ),
            
            # Ollama Models
            ModelType.OLLAMA_LLAMA33_70B: ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_streaming=True,
                supports_embeddings=True,
                embedding_model="llama3.3:70b"
            ),
            ModelType.OLLAMA_LLAMA31_8B: ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_streaming=True,
                supports_embeddings=True,
                embedding_model="llama3.1:8b"
            ),
            ModelType.OLLAMA_LLAMA31_70B: ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_streaming=True,
                supports_embeddings=True,
                embedding_model="llama3.1:70b"
            ),
            ModelType.OLLAMA_LLAMA31_405B: ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_streaming=True,
                supports_embeddings=True,
                embedding_model="llama3.1:405b"
            ),
            ModelType.OLLAMA_PHI4: ModelCapabilities(
                max_tokens=2048,
                context_window=2048,
                supports_streaming=True,
                supports_embeddings=False
            ),
            ModelType.OLLAMA_QWQ: ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_streaming=True,
                supports_embeddings=False
            )
        }
        return caps[self.model_type]
    
    @property
    def is_ollama(self) -> bool:
        """Check if current model is an Ollama model."""
        return self.model_type.name.startswith("OLLAMA_")
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    @property
    def ollama_host(self) -> str:
        return self._ollama_host
        
    @property
    def temperature(self) -> float:
        return self._temperature
        
    @property
    def max_tokens(self) -> int:
        """Get max tokens, falling back to model capability if not set."""
        return self._max_tokens or self.capabilities.max_tokens

# Global instance
config = ModelConfig()
