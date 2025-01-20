"""
Model configuration verification script.
Tests model switching and capability detection.
"""
import os
from typing import Dict, Any

from reverie.backend_server.config import config, ModelType
from reverie.backend_server.persona.prompt_template.ollama_client import client as ollama_client

def test_model_config(model_type: str) -> Dict[str, Any]:
    """Test configuration for a specific model type."""
    os.environ["MODEL_TYPE"] = model_type
    return {
        "model_type": model_type,
        "model_name": config.model_name,
        "is_ollama": config.is_ollama,
        "capabilities": ollama_client.get_model_capabilities(config.model_name) if config.is_ollama else None
    }

def main():
    """Test all supported model configurations."""
    print("Testing Model Configurations\n")
    
    # Test OpenAI Models
    openai_models = [
        "openai_gpt4o",
        "openai_gpt4o_mini",
        "openai_o1",
        "openai_o1_mini"
    ]
    
    print("=== OpenAI Models ===")
    for model in openai_models:
        result = test_model_config(model)
        print(f"\nModel Type: {result['model_type']}")
        print(f"Model Name: {result['model_name']}")
        print(f"Is Ollama: {result['is_ollama']}")
    
    # Test Ollama Models
    ollama_models = [
        "ollama_llama33_70b",
        "ollama_llama31_8b",
        "ollama_llama31_70b",
        "ollama_llama31_405b",
        "ollama_phi4",
        "ollama_qwq"
    ]
    
    print("\n=== Ollama Models ===")
    for model in ollama_models:
        result = test_model_config(model)
        print(f"\nModel Type: {result['model_type']}")
        print(f"Model Name: {result['model_name']}")
        print(f"Is Ollama: {result['is_ollama']}")
        if result['capabilities']:
            print("Capabilities:")
            print(f"  Max Tokens: {result['capabilities']['max_tokens']}")
            print(f"  Streaming: {result['capabilities']['supports_streaming']}")
            print(f"  Embeddings: {result['capabilities']['supports_embeddings']}")

if __name__ == "__main__":
    main()
