"""
Minimal model configuration verification script.
"""
import os
from reverie.backend_server.config import config

def test_model(model_type: str) -> None:
    """Test basic model configuration."""
    os.environ["MODEL_TYPE"] = model_type
    print(f"\nTesting {model_type}:")
    print(f"Model name: {config.model_name}")
    print(f"Is Ollama: {config.is_ollama}")

def main():
    """Test core model configurations."""
    print("=== Testing Model Configuration ===")
    
    # Test OpenAI model
    test_model("openai_gpt4o")
    
    # Test Ollama model
    test_model("ollama_llama31_8b")

if __name__ == "__main__":
    main()
