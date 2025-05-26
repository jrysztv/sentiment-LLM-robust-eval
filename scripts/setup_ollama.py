#!/usr/bin/env python3
"""
Ollama setup and model management script.

This script helps set up Ollama and download the required models
for the Deep Learning Final Assignment.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_learning_final_assignment.core.config import load_experiment_config


def check_ollama_installation():
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama command failed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed or not in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def check_ollama_server(host="http://localhost:11434"):
    """Check if Ollama server is running."""
    try:
        import ollama

        client = ollama.Client(host=host)
        models = client.list()
        print(f"✅ Ollama server is running at {host}")
        print(f"📋 Available models: {len(models.get('models', []))}")
        return True, client
    except Exception as e:
        print(f"❌ Ollama server is not accessible at {host}: {e}")
        return False, None


def start_ollama_server():
    """Attempt to start Ollama server."""
    print("🔄 Attempting to start Ollama server...")
    try:
        # Try to start Ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait a bit for server to start
        time.sleep(3)

        # Check if it's running now
        is_running, client = check_ollama_server()
        if is_running:
            print("✅ Ollama server started successfully")
            return True, client
        else:
            print("❌ Failed to start Ollama server")
            return False, None

    except Exception as e:
        print(f"❌ Error starting Ollama server: {e}")
        return False, None


def list_available_models(client):
    """List all available models."""
    try:
        models = client.list()
        model_list = models.get("models", [])

        if not model_list:
            print("📋 No models are currently downloaded")
            return []

        print(f"📋 Available models ({len(model_list)}):")
        for model in model_list:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3) if size > 0 else 0
            print(f"  - {name} ({size_gb:.1f} GB)")

        return [model.get("name") for model in model_list]

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []


def check_model_availability(client, model_name):
    """Check if a specific model is available."""
    available_models = list_available_models(client)

    # Check exact match
    if model_name in available_models:
        print(f"✅ Model '{model_name}' is available")
        return True

    # Check partial match (for version tags)
    model_base = model_name.split(":")[0]
    for available_model in available_models:
        if available_model.startswith(model_base):
            print(
                f"✅ Similar model found: '{available_model}' (requested: '{model_name}')"
            )
            return True

    print(f"❌ Model '{model_name}' is not available")
    return False


def pull_model(client, model_name):
    """Download a model."""
    print(f"🔄 Downloading model '{model_name}'...")
    print("⚠️  This may take several minutes depending on model size and internet speed")

    try:
        # Use the client to pull the model
        client.pull(model_name)
        print(f"✅ Model '{model_name}' downloaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to download model '{model_name}': {e}")
        return False


def get_model_recommendations():
    """Get model recommendations based on system capabilities."""
    print("\n🤖 Model Recommendations:")
    print("For this assignment, we recommend:")
    print("  - qwen3:4b (Latest Qwen3 model, good performance)")
    print("  - qwen2.5:7b (Previous generation, stable)")
    print("  - llama3.2:3b (Alternative option)")
    print("\nNote: qwen3:4b is the latest available Qwen3 model.")
    print("This is the recommended model for the assignment.")


def main():
    """Main setup function."""
    print("🚀 Ollama Setup for Deep Learning Final Assignment")
    print("=" * 60)

    # Load configuration to get target model
    try:
        config_loader = load_experiment_config()
        ollama_config = config_loader.get_ollama_config()
        target_model = ollama_config.model_name
        auto_pull = ollama_config.auto_pull
        host = ollama_config.host
        print(f"🎯 Target model from config: {target_model}")
        print(f"🔧 Ollama host: {host}")
        print(f"🔄 Auto-pull enabled: {auto_pull}")
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        target_model = "qwen3:4b"  # Default
        auto_pull = True
        host = "http://localhost:11434"
        print(f"🎯 Using default target model: {target_model}")
        print(f"🔧 Using default host: {host}")

    print()

    # Step 1: Check Ollama installation
    print("1️⃣ Checking Ollama installation...")
    if not check_ollama_installation():
        print("\n❌ Ollama is not installed!")
        print("Please install Ollama from: https://ollama.ai/")
        print("Installation instructions:")
        print("  - macOS: brew install ollama")
        print("  - Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  - Windows: Download from https://ollama.ai/download")
        return 1

    print()

    # Step 2: Check/start Ollama server
    print("2️⃣ Checking Ollama server...")
    is_running, client = check_ollama_server(host)

    if not is_running:
        is_running, client = start_ollama_server()

    if not is_running:
        print("\n❌ Could not start Ollama server!")
        print("Please manually start Ollama:")
        print("  Run: ollama serve")
        print("  Then run this script again")
        return 1

    print()

    # Step 3: Check available models
    print("3️⃣ Checking available models...")
    available_models = list_available_models(client)

    print()

    # Step 4: Check target model
    print(f"4️⃣ Checking target model: {target_model}")
    model_available = check_model_availability(client, target_model)

    if not model_available:
        print(f"\n🔄 Model '{target_model}' needs to be downloaded")

        if auto_pull:
            print(
                f"🔄 Auto-pull is enabled, downloading '{target_model}' automatically..."
            )
            success = pull_model(client, target_model)
            if not success:
                print(f"\n❌ Failed to download '{target_model}'")
                get_model_recommendations()
                return 1
        else:
            response = input(f"Download '{target_model}' now? (y/n): ").lower().strip()

            if response in ["y", "yes"]:
                success = pull_model(client, target_model)
                if not success:
                    print(f"\n❌ Failed to download '{target_model}'")
                    get_model_recommendations()
                    return 1
            else:
                print("⚠️  Model not downloaded. You can download it later with:")
                print(f"  ollama pull {target_model}")
                get_model_recommendations()
                return 1

    print()

    # Step 5: Final verification
    print("5️⃣ Final verification...")
    try:
        from deep_learning_final_assignment.core.models.ollama_model import OllamaModel

        # Use configuration for model initialization
        model = OllamaModel(target_model, host=host)
        if model.is_available:
            print(f"✅ Model '{target_model}' is ready for use!")

            # Test generation
            print("🧪 Testing model generation...")
            test_response = model.generate("Hello, how are you?", max_tokens=10)
            print(f"✅ Test successful: {test_response.content[:50]}...")

        else:
            print(f"❌ Model '{target_model}' is not available through our interface")
            return 1

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return 1

    print()
    print("🎉 Ollama setup complete!")
    print("You can now run the baseline experiment:")
    print("  python scripts/run_baseline.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
