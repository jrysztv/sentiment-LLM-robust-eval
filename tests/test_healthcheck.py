"""
Healthcheck tests for model connectivity.
"""

import pytest
import asyncio
import os
from unittest.mock import patch

from deep_learning_final_assignment.core.models import (
    OpenAIModel,
    OllamaModel,
    AsyncOpenAIModel,
    AsyncOllamaModel,
)


class TestModelHealthcheck:
    """Test connectivity to model services."""

    def test_openai_healthcheck(self):
        """Test OpenAI model connectivity."""
        # Skip if no API key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        try:
            model = OpenAIModel("gpt-4-turbo")
            is_available = model.is_available
            print(f"OpenAI model availability: {is_available}")

            if is_available:
                # Try a simple generation
                response = model.generate("Say 'hello' in JSON format", max_tokens=10)
                print(f"OpenAI test response: {response.content[:100]}...")
                assert response.content is not None
                print("✅ OpenAI healthcheck passed")
            else:
                print("❌ OpenAI model not available")

        except Exception as e:
            print(f"❌ OpenAI healthcheck failed: {e}")
            pytest.fail(f"OpenAI healthcheck failed: {e}")

    def test_ollama_healthcheck(self):
        """Test Ollama model connectivity."""
        try:
            model = OllamaModel("qwen3:4b")
            is_available = model.is_available
            print(f"Ollama model availability: {is_available}")

            if is_available:
                # Try a simple generation
                response = model.generate("Say 'hello' in JSON format", max_tokens=10)
                print(f"Ollama test response: {response.content[:100]}...")
                assert response.content is not None
                print("✅ Ollama healthcheck passed")
            else:
                print("❌ Ollama model not available - is Ollama running?")
                print("To start Ollama: ollama serve")
                print("To pull model: ollama pull qwen3:4b")

        except Exception as e:
            print(f"❌ Ollama healthcheck failed: {e}")
            print("Make sure Ollama is running and the model is available")
            # Don't fail the test for Ollama since it's optional
            pytest.skip(f"Ollama not available: {e}")

    @pytest.mark.asyncio
    async def test_async_openai_healthcheck(self):
        """Test async OpenAI model connectivity."""
        # Skip if no API key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        model = None
        try:
            model = AsyncOpenAIModel("gpt-4-turbo")
            is_available = model.is_available
            print(f"Async OpenAI model availability: {is_available}")

            if is_available:
                # Try a simple generation
                response = await model.generate(
                    "Say 'hello' in JSON format", max_tokens=10
                )
                print(f"Async OpenAI test response: {response.content[:100]}...")
                assert response.content is not None
                print("✅ Async OpenAI healthcheck passed")
            else:
                print("❌ Async OpenAI model not available")

        except Exception as e:
            print(f"❌ Async OpenAI healthcheck failed: {e}")
            pytest.fail(f"Async OpenAI healthcheck failed: {e}")
        finally:
            if model:
                await model.close()

    @pytest.mark.asyncio
    async def test_async_ollama_healthcheck(self):
        """Test async Ollama model connectivity."""
        model = None
        try:
            model = AsyncOllamaModel("qwen3:4b")
            is_available = model.is_available
            print(f"Async Ollama model availability: {is_available}")

            if is_available:
                # Try a simple generation
                response = await model.generate(
                    "Say 'hello' in JSON format", max_tokens=10
                )
                print(f"Async Ollama test response: {response.content[:100]}...")
                assert response.content is not None
                print("✅ Async Ollama healthcheck passed")
            else:
                print("❌ Async Ollama model not available - is Ollama running?")
                print("To start Ollama: ollama serve")
                print("To pull model: ollama pull qwen3:4b")

        except Exception as e:
            print(f"❌ Async Ollama healthcheck failed: {e}")
            print("Make sure Ollama is running and the model is available")
            # Don't fail the test for Ollama since it's optional
            pytest.skip(f"Async Ollama not available: {e}")
        finally:
            if model:
                await model.close()

    def test_service_status_summary(self):
        """Print a summary of all service statuses."""
        print("\n" + "=" * 50)
        print("MODEL SERVICE STATUS SUMMARY")
        print("=" * 50)

        # Test OpenAI
        openai_status = "❌ Not Available"
        if os.getenv("OPENAI_API_KEY"):
            try:
                model = OpenAIModel("gpt-4-turbo")
                if model.is_available:
                    openai_status = "✅ Available"
            except:
                pass
        else:
            openai_status = "⚠️  No API Key"

        # Test Ollama
        ollama_status = "❌ Not Available"
        try:
            model = OllamaModel("qwen3:4b")
            if model.is_available:
                ollama_status = "✅ Available"
        except:
            pass

        print(f"OpenAI (GPT-4):     {openai_status}")
        print(f"Ollama (Qwen3:4b):  {ollama_status}")
        print("=" * 50)

        if "❌" in openai_status and "❌" in ollama_status:
            print("⚠️  WARNING: No models are available!")
            print("   - For OpenAI: Set OPENAI_API_KEY environment variable")
            print("   - For Ollama: Run 'ollama serve' and 'ollama pull qwen3:4b'")
        elif "❌" in ollama_status:
            print("ℹ️  INFO: Only OpenAI is available")
            print(
                "   - To enable Ollama: Run 'ollama serve' and 'ollama pull qwen3:4b'"
            )
        elif "❌" in openai_status:
            print("ℹ️  INFO: Only Ollama is available")
            print("   - To enable OpenAI: Set OPENAI_API_KEY environment variable")
        else:
            print("🎉 SUCCESS: All models are available!")
