"""
Tests for async model interfaces with rate limiting.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from deep_learning_final_assignment.core.models.base import ModelResponse
from deep_learning_final_assignment.core.models.async_openai_model import (
    AsyncOpenAIModel,
)
from deep_learning_final_assignment.core.models.async_ollama_model import (
    AsyncOllamaModel,
)
from deep_learning_final_assignment.core.models.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test the RateLimiter class."""

    def test_rate_limiter_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=30000)

        assert limiter.requests_per_minute == 60
        assert limiter.tokens_per_minute == 30000
        assert limiter.request_interval == 1.0  # 60 requests/min = 1 request/second
        assert len(limiter.request_times) == 0
        assert limiter.token_count == 0

    def test_rate_limiter_no_limits(self):
        """Test RateLimiter with no limits."""
        limiter = RateLimiter()

        assert limiter.requests_per_minute is None
        assert limiter.tokens_per_minute is None
        assert limiter.request_interval == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_request_limiting(self):
        """Test request rate limiting."""
        limiter = RateLimiter(requests_per_minute=120)  # 2 requests per second

        start_time = datetime.now()

        # First request should be immediate
        await limiter.wait_if_needed(estimated_tokens=100)
        first_request_time = datetime.now()

        # Second request should wait ~0.5 seconds
        await limiter.wait_if_needed(estimated_tokens=100)
        second_request_time = datetime.now()

        # Check that we waited approximately the right amount
        time_diff = (second_request_time - first_request_time).total_seconds()
        assert 0.4 <= time_diff <= 0.6  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limiter_token_limiting(self):
        """Test token rate limiting with configurable time window."""
        # Use a 2-second time window for fast testing
        limiter = RateLimiter(
            tokens_per_minute=1000, time_window_seconds=2.0
        )  # Very low limit for testing

        # First request with 500 tokens should be fine
        await limiter.wait_if_needed(estimated_tokens=500)

        # Second request with 600 tokens should wait for token reset
        start_time = datetime.now()
        await limiter.wait_if_needed(estimated_tokens=600)
        end_time = datetime.now()

        # Should have waited for token window to reset (around 2 seconds)
        time_diff = (end_time - start_time).total_seconds()
        assert 1.5 <= time_diff <= 2.5  # Should have waited approximately 2 seconds

    @pytest.mark.asyncio
    async def test_rate_limiter_no_wait_when_no_limits(self):
        """Test that no waiting occurs when no limits are set."""
        limiter = RateLimiter()

        start_time = datetime.now()
        await limiter.wait_if_needed(estimated_tokens=10000)
        await limiter.wait_if_needed(estimated_tokens=10000)
        await limiter.wait_if_needed(estimated_tokens=10000)
        end_time = datetime.now()

        # Should be very fast with no limits
        time_diff = (end_time - start_time).total_seconds()
        assert time_diff < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_custom_time_window(self):
        """Test rate limiter with custom time window for faster testing."""
        # Use a 2-second time window with 2 requests per window = 1 request per second
        limiter = RateLimiter(
            requests_per_minute=60,  # 60 requests per 2-second window = 1 request per second
            time_window_seconds=2.0,
        )

        start_time = datetime.now()

        # First request should be immediate
        await limiter.wait_if_needed(estimated_tokens=100)

        # Second request should wait ~1 second (2.0/60 = 0.033 seconds interval)
        await limiter.wait_if_needed(estimated_tokens=100)

        end_time = datetime.now()
        time_diff = (end_time - start_time).total_seconds()

        # Should have waited approximately the request interval (0.033 seconds)
        assert 0.02 <= time_diff <= 0.1  # Very small wait time


class TestAsyncOpenAIModel:
    """Test the AsyncOpenAIModel class."""

    @pytest.mark.asyncio
    async def test_async_openai_model_init(self, mock_env_vars):
        """Test AsyncOpenAI model initialization."""
        with patch(
            "deep_learning_final_assignment.core.models.async_openai_model.AsyncOpenAI"
        ) as mock_openai:
            model = AsyncOpenAIModel("gpt-4-turbo", api_key="test_key")

            assert model.model_name == "gpt-4-turbo"
            assert model.api_key == "test_key"
            assert isinstance(model.rate_limiter, RateLimiter)
            assert (
                model.rate_limiter.requests_per_minute == 120
            )  # Conservative rate limit for baseline experiment
            assert model.rate_limiter.tokens_per_minute == 25000

    @pytest.mark.asyncio
    async def test_async_openai_model_generate(self, mock_env_vars):
        """Test AsyncOpenAI model generate method."""
        with patch(
            "deep_learning_final_assignment.core.models.async_openai_model.AsyncOpenAI"
        ) as mock_openai_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"sentiment": "Positive"}'
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4-turbo"
            mock_response.usage = Mock()
            mock_response.usage.model_dump.return_value = {"total_tokens": 50}

            mock_client.chat.completions.create.return_value = mock_response

            # Test
            model = AsyncOpenAIModel("gpt-4-turbo")
            model.client = mock_client

            response = await model.generate("Test prompt", temperature=0.2)

            assert isinstance(response, ModelResponse)
            assert response.content == '{"sentiment": "Positive"}'
            assert response.model_name == "gpt-4-turbo"
            assert response.prompt_used == "Test prompt"
            assert response.metadata["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_async_openai_model_batch_generate(self, mock_env_vars):
        """Test AsyncOpenAI model batch_generate method."""
        with patch(
            "deep_learning_final_assignment.core.models.async_openai_model.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            async def mock_create(**kwargs):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = '{"sentiment": "Positive"}'
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "gpt-4-turbo"
                mock_response.usage = Mock()
                mock_response.usage.model_dump.return_value = {"total_tokens": 50}
                return mock_response

            mock_client.chat.completions.create.side_effect = mock_create

            model = AsyncOpenAIModel("gpt-4-turbo")
            model.client = mock_client

            # Mock rate limiter to avoid waiting in tests
            model.rate_limiter = Mock()
            model.rate_limiter.wait_if_needed = AsyncMock()

            # Mock the is_available property by patching the class method
            with patch.object(
                AsyncOpenAIModel,
                "is_available",
                new_callable=lambda: property(lambda self: True),
            ):
                responses = await model.batch_generate(["Prompt 1", "Prompt 2"])

                assert len(responses) == 2
                assert all(isinstance(r, ModelResponse) for r in responses)
                assert all(r.content == '{"sentiment": "Positive"}' for r in responses)

    @pytest.mark.asyncio
    async def test_async_openai_model_rate_limiting_integration(self, mock_env_vars):
        """Test that rate limiting is properly integrated."""
        with patch(
            "deep_learning_final_assignment.core.models.async_openai_model.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"sentiment": "Positive"}'
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4-turbo"
            mock_response.usage = Mock()
            mock_response.usage.model_dump.return_value = {"total_tokens": 50}

            mock_client.chat.completions.create.return_value = mock_response

            # Create model with very restrictive rate limiting for testing
            model = AsyncOpenAIModel(
                "gpt-4-turbo", requests_per_minute=60
            )  # 1 request per second
            model.client = mock_client

            # Mock the is_available property by patching the class method
            with patch.object(
                AsyncOpenAIModel,
                "is_available",
                new_callable=lambda: property(lambda self: True),
            ):
                start_time = datetime.now()

                # Make two requests
                await model.generate("Test prompt 1")
                await model.generate("Test prompt 2")

                end_time = datetime.now()

                # Should have taken at least 1 second due to rate limiting
                time_diff = (end_time - start_time).total_seconds()
                assert time_diff >= 0.9  # Allow some tolerance


class TestAsyncOllamaModel:
    """Test the AsyncOllamaModel class."""

    @pytest.mark.asyncio
    async def test_async_ollama_model_init(self):
        """Test AsyncOllama model initialization."""
        with patch(
            "deep_learning_final_assignment.core.models.async_ollama_model.httpx.AsyncClient"
        ) as mock_client:
            model = AsyncOllamaModel("qwen3:4b")

            assert model.model_name == "qwen3:4b"
            assert model.host == "http://localhost:11434"
            assert model.rate_limiter is None  # No rate limiting by default for Ollama

    @pytest.mark.asyncio
    async def test_async_ollama_model_with_rate_limiting(self):
        """Test AsyncOllama model with optional rate limiting."""
        with patch(
            "deep_learning_final_assignment.core.models.async_ollama_model.httpx.AsyncClient"
        ) as mock_client:
            model = AsyncOllamaModel("qwen3:4b", requests_per_minute=120)

            assert model.rate_limiter is not None
            assert model.rate_limiter.requests_per_minute == 120

    @pytest.mark.asyncio
    async def test_async_ollama_model_generate(self):
        """Test AsyncOllama model generate method."""
        with patch(
            "deep_learning_final_assignment.core.models.async_ollama_model.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": '{"sentiment": "Positive"}',
                "model": "qwen3:4b",
                "created_at": "2024-01-01T00:00:00Z",
                "done": True,
                "total_duration": 1000000,
                "eval_count": 50,
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            model = AsyncOllamaModel("qwen3:4b")
            model.client = mock_client

            # Mock the is_available property by patching the class method
            with patch.object(
                AsyncOllamaModel,
                "is_available",
                new_callable=lambda: property(lambda self: True),
            ):
                response = await model.generate("Test prompt", temperature=0.2)

                assert isinstance(response, ModelResponse)
                assert response.content == '{"sentiment": "Positive"}'
                assert response.model_name == "qwen3:4b"
                assert response.prompt_used == "Test prompt"

    @pytest.mark.asyncio
    async def test_async_ollama_model_batch_generate(self):
        """Test AsyncOllama model batch_generate method."""
        with patch(
            "deep_learning_final_assignment.core.models.async_ollama_model.httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": '{"sentiment": "Positive"}',
                "model": "qwen3:4b",
                "created_at": "2024-01-01T00:00:00Z",
                "done": True,
                "total_duration": 1000000,
                "eval_count": 50,
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            model = AsyncOllamaModel("qwen3:4b")
            model.client = mock_client

            # Mock the is_available property by patching the class method
            with patch.object(
                AsyncOllamaModel,
                "is_available",
                new_callable=lambda: property(lambda self: True),
            ):
                responses = await model.batch_generate(["Prompt 1", "Prompt 2"])

                assert len(responses) == 2
                assert all(isinstance(r, ModelResponse) for r in responses)
                assert all(r.content == '{"sentiment": "Positive"}' for r in responses)
