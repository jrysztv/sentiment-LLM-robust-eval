"""
Tests for model interfaces and implementations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from deep_learning_final_assignment.core.models.base import BaseModel, ModelResponse
from deep_learning_final_assignment.core.models.openai_model import OpenAIModel
from deep_learning_final_assignment.core.models.ollama_model import OllamaModel


class TestModelResponse:
    """Test the ModelResponse dataclass."""

    def test_model_response_creation(self):
        """Test creating a ModelResponse."""
        response = ModelResponse(
            content="Test response",
            model_name="test_model",
            prompt_used="Test prompt",
            metadata={"temperature": 0.5},
        )

        assert response.content == "Test response"
        assert response.model_name == "test_model"
        assert response.prompt_used == "Test prompt"
        assert response.metadata == {"temperature": 0.5}

    def test_model_response_default_metadata(self):
        """Test ModelResponse with default metadata."""
        response = ModelResponse(
            content="Test response", model_name="test_model", prompt_used="Test prompt"
        )

        assert response.metadata == {}


class TestBaseModel:
    """Test the BaseModel abstract class."""

    def test_base_model_cannot_be_instantiated(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel("test_model")

    def test_concrete_model_implementation(self):
        """Test a concrete implementation of BaseModel."""

        class TestModel(BaseModel):
            def _setup(self):
                self.setup_called = True

            def generate(self, prompt, **kwargs):
                return ModelResponse(
                    content="Test response",
                    model_name=self.model_name,
                    prompt_used=prompt,
                )

            def batch_generate(self, prompts, **kwargs):
                return [self.generate(prompt, **kwargs) for prompt in prompts]

            @property
            def is_available(self):
                return True

        model = TestModel("test_model", param1="value1")

        assert model.model_name == "test_model"
        assert model.config == {"param1": "value1"}
        assert model.setup_called is True
        assert model.is_available is True

        # Test generate
        response = model.generate("Test prompt")
        assert isinstance(response, ModelResponse)
        assert response.content == "Test response"

        # Test batch_generate
        responses = model.batch_generate(["Prompt 1", "Prompt 2"])
        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)

        # Test get_info
        info = model.get_info()
        expected_info = {
            "model_name": "test_model",
            "config": {"param1": "value1"},
            "available": True,
        }
        assert info == expected_info


class TestOpenAIModel:
    """Test the OpenAIModel class."""

    def test_openai_model_init_with_api_key(self, mock_env_vars):
        """Test OpenAI model initialization with API key."""
        with patch(
            "deep_learning_final_assignment.core.models.openai_model.OpenAI"
        ) as mock_openai:
            model = OpenAIModel("gpt-4-turbo", api_key="test_key")

            assert model.model_name == "gpt-4-turbo"
            assert model.api_key == "test_key"
            mock_openai.assert_called_once_with(api_key="test_key")

    def test_openai_model_init_with_env_var(self, mock_env_vars):
        """Test OpenAI model initialization with environment variable."""
        with patch(
            "deep_learning_final_assignment.core.models.openai_model.OpenAI"
        ) as mock_openai:
            model = OpenAIModel("gpt-4-turbo")

            assert model.api_key == "test_openai_key"
            mock_openai.assert_called_once_with(api_key="test_openai_key")

    def test_openai_model_init_no_api_key(self, monkeypatch):
        """Test OpenAI model initialization without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIModel("gpt-4-turbo")

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_generate(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model generate method."""
        # Setup mock
        mock_client = Mock()
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
        model = OpenAIModel("gpt-4-turbo")
        model.client = mock_client  # Set directly for testing

        response = model.generate("Test prompt", temperature=0.2)

        assert isinstance(response, ModelResponse)
        assert response.content == '{"sentiment": "Positive"}'
        assert response.model_name == "gpt-4-turbo"
        assert response.prompt_used == "Test prompt"
        assert response.metadata["temperature"] == 0.2

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_generate_error(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model generate method with error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        model = OpenAIModel("gpt-4-turbo")
        model.client = mock_client

        # Mock is_available to return True to bypass availability check
        with patch.object(
            type(model),
            "is_available",
            new_callable=lambda: property(lambda self: True),
        ):
            with pytest.raises(RuntimeError, match="OpenAI API call failed"):
                model.generate("Test prompt")

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_batch_generate(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model batch_generate method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"sentiment": "Positive"}'
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4-turbo"
            mock_response.usage = Mock()
            mock_response.usage.model_dump.return_value = {"total_tokens": 50}
            return mock_response

        mock_client.chat.completions.create.side_effect = mock_create

        model = OpenAIModel("gpt-4-turbo")
        model.client = mock_client

        # Mock is_available to return True and reset call count
        with patch.object(
            type(model),
            "is_available",
            new_callable=lambda: property(lambda self: True),
        ):
            mock_client.chat.completions.create.call_count = 0  # Reset counter
            responses = model.batch_generate(["Prompt 1", "Prompt 2"])

            assert len(responses) == 2
            assert all(isinstance(r, ModelResponse) for r in responses)
            assert mock_client.chat.completions.create.call_count == 2

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_is_available_true(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model is_available when available."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel("gpt-4-turbo")
        model.client = mock_client

        assert model.is_available is True

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_is_available_false(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model is_available when not available."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        model = OpenAIModel("gpt-4-turbo")
        model.client = mock_client

        assert model.is_available is False


class TestOllamaModel:
    """Test the OllamaModel class."""

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_init(self, mock_ollama):
        """Test Ollama model initialization."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}

        model = OllamaModel("qwen2.5:7b", host="http://localhost:11434")

        assert model.model_name == "qwen2.5:7b"
        assert model.host == "http://localhost:11434"
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_init_error(self, mock_ollama):
        """Test Ollama model initialization with connection error."""
        mock_ollama.Client.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to connect to Ollama server"):
            OllamaModel("qwen2.5:7b")

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_generate(self, mock_ollama):
        """Test Ollama model generate method."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }  # Make model available

        mock_response = {
            "response": '{"sentiment": "Neutral"}',
            "model": "qwen2.5:7b",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000,
            "eval_count": 20,
            "eval_duration": 300000,
        }

        mock_client.generate.return_value = mock_response

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        response = model.generate("Test prompt", temperature=0.2)

        assert isinstance(response, ModelResponse)
        assert response.content == '{"sentiment": "Neutral"}'
        assert response.model_name == "qwen2.5:7b"
        assert response.prompt_used == "Test prompt"
        assert response.metadata["options"]["temperature"] == 0.2

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_generate_error(self, mock_ollama):
        """Test Ollama model generate method with error."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }  # Make model available
        mock_client.generate.side_effect = Exception("Generation failed")

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        with pytest.raises(RuntimeError, match="Ollama API call failed"):
            model.generate("Test prompt")

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_batch_generate(self, mock_ollama):
        """Test Ollama model batch_generate method."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }  # Make model available

        def mock_generate(**kwargs):
            return {
                "response": '{"sentiment": "Neutral"}',
                "model": "qwen2.5:7b",
                "done": True,
            }

        mock_client.generate.side_effect = mock_generate

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        responses = model.batch_generate(["Prompt 1", "Prompt 2"])

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert mock_client.generate.call_count == 2

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_is_available_true(self, mock_ollama):
        """Test Ollama model is_available when model is available."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b"}, {"name": "llama2:7b"}]
        }

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        assert model.is_available is True

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_is_available_false(self, mock_ollama):
        """Test Ollama model is_available when model is not available."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "llama2:7b"}]  # Different model
        }

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        assert model.is_available is False

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_is_available_partial_match(self, mock_ollama):
        """Test Ollama model is_available with partial name match."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b-instruct"}, {"name": "llama2:7b"}]
        }

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        assert model.is_available is True

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_pull_model_success(self, mock_ollama):
        """Test Ollama model pull_model method success."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        mock_client.pull.return_value = None

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        result = model.pull_model()

        assert result is True
        mock_client.pull.assert_called_once_with("qwen2.5:7b")

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_pull_model_failure(self, mock_ollama):
        """Test Ollama model pull_model method failure."""
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {"models": []}
        mock_client.pull.side_effect = Exception("Pull failed")

        model = OllamaModel("qwen2.5:7b")
        model.client = mock_client

        result = model.pull_model()

        assert result is False
