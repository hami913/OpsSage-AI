import pytest
from unittest.mock import MagicMock, patch

from src.agents.llm_client import GeminiClient


def test_gemini_client_initialization():
    with patch("src.agents.llm_client.genai.Client") as mock_client:
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
            client = GeminiClient()

            assert client.model == "gemini-2.5-flash"
            mock_client.assert_called_once_with(api_key="test-key")


def test_missing_api_key_raises_error():
    with patch("src.agents.llm_client.load_dotenv"), patch.dict("os.environ", {}, clear=True):
        with patch("src.agents.llm_client.genai.Client"):
            with pytest.raises(ValueError, match="LLM_API_KEY"):
                GeminiClient()


def test_generate_text_success():
    with patch("src.agents.llm_client.genai.Client") as mock_client:
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
            mock_response = MagicMock()
            mock_response.text = "Root cause analysis completed."

            mock_client.return_value.models.generate_content.return_value = (
                mock_response
            )

            client = GeminiClient()
            result = client.generate_text("Analyze this incident")

            assert result == "Root cause analysis completed."


def test_empty_prompt_raises_error():
    with patch("src.agents.llm_client.genai.Client"):
        with patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
            client = GeminiClient()

            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                client.generate_text("   ")
