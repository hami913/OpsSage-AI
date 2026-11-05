import os

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    """Central client for Google Gemini API communication."""

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv(
            "LLM_MODEL",
            "gemini-2.5-flash"
        )

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is missing from environment configuration."
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str) -> str:
        """Generate text using the configured Gemini model."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text.strip()