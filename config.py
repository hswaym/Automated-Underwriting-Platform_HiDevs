"""Application configuration — reads from .env via pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Open-source LLM options: "open-source", "ollama", "groq", "built-in"
    llm_provider: str = "open-source"
    # Open-source model name (e.g. llama-3.3-70b-versatile, llama3.2, qwen2.5)
    model: str = "Llama 3.3 70B (Open-Source)"
    # Optional Groq free tier key (free of cost)
    groq_api_key: str = ""
    # Optional local Ollama server URL (free, local, offline)
    ollama_base_url: str = "http://localhost:11434"
    # Legacy Anthropic key (optional)
    anthropic_api_key: str = ""
    guidelines_path: Path = Path("./guidelines/underwriting_guidelines.txt")
    max_image_dimension: int = 1280

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Module-level singleton — import settings from config
settings = Settings()  # type: ignore[call-arg]
