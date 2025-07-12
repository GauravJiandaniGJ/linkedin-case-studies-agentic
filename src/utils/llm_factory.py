import os
from typing import Any
from dotenv import load_dotenv

# Import OpenAI support (only from langchain_openai)
from langchain_openai import ChatOpenAI

# Try to import Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

class GeminiLangChainWrapper:
    """Wrapper to make Gemini API compatible with LangChain interface"""

    def __init__(self, api_key: str, model: str, temperature: float):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def invoke(self, messages):
        """Convert LangChain messages to Gemini format and get response"""
        # Extract the last user message
        prompt = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                prompt += msg.content + "\n\n"

        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )

        # Create a response object compatible with LangChain
        class Response:
            def __init__(self, text):
                self.content = text

        return Response(response.text)

class LLMFactory:
    """Factory class to create LLM instances based on provider"""

    @staticmethod
    def create_llm(provider: str = None, temperature: float = 0.7) -> Any:
        """
        Create an LLM instance based on the provider

        Args:
            provider: LLM provider name (openai, gemini, deepseek)
            temperature: Model temperature for randomness

        Returns:
            LLM instance
        """
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print("DEBUG: Creating ChatOpenAI with", {
                "api_key": api_key,
                "model": model,
                "temperature": temperature
            })
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature
            )

        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ValueError("Gemini support not available. Please install google-generativeai package.")
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return GeminiLangChainWrapper(api_key, model, temperature)

        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            api_key = os.getenv("DEEPSEEK_API_KEY")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            print("DEBUG: Creating ChatOpenAI (DeepSeek) with", {
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "base_url": "https://api.deepseek.com/v1"
            })
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                base_url="https://api.deepseek.com/v1",
            )

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")
            print("DEBUG: Creating DeepSeek (OpenRouter) with", {
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "base_url": "https://openrouter.ai/api/v1"
            })
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                base_url="https://openrouter.ai/api/v1",
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
