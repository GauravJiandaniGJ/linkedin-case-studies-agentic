import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage
from dotenv import load_dotenv

load_dotenv()

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
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=temperature
            )
        
        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            api_key = os.getenv("DEEPSEEK_API_KEY")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                base_url="https://api.deepseek.com/v1"
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
