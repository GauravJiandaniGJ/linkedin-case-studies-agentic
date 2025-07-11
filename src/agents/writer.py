from typing import Dict, Optional
from langchain.schema import HumanMessage, SystemMessage
from src.utils.llm_factory import LLMFactory

class WriterAgent:
    """AI Agent that writes LinkedIn posts from case studies"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMFactory.create_llm(provider=llm_provider)
        self.system_prompt = """You are an expert LinkedIn content writer. Your task is to:
1. Take a technical case study and create a crispy, engaging LinkedIn post
2. Keep it EXACTLY 50 words (no more, no less)
3. Make it attention-grabbing for LinkedIn scrollers
4. Include relevant hashtags at the end

When given feedback, refine your post to address the concerns while maintaining the 50-word limit."""
    
    def write_initial_post(self, case_study: str) -> str:
        """Create initial LinkedIn post from case study"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Create a LinkedIn post from this case study:\n\n{case_study}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def refine_post(self, current_post: str, feedback: str) -> str:
        """Refine post based on critic feedback"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Current post:
{current_post}

Feedback from a layman reader:
{feedback}

Please refine the post to address these concerns while keeping it crispy and EXACTLY 50 words.""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
