from typing import Dict, Optional, Tuple
from langchain.schema import HumanMessage, SystemMessage
from src.utils.llm_factory import LLMFactory

class CriticAgent:
    """AI Agent that acts as a layman reviewing LinkedIn posts"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMFactory.create_llm(provider=llm_provider)
        self.system_prompt = """You are a regular LinkedIn user with NO technical background. You're scrolling through your feed during a coffee break. 

Your task is to:
1. Read the LinkedIn post as if you're a layman
2. Identify ANY technical jargon, complex concepts, or industry-specific terms you don't understand
3. Provide specific feedback about what confused you
4. Be honest - if you can understand it, say so. If not, explain what's unclear

Remember: You know basic business terms but NOT technical/industry-specific jargon."""
    
    def analyze_post(self, post: str) -> Tuple[bool, str]:
        """
        Analyze if post is understandable to a layman
        
        Returns:
            Tuple of (is_understandable: bool, feedback: str)
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Read this LinkedIn post and tell me:
1. Can you understand what this is about?
2. What specific terms or concepts are confusing?
3. Would you stop scrolling to read this, or would you skip it?

Post:
{post}

Please start your response with "UNDERSTANDABLE: YES" or "UNDERSTANDABLE: NO" followed by your feedback.""")
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Parse response
        is_understandable = "UNDERSTANDABLE: YES" in content
        
        # Extract feedback (everything after the first line)
        feedback_lines = content.split('\n')[1:]
        feedback = '\n'.join(feedback_lines).strip()
        
        return is_understandable, feedback
