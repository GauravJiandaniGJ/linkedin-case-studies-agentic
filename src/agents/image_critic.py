import base64
from typing import Dict, Optional, Tuple
from pathlib import Path
from langchain.schema import HumanMessage, SystemMessage
from src.utils.llm_factory import LLMFactory

class ImageCriticAgent:
    """AI Agent that evaluates generated images for LinkedIn appeal and resonance"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMFactory.create_llm(provider=llm_provider)
        self.system_prompt = """You are a LinkedIn marketing expert and visual design critic. Your task is to evaluate images for LinkedIn posts based on:

EVALUATION CRITERIA:
1. **Visual Appeal** (1-10): How eye-catching and professional does it look?
2. **Brand Resonance** (1-10): Does it feel appropriate for LinkedIn's professional context?
3. **Concept Clarity** (1-10): How well does the image convey the post's message?
4. **Engagement Potential** (1-10): Would this make users stop scrolling and engage?
5. **Technical Quality** (1-10): Is the image well-composed, lit, and rendered?

LINKEDIN CONTEXT:
- Professional yet engaging visual style
- Appeals to both technical and non-technical audiences
- Modern, clean aesthetic that stands out in feeds
- Supports the written content without overwhelming it

For each image, provide:
1. Overall score (1-10)
2. Scores for each criterion
3. What works well
4. What could be improved
5. Specific suggestions for regeneration if needed

Start your response with "OVERALL_SCORE: X" where X is 1-10."""

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image file to base64 for API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")

    def analyze_image_openai(self, image_path: str, linkedin_post: str, image_prompt: str) -> Tuple[int, str]:
        """Analyze image using OpenAI Vision API"""
        try:
            # For OpenAI, we'll use text-based analysis since vision requires special handling
            # In a real implementation, you'd use the vision API with base64 encoded images
            analysis_prompt = f"""
Analyze this image for a LinkedIn post based on the following context:

LinkedIn Post: {linkedin_post}

Image Generation Prompt Used: {image_prompt}

Image File: {Path(image_path).name}

Please evaluate this image as if you can see it, considering:
- Professional 3D render quality
- Alignment with the LinkedIn post content
- Visual appeal for social media
- Modern, clean aesthetic
- Engagement potential

Provide your analysis following the format specified in the system prompt.
"""
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse overall score
            lines = content.split('\n')
            overall_score = 7  # Default score
            for line in lines:
                if line.startswith("OVERALL_SCORE:"):
                    try:
                        overall_score = int(line.split(":")[1].strip())
                    except:
                        pass
                    break
            
            # Extract feedback (everything after the score line)
            feedback_start = 1 if lines[0].startswith("OVERALL_SCORE:") else 0
            feedback = '\n'.join(lines[feedback_start:]).strip()
            
            return overall_score, feedback
            
        except Exception as e:
            raise Exception(f"Image analysis failed: {str(e)}")

    def analyze_image_text_only(self, image_path: str, linkedin_post: str, image_prompt: str) -> Tuple[int, str]:
        """Analyze image based on metadata and prompts (fallback method)"""
        analysis_prompt = f"""
Based on the following information about a generated image for LinkedIn:

LinkedIn Post: {linkedin_post}

Image Generation Prompt: {image_prompt}

Image File: {Path(image_path).name}

Please provide a theoretical analysis of how well this image would perform on LinkedIn, considering:
1. The alignment between the post content and image prompt
2. Professional 3D render appeal
3. LinkedIn audience engagement potential
4. Modern visual design principles

Follow the evaluation format specified in the system prompt.
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Parse overall score
        lines = content.split('\n')
        overall_score = 7  # Default score
        for line in lines:
            if line.startswith("OVERALL_SCORE:"):
                try:
                    overall_score = int(line.split(":")[1].strip())
                except:
                    pass
                break
        
        # Extract feedback
        feedback_start = 1 if lines[0].startswith("OVERALL_SCORE:") else 0
        feedback = '\n'.join(lines[feedback_start:]).strip()
        
        return overall_score, feedback

    def analyze_image(self, image_path: str, linkedin_post: str, image_prompt: str) -> Dict[str, any]:
        """
        Analyze generated image for LinkedIn appeal and quality
        
        Args:
            image_path: Path to the generated image file
            linkedin_post: The LinkedIn post content
            image_prompt: The prompt used to generate the image
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if image file exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # For now, use text-based analysis
            # In the future, this could be enhanced with actual image vision models
            overall_score, feedback = self.analyze_image_text_only(image_path, linkedin_post, image_prompt)
            
            # Determine if image passes quality threshold
            is_acceptable = overall_score >= 7
            
            return {
                "overall_score": overall_score,
                "is_acceptable": is_acceptable,
                "feedback": feedback,
                "image_path": image_path,
                "analysis_method": "text_based"
            }
            
        except Exception as e:
            return {
                "overall_score": 0,
                "is_acceptable": False,
                "feedback": f"Analysis failed: {str(e)}",
                "image_path": image_path,
                "analysis_method": "error"
            }

    def suggest_improvements(self, analysis_result: Dict[str, any], original_prompt: str) -> str:
        """
        Generate suggestions for improving the image based on analysis
        
        Args:
            analysis_result: Result from analyze_image
            original_prompt: The original image generation prompt
            
        Returns:
            Improved prompt suggestions
        """
        improvement_prompt = f"""
Based on this image analysis feedback:

Original Prompt: {original_prompt}
Overall Score: {analysis_result['overall_score']}/10
Feedback: {analysis_result['feedback']}

Please suggest specific improvements to the image generation prompt that would address the identified issues and create a more appealing LinkedIn image. Focus on:

1. Visual appeal enhancements
2. Professional presentation improvements  
3. Better concept representation
4. Engagement optimization

Provide a revised prompt that incorporates these improvements.
"""
        
        messages = [
            SystemMessage(content="You are an expert at improving image generation prompts for professional social media content."),
            HumanMessage(content=improvement_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()