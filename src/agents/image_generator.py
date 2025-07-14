import os
import requests
import base64
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from langchain.schema import HumanMessage, SystemMessage
from src.utils.llm_factory import LLMFactory
from dotenv import load_dotenv

load_dotenv()

class ImageGeneratorAgent:
    """AI Agent that generates appealing 3D images for LinkedIn posts"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMFactory.create_llm(provider=llm_provider)
        self.image_provider = os.getenv("IMAGE_PROVIDER", "openai").lower()
        self.output_dir = Path("generated_images")
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate image provider support
        self.supported_providers = ["openai", "stability", "gemini"]
        self.unsupported_providers = {
            "openrouter": "OpenRouter doesn't support image generation. Use OpenAI or Stability AI instead.",
            "deepseek": "DeepSeek's image generation (Janus Pro) is not available via standard API. Use OpenAI or Stability AI instead."
        }
        
        if self.image_provider not in self.supported_providers:
            if self.image_provider in self.unsupported_providers:
                raise ValueError(self.unsupported_providers[self.image_provider])
            else:
                raise ValueError(f"Unknown image provider '{self.image_provider}'. Supported providers: {', '.join(self.supported_providers)}")
        
        self.system_prompt = """You are an expert at creating image prompts for 3D renders that would be perfect for LinkedIn posts.

Your task is to analyze a LinkedIn post and create a detailed prompt for generating an appealing 3D image that:
1. Visually represents the core concept/achievement in the post
2. Is professional yet eye-catching for LinkedIn
3. Uses modern 3D aesthetics (clean, minimalist, corporate-friendly)
4. Resonates with both technical and non-technical audiences
5. Includes relevant visual metaphors that make complex concepts accessible

Focus on:
- Clean, modern 3D design elements
- Professional color schemes (blues, greens, whites, subtle gradients)
- Abstract representations of technology/business concepts
- Geometric shapes and modern typography when relevant
- Lighting that makes the image pop on social media

Avoid:
- Overly complex or cluttered designs
- Dark or gloomy themes
- Too many technical details
- Generic stock photo looks

Return ONLY the image generation prompt, nothing else."""

    def generate_image_prompt(self, linkedin_post: str) -> str:
        """Generate an optimized prompt for image generation"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Create an image generation prompt for this LinkedIn post:\n\n{linkedin_post}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def generate_image_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        enhanced_prompt = (
            f"A realistic photograph of a clean corporate workspace with a close-up of a laptop displaying a chatbot interface, next to a coffee cup and documents. "
            f"Minimalist, elegant, professional, LinkedIn-ready. {prompt}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "dall-e-3",
            "prompt": enhanced_prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid",
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        image_url = result["data"][0]["url"]
        
        return {
            "url": image_url,
            "prompt": enhanced_prompt,
            "provider": "openai",
            "model": "dall-e-3"
        }
    
    def generate_image_stability(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Stability AI"""
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise ValueError("STABILITY_API_KEY not found in environment variables")
        
        # Enhance prompt for better 3D results
        enhanced_prompt = f"Professional 3D render, modern corporate design, clean aesthetic, high quality, LinkedIn social media ready: {prompt}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [{"text": enhanced_prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
            "style_preset": "photographic"
        }
        
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Stability AI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        image_data = result["artifacts"][0]["base64"]
        
        return {
            "base64": image_data,
            "prompt": enhanced_prompt,
            "provider": "stability",
            "model": "stable-diffusion-xl"
        }
    
    def generate_image_gemini(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Google Gemini/Imagen"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Enhanced prompt for better 3D results
        enhanced_prompt = f"Professional 3D render, modern corporate design, clean aesthetic, LinkedIn-ready social media image: {prompt}"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        # Using Imagen 3 via Gemini API
        data = {
            "contents": [{
                "parts": [{
                    "text": f"Generate an image: {enhanced_prompt}"
                }]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.7
            }
        }
        
        # Note: This is a simplified implementation
        # In reality, you'd need to use the proper Gemini API client
        # For now, we'll raise an informative error
        raise NotImplementedError(
            "Gemini image generation requires Google AI Studio setup and paid tier access. "
            "Please use OpenAI or Stability AI for now, or implement full Gemini API integration."
        )
    
    def download_image(self, image_url: str, filename: str) -> str:
        """Download image from URL and save locally"""
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
        
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return str(filepath)
    
    def save_base64_image(self, base64_data: str, filename: str) -> str:
        """Save base64 image data to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        
        return str(filepath)
    
    def generate_image(self, linkedin_post: str) -> Dict[str, Any]:
        """
        Generate an appealing image for the LinkedIn post
        
        Args:
            linkedin_post: The final LinkedIn post content
            
        Returns:
            Dictionary with image info and local file path
        """
        # Generate optimized prompt
        image_prompt = self.generate_image_prompt(linkedin_post)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if self.image_provider == "openai":
                result = self.generate_image_openai(image_prompt)
                filename = f"linkedin_image_openai_{timestamp}.png"
                filepath = self.download_image(result["url"], filename)
                
            elif self.image_provider == "stability":
                result = self.generate_image_stability(image_prompt)
                filename = f"linkedin_image_stability_{timestamp}.png"
                filepath = self.save_base64_image(result["base64"], filename)
                
            elif self.image_provider == "gemini":
                result = self.generate_image_gemini(image_prompt)
                filename = f"linkedin_image_gemini_{timestamp}.png"
                # Implementation would depend on Gemini's response format
                filepath = self.save_base64_image(result["base64"], filename)
                
            else:
                raise ValueError(f"Unknown image provider: {self.image_provider}")
            
            return {
                "filepath": filepath,
                "filename": filename,
                "prompt": image_prompt,
                "enhanced_prompt": result["prompt"],
                "provider": result["provider"],
                "model": result.get("model", "unknown"),
                "timestamp": timestamp
            }
            
        except Exception as e:
            raise Exception(f"Image generation failed: {str(e)}")
