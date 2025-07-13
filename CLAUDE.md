# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Normal mode with image generation
python main.py

# Test all components
python main.py --test
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Copy `.env.example` to `.env` and configure API keys:
- Set `LLM_PROVIDER` to one of: `openai`, `gemini`, `deepseek`, `openrouter`
- Set `IMAGE_PROVIDER` to one of: `openai`, `stability`
- Add corresponding API keys for your chosen providers

## Architecture

This is a multi-agent AI system that converts technical case studies into LinkedIn posts with appealing images through iterative refinement:

### Core Components

**AgentOrchestrator** (`src/orchestrator.py`): 
- Main workflow coordinator
- Manages iterations between Writer and Critic agents
- Orchestrates image generation and validation workflow
- Handles console output and user interaction
- Stores iteration history and displays comprehensive summary

**WriterAgent** (`src/agents/writer.py`):
- Creates initial LinkedIn posts from case studies
- Refines posts based on Critic feedback
- Enforces 60-word limit constraint
- Uses LangChain message format

**CriticAgent** (`src/agents/critic.py`):
- Acts as layman reviewer with no technical background
- Analyzes posts for understandability
- Returns boolean understandable status plus detailed feedback
- Uses "UNDERSTANDABLE: YES/NO" response parsing

**ImageGeneratorAgent** (`src/agents/image_generator.py`):
- Generates 3D appealing images based on final LinkedIn post
- Creates optimized prompts for professional, LinkedIn-ready visuals
- Supports OpenAI DALL-E and Stability AI providers
- Downloads and saves images locally in `generated_images/` folder

**ImageCriticAgent** (`src/agents/image_critic.py`):
- Evaluates generated images for LinkedIn appeal and quality
- Scores images on visual appeal, brand resonance, concept clarity
- Provides improvement suggestions for image refinement
- Uses text-based analysis (can be enhanced with vision models)

**TestingAgent** (`src/agents/testing.py`):
- Comprehensive testing suite for all system components
- Validates environment setup and API configurations
- Tests individual agents and integration workflows
- Provides detailed reports and recommendations

**LLMFactory** (`src/utils/llm_factory.py`):
- Creates LLM instances based on provider configuration
- Supports OpenAI, Gemini (via wrapper), DeepSeek, and OpenRouter
- Handles provider-specific authentication and base URLs
- GeminiLangChainWrapper provides LangChain compatibility

### Workflow
1. User provides case study (manual input or file)
2. Writer creates initial 60-word LinkedIn post
3. Critic analyzes post for layman understandability
4. If not understandable, Writer refines based on feedback
5. Process repeats up to MAX_ITERATIONS (default: 5)
6. **Image Generation Phase** (if enabled):
   - ImageGenerator creates appealing 3D image based on final post
   - ImageCritic evaluates image quality and LinkedIn appeal
   - If score < 7/10, regenerate with improved prompts
   - Process repeats up to MAX_IMAGE_ITERATIONS (default: 3)
7. Final post and image saved with complete iteration history

### Key Configuration
- `MAX_ITERATIONS`: Maximum text refinement cycles (default: 5)
- `MAX_IMAGE_ITERATIONS`: Maximum image refinement cycles (default: 3)
- `TEMPERATURE`: Model creativity setting (default: 0.7)
- `IMAGE_PROVIDER`: Image generation service (`openai` or `stability`)
- Word count target: Exactly 60 words for LinkedIn posts
- Image output: Professional 3D renders in `generated_images/` folder