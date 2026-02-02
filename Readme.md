# LinkedIn AI Agent - Case Study to Post Converter

An intelligent Python-based AI agent system that converts technical case studies into crispy, layman-friendly LinkedIn posts through iterative refinement between a Writer and Critic agent.

## 🎯 Features

- **Dual Agent System**: Writer agent creates posts, Critic agent provides layman perspective
- **Iterative Refinement**: Automatically refines content until it's understandable to non-technical audiences
- **Multi-LLM Support**: Works with OpenAI, Google Gemini, and DeepSeek
- **50-Word Precision**: Creates perfectly sized LinkedIn posts
- **Beautiful CLI**: Rich terminal interface with progress tracking
- **Flexible Configuration**: Easy switching between LLM providers via environment variables

## 🚀 Quick Start
![Untitled design (1) (1)](https://github.com/user-attachments/assets/87ab2d8e-09af-4f40-8422-143f2389c649)

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least one API key (OpenAI, Gemini, or DeepSeek)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/linkedin-ai-agent.git
cd linkedin-ai-agent
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Optional: For Gemini support
pip install google-generativeai langchain-google-genai
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your preferred text editor
```

5. **Run the application**
```bash
python main.py
```

## 🔑 API Keys Setup

### OpenAI (Recommended for beginners)
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Add to `.env`: `OPENAI_API_KEY=your_key_here`

**Pricing**: ~$0.002 per run with GPT-4o-mini

### Google Gemini (Free tier available)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key
5. Add to `.env`: `GEMINI_API_KEY=your_key_here`

**Pricing**: Free tier includes 15 requests/minute

### DeepSeek (Cost-effective)
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Register an account
3. Go to API Keys section
4. Create new API key
5. Add to `.env`: `DEEPSEEK_API_KEY=your_key_here`

**Pricing**: Very competitive rates, ~$0.001 per run

## 📁 Project Structure

```
linkedin-ai-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── writer.py      # Writer agent implementation
│   │   └── critic.py      # Critic agent implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm_factory.py # LLM provider factory
│   └── orchestrator.py    # Main workflow orchestrator
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
# LLM Provider: openai, gemini, or deepseek
LLM_PROVIDER=openai

# API Keys (add only the one you're using)
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Model names (optional - defaults shown)
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-1.5-flash
DEEPSEEK_MODEL=deepseek-chat

# Agent settings
MAX_ITERATIONS=5        # Maximum refinement iterations
TEMPERATURE=0.7         # Model creativity (0.0-1.0)
```

## 💡 Usage Examples

### Basic Usage
```bash
python main.py
# Enter your case study when prompted
```

### Using a File
```bash
# Create a case study file
echo "Your technical case study here..." > case_study.txt

# Run the agent
python main.py
# Choose "Yes" when asked about file input
# Enter: case_study.txt
```

### Example Case Study
```
Our fintech startup implemented a microservices architecture using 
Kubernetes, reducing deployment time from 2 hours to 5 minutes. 
We used GitOps with ArgoCD for declarative deployments, implemented 
service mesh with Istio for inter-service communication, and achieved 
99.9% uptime. The system now handles 100,000 transactions per second 
with automatic scaling based on load.
```

### Expected Output
```
Transformed finance deployments: 2 hours to 5 minutes! Our smart 
system now handles 100,000 transactions every second with 99.9% 
reliability. Like having a super-fast, always-open bank that never 
sleeps. Technology making money movement instantaneous and dependable.
#Fintech #Innovation #TechTransformation #DigitalBanking
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'openai'"**
   - Make sure virtual environment is activated
   - Run: `pip install -r requirements.txt`

2. **"Invalid API key"**
   - Check your `.env` file has the correct key
   - Ensure no extra spaces or quotes around the key

3. **"Gemini not available"**
   - This is normal if google-generativeai didn't install
   - Use OpenAI or DeepSeek instead

4. **Rate limit errors**
   - Wait a few minutes before retrying
   - Consider using a different LLM provider

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM orchestration
- Terminal UI powered by [Rich](https://rich.readthedocs.io/)
- Inspired by the need for better technical communication

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your.email@example.com

---
Made with ❤️ for better LinkedIn content
