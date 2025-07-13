import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Import all agents for testing
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.agents.image_generator import ImageGeneratorAgent
from src.agents.image_critic import ImageCriticAgent
from src.orchestrator import AgentOrchestrator

load_dotenv()

class TestingAgent:
    """AI Agent that tests all components of the LinkedIn AI system"""
    
    def __init__(self):
        self.console = Console()
        self.test_results: List[Dict[str, Any]] = []
        self.test_case_study = """
        Our fintech startup implemented a microservices architecture using 
        Kubernetes, reducing deployment time from 2 hours to 5 minutes. 
        We used GitOps with ArgoCD for declarative deployments, implemented 
        service mesh with Istio for inter-service communication, and achieved 
        99.9% uptime. The system now handles 100,000 transactions per second 
        with automatic scaling based on load.
        """
    
    def log_test(self, test_name: str, status: str, details: str, duration: float = 0.0):
        """Log test result"""
        self.test_results.append({
            "test_name": test_name,
            "status": status,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_environment_setup(self) -> bool:
        """Test environment variables and dependencies"""
        self.console.print("\n[bold blue]Testing Environment Setup[/bold blue]")
        
        start_time = datetime.now()
        issues = []
        
        # Check required environment variables
        required_vars = ["LLM_PROVIDER"]
        optional_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", 
                        "STABILITY_API_KEY", "IMAGE_PROVIDER"]
        
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Missing required environment variable: {var}")
        
        # Check if at least one LLM API key is present
        llm_keys = [os.getenv(key) for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"]]
        if not any(llm_keys):
            issues.append("No LLM API keys found. Need at least one of: OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY")
        
        # Check image generation setup
        image_provider = os.getenv("IMAGE_PROVIDER", "openai")
        if image_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            issues.append("IMAGE_PROVIDER is 'openai' but OPENAI_API_KEY is missing")
        elif image_provider == "stability" and not os.getenv("STABILITY_API_KEY"):
            issues.append("IMAGE_PROVIDER is 'stability' but STABILITY_API_KEY is missing")
        
        # Check Python dependencies
        try:
            import openai
            import langchain
            import rich
            import requests
        except ImportError as e:
            issues.append(f"Missing Python dependency: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if issues:
            self.log_test("Environment Setup", "FAILED", "; ".join(issues), duration)
            self.console.print(f"[red]❌ Environment Setup Failed[/red]")
            for issue in issues:
                self.console.print(f"   - {issue}")
            return False
        else:
            self.log_test("Environment Setup", "PASSED", "All environment checks passed", duration)
            self.console.print("[green]✅ Environment Setup Passed[/green]")
            return True
    
    def test_writer_agent(self) -> bool:
        """Test WriterAgent functionality"""
        self.console.print("\n[bold blue]Testing Writer Agent[/bold blue]")
        
        start_time = datetime.now()
        
        try:
            writer = WriterAgent()
            
            # Test initial post creation
            initial_post = writer.write_initial_post(self.test_case_study)
            
            if not initial_post or len(initial_post.strip()) == 0:
                raise Exception("Writer returned empty post")
            
            word_count = len(initial_post.split())
            if word_count > 80:  # Allow some flexibility
                raise Exception(f"Post too long: {word_count} words (should be ~60)")
            
            # Test post refinement
            sample_feedback = "The post uses too many technical terms. Make it simpler."
            refined_post = writer.refine_post(initial_post, sample_feedback)
            
            if not refined_post or refined_post == initial_post:
                raise Exception("Writer failed to refine post or returned identical content")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Writer Agent", "PASSED", f"Created and refined posts successfully", duration)
            self.console.print(f"[green]✅ Writer Agent Passed[/green]")
            self.console.print(f"   Initial post: {word_count} words")
            self.console.print(f"   Refined successfully: {len(refined_post.split())} words")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Writer Agent", "FAILED", str(e), duration)
            self.console.print(f"[red]❌ Writer Agent Failed: {e}[/red]")
            return False
    
    def test_critic_agent(self) -> bool:
        """Test CriticAgent functionality"""
        self.console.print("\n[bold blue]Testing Critic Agent[/bold blue]")
        
        start_time = datetime.now()
        
        try:
            critic = CriticAgent()
            
            # Test with technical post (should be rejected)
            technical_post = "Implemented microservices with Kubernetes, ArgoCD, and Istio mesh for 99.9% uptime."
            is_understandable, feedback = critic.analyze_post(technical_post)
            
            if not feedback or len(feedback.strip()) == 0:
                raise Exception("Critic returned empty feedback")
            
            # Test with simple post (might be accepted)
            simple_post = "Our team made our website faster! Now it loads in 2 seconds instead of 30 seconds. Happy customers!"
            is_understandable2, feedback2 = critic.analyze_post(simple_post)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Critic Agent", "PASSED", f"Analyzed posts and provided feedback", duration)
            self.console.print(f"[green]✅ Critic Agent Passed[/green]")
            self.console.print(f"   Technical post understandable: {is_understandable}")
            self.console.print(f"   Simple post understandable: {is_understandable2}")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Critic Agent", "FAILED", str(e), duration)
            self.console.print(f"[red]❌ Critic Agent Failed: {e}[/red]")
            return False
    
    def test_image_generator_agent(self) -> bool:
        """Test ImageGeneratorAgent functionality"""
        self.console.print("\n[bold blue]Testing Image Generator Agent[/bold blue]")
        
        start_time = datetime.now()
        
        try:
            # Check if image generation is properly configured
            image_provider = os.getenv("IMAGE_PROVIDER", "openai")
            if image_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                raise Exception("OpenAI API key required for image generation")
            elif image_provider == "stability" and not os.getenv("STABILITY_API_KEY"):
                raise Exception("Stability AI API key required for image generation")
            
            image_gen = ImageGeneratorAgent()
            
            # Test prompt generation (this doesn't require API calls)
            test_post = "We reduced deployment time from 2 hours to 5 minutes using smart automation!"
            image_prompt = image_gen.generate_image_prompt(test_post)
            
            if not image_prompt or len(image_prompt.strip()) == 0:
                raise Exception("Image generator returned empty prompt")
            
            # Check if prompt mentions relevant concepts
            prompt_lower = image_prompt.lower()
            if not any(word in prompt_lower for word in ["3d", "professional", "modern", "clean"]):
                raise Exception("Image prompt doesn't include expected professional/3D keywords")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Image Generator Agent", "PASSED", f"Generated appropriate image prompt", duration)
            self.console.print(f"[green]✅ Image Generator Agent Passed[/green]")
            self.console.print(f"   Generated prompt length: {len(image_prompt)} chars")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Image Generator Agent", "FAILED", str(e), duration)
            self.console.print(f"[red]❌ Image Generator Agent Failed: {e}[/red]")
            return False
    
    def test_image_critic_agent(self) -> bool:
        """Test ImageCriticAgent functionality"""
        self.console.print("\n[bold blue]Testing Image Critic Agent[/bold blue]")
        
        start_time = datetime.now()
        
        try:
            image_critic = ImageCriticAgent()
            
            # Test analysis with mock data (no actual image file needed for text-based analysis)
            test_image_path = "test_image.png"
            test_post = "We reduced deployment time from 2 hours to 5 minutes!"
            test_prompt = "Professional 3D render of fast deployment concept"
            
            # Test improvement suggestions
            mock_analysis = {
                "overall_score": 6,
                "is_acceptable": False,
                "feedback": "Image lacks professional appeal and modern design elements",
                "image_path": test_image_path
            }
            
            suggestions = image_critic.suggest_improvements(mock_analysis, test_prompt)
            
            if not suggestions or len(suggestions.strip()) == 0:
                raise Exception("Image critic returned empty suggestions")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Image Critic Agent", "PASSED", f"Generated improvement suggestions", duration)
            self.console.print(f"[green]✅ Image Critic Agent Passed[/green]")
            self.console.print(f"   Generated suggestions length: {len(suggestions)} chars")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Image Critic Agent", "FAILED", str(e), duration)
            self.console.print(f"[red]❌ Image Critic Agent Failed: {e}[/red]")
            return False
    
    def test_orchestrator_integration(self) -> bool:
        """Test AgentOrchestrator integration"""
        self.console.print("\n[bold blue]Testing Orchestrator Integration[/bold blue]")
        
        start_time = datetime.now()
        
        try:
            # Test orchestrator initialization
            orchestrator = AgentOrchestrator()
            
            if not hasattr(orchestrator, 'writer') or not hasattr(orchestrator, 'critic'):
                raise Exception("Orchestrator missing required agents")
            
            # Verify configuration
            max_iterations = orchestrator.max_iterations
            if max_iterations <= 0 or max_iterations > 20:
                raise Exception(f"Invalid max_iterations: {max_iterations}")
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Orchestrator Integration", "PASSED", f"Orchestrator initialized successfully", duration)
            self.console.print(f"[green]✅ Orchestrator Integration Passed[/green]")
            self.console.print(f"   Max iterations: {max_iterations}")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_test("Orchestrator Integration", "FAILED", str(e), duration)
            self.console.print(f"[red]❌ Orchestrator Integration Failed: {e}[/red]")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        self.console.print("[bold green]🧪 LinkedIn AI Agent Testing Suite[/bold green]")
        self.console.print("=" * 60)
        
        test_functions = [
            self.test_environment_setup,
            self.test_writer_agent,
            self.test_critic_agent,
            self.test_image_generator_agent,
            self.test_image_critic_agent,
            self.test_orchestrator_integration
        ]
        
        passed = 0
        failed = 0
        
        for test_func in test_functions:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.console.print(f"[red]❌ {test_func.__name__} crashed: {e}[/red]")
                failed += 1
        
        # Display summary
        self.display_test_summary(passed, failed)
        
        return {
            "total_tests": len(test_functions),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_functions) * 100,
            "results": self.test_results
        }
    
    def display_test_summary(self, passed: int, failed: int):
        """Display test summary table"""
        self.console.print("\n[bold blue]📊 Test Summary[/bold blue]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test Name", style="cyan", width=30)
        table.add_column("Status", width=12)
        table.add_column("Duration", width=10)
        table.add_column("Details", width=40)
        
        for result in self.test_results:
            status_color = "green" if result["status"] == "PASSED" else "red"
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            
            table.add_row(
                result["test_name"],
                f"[{status_color}]{status_icon} {result['status']}[/{status_color}]",
                f"{result['duration']:.2f}s",
                result["details"][:37] + "..." if len(result["details"]) > 40 else result["details"]
            )
        
        self.console.print(table)
        
        # Overall summary
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        summary_text = f"Tests: {total} | Passed: {passed} | Failed: {failed} | Success Rate: {success_rate:.1f}%"
        summary_color = "green" if failed == 0 else "yellow" if success_rate >= 70 else "red"
        
        panel = Panel(
            summary_text,
            title="[bold]Overall Results[/bold]",
            border_style=summary_color
        )
        self.console.print(panel)
        
        # Recommendations
        if failed > 0:
            self.console.print("\n[bold yellow]⚠️  Recommendations:[/bold yellow]")
            if any("environment" in r["test_name"].lower() for r in self.test_results if r["status"] == "FAILED"):
                self.console.print("   - Check your .env file configuration")
                self.console.print("   - Verify API keys are correctly set")
            if any("agent" in r["test_name"].lower() for r in self.test_results if r["status"] == "FAILED"):
                self.console.print("   - Ensure all dependencies are installed")
                self.console.print("   - Check internet connectivity for API calls")
        else:
            self.console.print(f"\n[bold green]🎉 All tests passed! System is ready to use.[/bold green]")