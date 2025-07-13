import os
from typing import List, Dict, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.agents.image_generator import ImageGeneratorAgent
from src.agents.image_critic import ImageCriticAgent
from dotenv import load_dotenv

load_dotenv()

class AgentOrchestrator:
    """Orchestrates the interaction between Writer and Critic agents"""
    
    def __init__(self, llm_provider: Optional[str] = None, enable_images: bool = True):
        self.console = Console()
        self.writer = WriterAgent(llm_provider)
        self.critic = CriticAgent(llm_provider)
        self.enable_images = enable_images
        
        # Initialize image agents if enabled
        if self.enable_images:
            try:
                self.image_generator = ImageGeneratorAgent(llm_provider)
                self.image_critic = ImageCriticAgent(llm_provider)
            except ValueError as e:
                # Show specific provider error messages
                self.console.print(f"[yellow]⚠️  Image generation disabled: {e}[/yellow]")
                self.enable_images = False
            except Exception as e:
                self.console.print(f"[yellow]⚠️  Image generation disabled due to error: {e}[/yellow]")
                self.enable_images = False
        
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 5))
        self.max_image_iterations = int(os.getenv("MAX_IMAGE_ITERATIONS", 3))
        self.iteration_history: List[Dict] = []
        self.image_history: List[Dict] = []
    
    def run(self, case_study: str) -> str:
        """
        Run the agent workflow
        
        Args:
            case_study: The technical case study to convert
            
        Returns:
            Final LinkedIn post
        """
        self.console.print("\n[bold blue]🚀 Starting AI Agent Workflow[/bold blue]\n")
        
        # Initial post creation
        self.console.print("[yellow]📝 Writer creating initial post...[/yellow]")
        current_post = self.writer.write_initial_post(case_study)
        
        self._display_post(current_post, "Initial Post", iteration=0)
        
        # Iterative refinement loop
        for iteration in range(1, self.max_iterations + 1):
            self.console.print(f"\n[cyan]🔄 Iteration {iteration}/{self.max_iterations}[/cyan]")
            
            # Critic analysis
            self.console.print("[yellow]👀 Critic analyzing post...[/yellow]")
            is_understandable, feedback = self.critic.analyze_post(current_post)
            
            self._display_feedback(feedback, is_understandable)
            
            # Store iteration data
            self.iteration_history.append({
                "iteration": iteration,
                "post": current_post,
                "feedback": feedback,
                "is_understandable": is_understandable
            })
            
            # Check if we're done
            if is_understandable:
                self.console.print("\n[bold green]✅ Post is now understandable to laymen![/bold green]")
                break
            
            # Refine post
            self.console.print("[yellow]✏️  Writer refining post based on feedback...[/yellow]")
            current_post = self.writer.refine_post(current_post, feedback)
            
            self._display_post(current_post, f"Refined Post (Iteration {iteration})", iteration)
        
        else:
            # Max iterations reached
            self.console.print("\n[bold yellow]⚠️  Maximum iterations reached[/bold yellow]")
        
        # Generate image if enabled
        image_info = None
        if self.enable_images:
            image_info = self._generate_and_validate_image(current_post)
        
        # Display summary
        self._display_summary(image_info)
        
        return current_post
    
    def _display_post(self, post: str, title: str, iteration: int):
        """Display post in a formatted panel"""
        word_count = len(post.split())
        panel = Panel(
            post,
            title=f"[bold]{title}[/bold]",
            subtitle=f"Word count: {word_count}",
            border_style="green" if iteration == 0 else "blue"
        )
        self.console.print(panel)
    
    def _display_feedback(self, feedback: str, is_understandable: bool):
        """Display critic feedback"""
        status = "✅ Understandable" if is_understandable else "❌ Not Understandable"
        panel = Panel(
            feedback,
            title=f"[bold]Critic Feedback - {status}[/bold]",
            border_style="green" if is_understandable else "red"
        )
        self.console.print(panel)
    
    def _generate_and_validate_image(self, final_post: str) -> Optional[Dict]:
        """Generate and validate image for the final LinkedIn post"""
        self.console.print("\n[bold purple]🎨 Generating Image for LinkedIn Post[/bold purple]")
        
        try:
            # Generate initial image
            self.console.print("[yellow]🖼️  Generating image...[/yellow]")
            image_info = self.image_generator.generate_image(final_post)
            
            self._display_image_info(image_info, "Generated Image")
            
            # Validate image with critic
            current_image = image_info
            
            for iteration in range(1, self.max_image_iterations + 1):
                self.console.print(f"\n[cyan]🔍 Image Validation {iteration}/{self.max_image_iterations}[/cyan]")
                
                analysis = self.image_critic.analyze_image(
                    current_image["filepath"],
                    final_post,
                    current_image["prompt"]
                )
                
                self._display_image_analysis(analysis)
                
                # Store image iteration data
                self.image_history.append({
                    "iteration": iteration,
                    "image_info": current_image,
                    "analysis": analysis
                })
                
                # Check if image is acceptable
                if analysis["is_acceptable"]:
                    self.console.print("\n[bold green]✅ Image meets quality standards![/bold green]")
                    break
                
                # Generate improved image if not the last iteration
                if iteration < self.max_image_iterations:
                    self.console.print("[yellow]🔄 Generating improved image...[/yellow]")
                    
                    # Get improvement suggestions and generate new image
                    improved_prompt = self.image_critic.suggest_improvements(
                        analysis, current_image["prompt"]
                    )
                    
                    # Generate new image (using original post content, as improved_prompt is just for reference)
                    current_image = self.image_generator.generate_image(final_post)
                    self._display_image_info(current_image, f"Improved Image (Iteration {iteration})")
            
            else:
                self.console.print("\n[bold yellow]⚠️  Maximum image iterations reached[/bold yellow]")
            
            return current_image
            
        except Exception as e:
            self.console.print(f"\n[red]❌ Image generation failed: {e}[/red]")
            return None
    
    def _display_image_info(self, image_info: Dict, title: str):
        """Display image generation information"""
        info_text = f"""
Provider: {image_info['provider']}
Model: {image_info.get('model', 'Unknown')}
File: {image_info['filename']}
Path: {image_info['filepath']}

Prompt: {image_info['prompt'][:100]}...
"""
        panel = Panel(
            info_text.strip(),
            title=f"[bold]{title}[/bold]",
            border_style="purple"
        )
        self.console.print(panel)
    
    def _display_image_analysis(self, analysis: Dict):
        """Display image critic analysis"""
        status = "✅ Acceptable" if analysis["is_acceptable"] else "❌ Needs Improvement"
        score = analysis["overall_score"]
        
        panel = Panel(
            analysis["feedback"],
            title=f"[bold]Image Analysis - {status} (Score: {score}/10)[/bold]",
            border_style="green" if analysis["is_acceptable"] else "red"
        )
        self.console.print(panel)
    
    def _display_summary(self, image_info: Optional[Dict] = None):
        """Display summary of all iterations"""
        self.console.print("\n[bold blue]📊 Workflow Summary[/bold blue]\n")
        
        # Text workflow summary
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Iteration", style="cyan", width=10)
        table.add_column("Status", width=15)
        table.add_column("Key Feedback", width=50)
        
        for item in self.iteration_history:
            status = "✅ Clear" if item["is_understandable"] else "❌ Unclear"
            # Extract first line of feedback for summary
            feedback_summary = item["feedback"].split('\n')[0][:47] + "..."
            table.add_row(
                str(item["iteration"]),
                status,
                feedback_summary
            )
        
        self.console.print(table)
        
        # Image workflow summary if images were generated
        if self.enable_images and self.image_history:
            self.console.print("\n[bold purple]🎨 Image Generation Summary[/bold purple]\n")
            
            image_table = Table(show_header=True, header_style="bold purple")
            image_table.add_column("Iteration", style="cyan", width=10)
            image_table.add_column("Score", width=10)
            image_table.add_column("Status", width=15)
            image_table.add_column("Key Issues", width=40)
            
            for item in self.image_history:
                analysis = item["analysis"]
                status = "✅ Good" if analysis["is_acceptable"] else "❌ Poor"
                score = f"{analysis['overall_score']}/10"
                
                # Extract key issues from feedback
                feedback_summary = analysis["feedback"].split('\n')[0][:37] + "..."
                
                image_table.add_row(
                    str(item["iteration"]),
                    score,
                    status,
                    feedback_summary
                )
            
            self.console.print(image_table)
        
        # Final deliverables summary
        self.console.print("\n[bold green]🎯 Final Deliverables[/bold green]")
        deliverables = []
        deliverables.append("✅ LinkedIn Post (60 words)")
        
        if image_info:
            deliverables.append(f"✅ Generated Image: {image_info['filename']}")
        elif self.enable_images:
            deliverables.append("❌ Image generation failed")
        else:
            deliverables.append("⚠️  Image generation disabled")
        
        for deliverable in deliverables:
            self.console.print(f"   {deliverable}")
