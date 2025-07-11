import os
from typing import List, Dict, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from dotenv import load_dotenv

load_dotenv()

class AgentOrchestrator:
    """Orchestrates the interaction between Writer and Critic agents"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.writer = WriterAgent(llm_provider)
        self.critic = CriticAgent(llm_provider)
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 5))
        self.console = Console()
        self.iteration_history: List[Dict] = []
    
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
        
        # Display summary
        self._display_summary()
        
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
    
    def _display_summary(self):
        """Display summary of all iterations"""
        self.console.print("\n[bold blue]📊 Workflow Summary[/bold blue]\n")
        
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
