#!/usr/bin/env python3
"""
LinkedIn AI Agent - Converts technical case studies to layman-friendly posts
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.orchestrator import AgentOrchestrator

load_dotenv()

def get_case_study() -> str:
    """Get case study input from user"""
    console = Console()
    
    console.print("\n[bold blue]LinkedIn AI Agent System[/bold blue]")
    console.print("Convert technical case studies into crispy, layman-friendly LinkedIn posts!\n")
    
    # Check for case study file
    if Confirm.ask("Do you have a case study in a file?"):
        file_path = Prompt.ask("Enter the file path")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            console.print("Please enter the case study manually.\n")
    
    # Manual input
    console.print("Enter your case study (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    return '\n'.join(lines[:-1])  # Remove last empty line

def main():
    """Main application entry point"""
    console = Console()
    
    try:
        # Display current configuration
        provider = os.getenv("LLM_PROVIDER", "openai")
        console.print(f"\n[dim]Using LLM Provider: {provider}[/dim]")
        
        # Get case study
        case_study = get_case_study()
        
        if not case_study.strip():
            console.print("[red]No case study provided. Exiting.[/red]")
            return
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        
        # Run the workflow
        final_post = orchestrator.run(case_study)
        
        # Display final result
        console.print("\n[bold green]🎉 Final LinkedIn Post Ready![/bold green]\n")
        console.print(final_post)
        
        # Save option
        if Confirm.ask("\nWould you like to save the final post?"):
            filename = f"linkedin_post_{provider}_{len(orchestrator.iteration_history)}_iterations.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Case Study:\n{case_study}\n\n")
                f.write(f"Final Post:\n{final_post}\n\n")
                f.write("Iteration History:\n")
                for item in orchestrator.iteration_history:
                    f.write(f"\nIteration {item['iteration']}:\n")
                    f.write(f"Post: {item['post']}\n")
                    f.write(f"Feedback: {item['feedback']}\n")
                    f.write(f"Understandable: {item['is_understandable']}\n")
            console.print(f"[green]Saved to {filename}[/green]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise

if __name__ == "__main__":
    main()
