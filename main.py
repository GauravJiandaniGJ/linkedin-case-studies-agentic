#!/usr/bin/env python3
"""
LinkedIn AI Agent - Converts technical case studies to layman-friendly posts
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.orchestrator import AgentOrchestrator
from src.agents.testing import TestingAgent

load_dotenv()

def get_case_study() -> str:
    """Get case study input from user"""
    console = Console()
    
    console.print("\n[bold blue]LinkedIn AI Agent System[/bold blue]")
    console.print("Convert technical case studies into crispy, layman-friendly LinkedIn posts with images!\n")
    
    # Check for case study file
    # Read case study from file
    case_study_path = os.path.join(Path(__file__).parent, "case_study.txt")
    with open(case_study_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    """Main application entry point"""
    console = Console()
    
    try:
        # Check for test mode
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            console.print("[bold blue]🧪 Running Test Suite[/bold blue]")
            tester = TestingAgent()
            results = tester.run_all_tests()
            
            if results["success_rate"] >= 80:
                console.print(f"\n[bold green]✅ System ready! Success rate: {results['success_rate']:.1f}%[/bold green]")
                sys.exit(0)
            else:
                console.print(f"\n[bold red]❌ System not ready. Success rate: {results['success_rate']:.1f}%[/bold red]")
                sys.exit(1)
        
        # Display current configuration
        provider = os.getenv("LLM_PROVIDER", "openai")
        image_provider = os.getenv("IMAGE_PROVIDER", "openai")
        console.print(f"\n[dim]Using LLM Provider: {provider}[/dim]")
        console.print(f"[dim]Using Image Provider: {image_provider}[/dim]")
        
        # Check for image generation option
        enable_images = True
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("STABILITY_API_KEY"):
            console.print("[yellow]⚠️  No image generation API keys found. Running in text-only mode.[/yellow]")
            enable_images = False
        enable_images = False # Bypassing interactive prompt for image generation
        
        # Get case study
        case_study = get_case_study()
        
        if not case_study.strip():
            console.print("[red]No case study provided. Exiting.[/red]")
            return
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(enable_images=enable_images)
        
        # Run the workflow
        final_post = orchestrator.run(case_study)
        
        # Display final result
        console.print("\n[bold green]🎉 Final LinkedIn Post Ready![/bold green]\n")
        console.print(final_post)
        
        # Save option
        if True: # Always save the final post and assets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linkedin_post_{provider}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"LinkedIn AI Agent Results\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"LLM Provider: {provider}\n")
                if enable_images:
                    f.write(f"Image Provider: {image_provider}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Original Case Study:\n{case_study}\n\n")
                f.write(f"Final LinkedIn Post:\n{final_post}\n\n")
                
                f.write("Text Iteration History:\n")
                for item in orchestrator.iteration_history:
                    f.write(f"\nIteration {item['iteration']}:\n")
                    f.write(f"Post: {item['post']}\n")
                    f.write(f"Feedback: {item['feedback']}\n")
                    f.write(f"Understandable: {item['is_understandable']}\n")
                
                if enable_images and orchestrator.image_history:
                    f.write("\nImage Generation History:\n")
                    for item in orchestrator.image_history:
                        f.write(f"\nImage Iteration {item['iteration']}:\n")
                        f.write(f"File: {item['image_info']['filename']}\n")
                        f.write(f"Score: {item['analysis']['overall_score']}/10\n")
                        f.write(f"Acceptable: {item['analysis']['is_acceptable']}\n")
                        f.write(f"Feedback: {item['analysis']['feedback']}\n")
            
            console.print(f"[green]✅ Saved results to {filename}[/green]")
            
            # Show image file location if generated
            if enable_images and orchestrator.image_history:
                latest_image = orchestrator.image_history[-1]['image_info']
                console.print(f"[green]🖼️  Image saved: {latest_image['filepath']}[/green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise

if __name__ == "__main__":
    main()
