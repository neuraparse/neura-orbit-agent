"""Main CLI interface for Neura-Orbit-Agent."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt

from ..core.agent_brain import NeuraOrbitAgent
from ..core.advanced_agent_brain import AdvancedAgentBrain
from ..utils.config import Config, load_config
from ..utils.logger import setup_logging, get_cli_logger
from ..utils.exceptions import NeuraOrbitError

console = Console()
logger = get_cli_logger()


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
@click.pass_context
def cli(ctx, config: Optional[Path], verbose: bool, debug: bool):
    """Neura-Orbit-Agent: Advanced AI Agent for Screen Monitoring and System Automation."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config:
            ctx.obj["config"] = Config.load_from_file(config)
        else:
            ctx.obj["config"] = load_config()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    log_config = ctx.obj["config"].logging
    
    setup_logging(
        log_level=log_level,
        log_file=log_config.file.get("path"),
        console_enabled=log_config.console.get("enabled", True),
        file_enabled=log_config.file.get("enabled", True),
        colorize=log_config.console.get("colorize", True)
    )
    
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("task_description")
@click.option(
    "--provider",
    help="Specific LLM provider to use (ollama, openai, anthropic)"
)
@click.option(
    "--model",
    help="Specific model to use"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Analyze task without executing"
)
@click.option(
    "--advanced",
    is_flag=True,
    help="Use 2025 advanced agent brain with full intelligence"
)
@click.pass_context
def execute(ctx, task_description: str, provider: Optional[str], model: Optional[str], dry_run: bool, advanced: bool):
    """Execute a natural language task."""
    
    async def run_task():
        config = ctx.obj["config"]
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # Initialize agent
                task = progress.add_task("Initializing agent...", total=None)
                if advanced:
                    console.print("🚀 [bold blue]Using 2025 Advanced Agent Brain[/bold blue]")
                    agent = AdvancedAgentBrain(config)
                else:
                    agent = NeuraOrbitAgent(config)
                
                if dry_run:
                    # Just analyze the task
                    progress.update(task, description="Analyzing task...")
                    analysis = await agent.analyze_screen(
                        prompt=f"The user wants to: {task_description}. Analyze what would be needed to complete this task."
                    )
                    
                    progress.update(task, description="Complete", completed=True)
                    
                    console.print(Panel(
                        analysis,
                        title="Task Analysis",
                        border_style="blue"
                    ))
                else:
                    # Execute the task
                    progress.update(task, description="Executing task...")
                    if advanced:
                        result = await agent.execute_intelligent_task(task_description)
                    else:
                        result = await agent.execute_task(task_description)

                    progress.update(task, description="Complete", completed=True)
                    
                    if result.success:
                        console.print(Panel(
                            f"✅ {result.message}\n\nExecution time: {result.execution_time:.2f}s",
                            title="Task Completed",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel(
                            f"❌ {result.message}\n\nExecution time: {result.execution_time:.2f}s",
                            title="Task Failed",
                            border_style="red"
                        ))
                
                await agent.close()
                
        except NeuraOrbitError as e:
            console.print(f"[red]Agent error: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if ctx.obj["debug"]:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(run_task())


@cli.command()
@click.option(
    "--interval",
    "-i",
    type=float,
    default=5.0,
    help="Monitoring interval in seconds"
)
@click.option(
    "--duration",
    "-d",
    type=int,
    help="Monitoring duration in seconds (infinite if not specified)"
)
@click.pass_context
def monitor(ctx, interval: float, duration: Optional[int]):
    """Start screen monitoring mode."""
    
    async def run_monitor():
        config = ctx.obj["config"]
        
        try:
            agent = NeuraOrbitAgent(config)
            
            console.print(Panel(
                f"Starting screen monitoring...\n"
                f"Interval: {interval}s\n"
                f"Duration: {'Infinite' if duration is None else f'{duration}s'}\n"
                f"Press Ctrl+C to stop",
                title="Monitor Mode",
                border_style="blue"
            ))
            
            # Monitoring callback
            async def monitor_callback(data):
                timestamp = data["timestamp"]
                analysis = data["analysis"]
                
                console.print(f"\n[dim]{timestamp}[/dim]")
                console.print(Panel(
                    analysis,
                    title="Screen Analysis",
                    border_style="cyan"
                ))
            
            # Start monitoring
            await agent.start_monitoring(interval=interval, callback=monitor_callback)
            
            # Wait for duration or until interrupted
            if duration:
                await asyncio.sleep(duration)
                await agent.stop_monitoring()
            else:
                # Wait indefinitely
                try:
                    while agent.is_monitoring:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Stopping monitoring...[/yellow]")
                    await agent.stop_monitoring()
            
            await agent.close()
            console.print("[green]Monitoring stopped[/green]")
            
        except Exception as e:
            console.print(f"[red]Monitoring error: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_monitor())


@cli.command()
@click.option(
    "--save",
    "-s",
    type=click.Path(path_type=Path),
    help="Save screenshot to file"
)
@click.pass_context
def screenshot(ctx, save: Optional[Path]):
    """Take a screenshot and optionally analyze it."""
    
    async def take_screenshot():
        config = ctx.obj["config"]
        
        try:
            agent = NeuraOrbitAgent(config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Taking screenshot...", total=None)
                
                if save:
                    # Save to file
                    filepath = agent.screen_capture.capture_to_file(save)
                    progress.update(task, description="Analyzing screenshot...")
                    
                    analysis = await agent.analyze_screen()
                    
                    progress.update(task, description="Complete", completed=True)
                    
                    console.print(Panel(
                        f"Screenshot saved to: {filepath}\n\n{analysis}",
                        title="Screenshot Analysis",
                        border_style="green"
                    ))
                else:
                    # Just analyze
                    analysis = await agent.analyze_screen()
                    
                    progress.update(task, description="Complete", completed=True)
                    
                    console.print(Panel(
                        analysis,
                        title="Screen Analysis",
                        border_style="blue"
                    ))
            
            await agent.close()
            
        except Exception as e:
            console.print(f"[red]Screenshot error: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(take_screenshot())


@cli.command()
@click.pass_context
def status(ctx):
    """Show agent status and health check."""
    
    async def check_status():
        config = ctx.obj["config"]
        
        try:
            agent = NeuraOrbitAgent(config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Checking status...", total=None)
                health = await agent.health_check()
                progress.update(task, description="Complete", completed=True)
            
            # Create status table
            table = Table(title="Agent Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")
            
            # Overall status
            overall_status = "✅ Healthy" if health.get("overall", False) else "❌ Issues"
            table.add_row("Overall", overall_status, "")
            
            # Screen capture
            screen_status = "✅ Working" if health.get("screen_capture", False) else "❌ Failed"
            table.add_row("Screen Capture", screen_status, "")
            
            # System controller
            system_status = "✅ Working" if health.get("system_controller", False) else "❌ Failed"
            table.add_row("System Controller", system_status, "")
            
            # LLM providers
            llm_health = health.get("llm_providers", {})
            for provider, status in llm_health.items():
                provider_status = "✅ Connected" if status else "❌ Disconnected"
                table.add_row(f"LLM ({provider})", provider_status, "")
            
            console.print(table)
            
            # Show configuration summary
            console.print(Panel(
                f"Platform: {agent.system_controller.platform}\n"
                f"Screen Size: {agent.screen_capture.screen_size}\n"
                f"Default LLM: {config.llm.default_provider}\n"
                f"Monitoring: {'Active' if agent.is_monitoring else 'Inactive'}",
                title="Configuration",
                border_style="blue"
            ))
            
            await agent.close()
            
        except Exception as e:
            console.print(f"[red]Status check error: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(check_status())


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode."""
    
    async def run_interactive():
        config = ctx.obj["config"]
        
        try:
            agent = NeuraOrbitAgent(config)
            
            console.print(Panel(
                "Welcome to Neura-Orbit-Agent Interactive Mode!\n\n"
                "Commands:\n"
                "- Type any task description to execute it\n"
                "- 'analyze' - Analyze current screen\n"
                "- 'screenshot' - Take a screenshot\n"
                "- 'status' - Show agent status\n"
                "- 'history' - Show task history\n"
                "- 'quit' or 'exit' - Exit interactive mode",
                title="Interactive Mode",
                border_style="green"
            ))
            
            while True:
                try:
                    command = Prompt.ask("\n[bold cyan]neura-orbit>[/bold cyan]")
                    
                    if command.lower() in ["quit", "exit"]:
                        break
                    elif command.lower() == "analyze":
                        with console.status("Analyzing screen..."):
                            analysis = await agent.analyze_screen()
                        console.print(Panel(analysis, title="Screen Analysis", border_style="blue"))
                    elif command.lower() == "screenshot":
                        with console.status("Taking screenshot..."):
                            analysis = await agent.analyze_screen()
                        console.print(Panel(analysis, title="Screenshot Analysis", border_style="blue"))
                    elif command.lower() == "status":
                        with console.status("Checking status..."):
                            health = await agent.health_check()
                        status_text = "✅ All systems operational" if health.get("overall") else "❌ Some issues detected"
                        console.print(Panel(status_text, title="Status", border_style="green" if health.get("overall") else "red"))
                    elif command.lower() == "history":
                        history = await agent.get_task_history()
                        if history:
                            table = Table(title="Task History")
                            table.add_column("Time", style="dim")
                            table.add_column("Task")
                            table.add_column("Result", style="green")
                            
                            for task in history[-10:]:  # Show last 10 tasks
                                result_icon = "✅" if task["result"] else "❌"
                                table.add_row(
                                    f"{task['timestamp']:.0f}",
                                    task["description"][:50] + "..." if len(task["description"]) > 50 else task["description"],
                                    f"{result_icon} {task['message'][:30]}..."
                                )
                            console.print(table)
                        else:
                            console.print("[dim]No task history[/dim]")
                    else:
                        # Execute as task
                        with console.status(f"Executing: {command}"):
                            result = await agent.execute_task(command)
                        
                        if result.success:
                            console.print(Panel(
                                f"✅ {result.message}",
                                title="Task Completed",
                                border_style="green"
                            ))
                        else:
                            console.print(Panel(
                                f"❌ {result.message}",
                                title="Task Failed",
                                border_style="red"
                            ))
                
                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'quit' or 'exit' to leave interactive mode[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            
            await agent.close()
            console.print("[green]Goodbye![/green]")
            
        except Exception as e:
            console.print(f"[red]Interactive mode error: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_interactive())


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
