"""Interactive CLI for AI Chat Assistant"""

import asyncio
from typing import Optional
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import Config, Provider
from .conversation import ConversationHistory
from .task_router import TaskRouter, TaskType
from .providers.base import Message


class ChatCLI:
    """Interactive chat CLI with rich formatting"""

    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.conversation = ConversationHistory(config.history_window_size)
        self.conversation.clear_history()  # Ensure fresh start
        self.router = TaskRouter(config)
        self.current_provider: Optional[Provider] = None
        self.stream_mode = "sync"  # Default to sync mode
        self.multi_mode = False  # Default to single-line mode
        self._current_stream_task: Optional[asyncio.Task] = None  # Track async tasks
        self.record_enabled = False  # For /record command

    def _display_providers_table(self):
        """Show available providers in a table"""
        provider_table = Table(title="Available Providers")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Status", style="green")
        provider_table.add_column("Default Model", style="yellow")

        for provider in Provider:
            if self.config.has_api_key(provider):
                info = self.router.get_provider_info(provider)
                provider_table.add_row(
                    provider.value.title(),
                    "✅ Available",
                    info.get("default_model", "Unknown"),
                )
            else:
                provider_table.add_row(provider.value.title(), "❌ No API Key", "-")

        self.console.print(provider_table)
        self.console.print()

    def display_welcome(self):
        """Display welcome message and available providers"""
        welcome_text = """
# AI Chat Assistant

Multi-provider AI assistant with model configuration support.

## Available Features:
- 📚 Conversation History
- ⚡ Multiple Provider Support
- 🛠️ Model Configuration
- 💰 Cost Tracking
        """

        self.console.print(
            Panel(Markdown(welcome_text), title="Welcome", border_style="blue")
        )
        self._display_providers_table()

    async def handle_user_input(self, user_input: str) -> bool:
        """Handle user input and return False if should exit"""
        if not user_input.strip():  # Skip empty input
            return True
        if user_input.startswith("/"):
            return await self._handle_command(user_input)
        await self._process_chat_message(user_input)
        return True

    async def _handle_command(self, command: str) -> bool:
        """Handle CLI commands"""
        cmd = command.split()[0].lower()

        if cmd in ["/quit", "/exit"]:
            self.console.print("[green]Goodbye! 👋[/green]")
            return False

        if cmd == "/model":
            self._handle_models_command()
        elif cmd == "/provider":
            self._handle_provider_command(command)
        elif cmd == "/task":
            self._handle_task_command(command)
        elif cmd == "/stat":
            self._handle_stat_command()
        elif cmd == "/help":
            self._handle_help_command()
        elif cmd == "/mode":
            self._handle_mode_command(command)
        elif cmd == "/multi":
            self._handle_multi_command(command)
        elif cmd == "/record":
            self._handle_record_command(command)
        else:
            self.console.print(
                Panel(f"Unknown command: {cmd}", title="Error", border_style="red")
            )
            self._handle_help_command()

        return True  # Continue CLI loop for all commands except quit/exit

    def _handle_task_command(self, command: str):
        """Handle /task command to switch task types"""
        parts = command.split()
        if len(parts) > 1:
            task_type = parts[1].lower()
            if task_type in ["code", "coding"]:
                self.router.current_task_type = TaskType.CODE
                self.console.print("[green]Switched to coding tasks[/green]")
            elif task_type in ["learn", "learning", "knowledge"]:
                self.router.current_task_type = TaskType.KNOWLEDGE
                self.console.print("[green]Switched to learning tasks[/green]")
            elif task_type in ["write", "writing"]:
                self.router.current_task_type = TaskType.WRITE
                self.console.print("[green]Switched to writing tasks[/green]")
            else:
                self.console.print(
                    Panel(
                        f"Invalid task type: {task_type}",
                        title="Error",
                        border_style="red",
                    )
                )
        else:
            current_task = self.router.current_task_type
            if current_task == TaskType.CODE:
                current = "coding"
            elif current_task == TaskType.KNOWLEDGE:
                current = "learning"
            elif current_task == TaskType.WRITE:
                current = "writing"
            else:
                current = "unknown"
            self.console.print(f"Current task type: {current}")

    def _handle_stat_command(self):
        """Handle /stat command to show accurate conversation statistics"""
        stat_table = Table(title="Conversation Statistics")
        stat_table.add_column("Metric", style="cyan")
        stat_table.add_column("Value", style="yellow")

        stats = self.conversation.get_statistics()

        stat_table.add_row("Completed Chat Turns", str(stats["total_turns"]))
        stat_table.add_row("Input Tokens", str(stats["prompt_tokens"]))
        stat_table.add_row("Output Tokens", str(stats["completion_tokens"]))
        stat_table.add_row("Total Tokens", str(stats["total_tokens"]))
        stat_table.add_row(
            "Current Provider",
            self.current_provider.value.title() if self.current_provider else "None",
        )

        self.console.print(stat_table)

    def _handle_help_command(self):
        """Handle /help command to show available commands"""
        help_text = """
Available Commands:
- /help - Show this help message
- /model - List available models and show current model
- /provider [name] - Switch provider (claude/deepseek/gemini)
- /task [type] - Switch task type (code/learning)
- /stat - Show usage statistics
- /mode [sync|async] - Set streaming mode (default: sync)
- /multi [on/off] - Toggle multi-line input mode (default: off)
- /record [on/off] - Save responses to Markdown files (default: off)
- /exit - Quit the application
        """
        self.console.print(Panel(help_text, title="Help", border_style="blue"))

    def _handle_models_command(self):
        """Handle /models command"""
        provider = self.current_provider
        if not provider or not self.config.has_api_key(provider):
            self.console.print("[yellow]Select a provider first[/yellow]")
            return

        models = self.config.get_provider_models(provider)
        if not models:
            self.console.print(
                f"[yellow]No models for {provider.value.title()}[/yellow]"
            )
            return

        model_table = Table(title=f"{provider.value.title()} Models")
        model_table.add_column("Name", style="cyan")
        model_table.add_column("Supports Chat", style="green")
        model_table.add_column("Supports Coding", style="blue")

        for model in models:
            model_table.add_row(
                model.get("name", "Unknown"),
                "✅" if model["supports_chat"] else "❌",
                "✅" if model["supports_coding"] else "❌",
            )

        self.console.print(model_table)
        self.console.print(
            f"Default: [bold yellow]{self.config.get_default_model(provider)}[/bold yellow]"
        )

        # Show current task type and model
        current_task_type = self.router.current_task_type
        current_model = self.config.get_model_for_provider_and_task(
            provider, current_task_type
        )
        self.console.print(
            f"Current task: [bold]{current_task_type.value}[/bold] → model: [bold yellow]{current_model}[/bold yellow]"
        )

    def _handle_provider_command(self, command: str):
        """Handle /provider command"""
        parts = command.split()
        if len(parts) > 1:
            provider_name = parts[1].lower()
            try:
                provider = Provider(provider_name)
                if self.config.has_api_key(provider):
                    # Clear any pending async operations
                    if (
                        hasattr(self, "_current_stream_task")
                        and self._current_stream_task is not None
                    ):
                        self._current_stream_task.cancel()

                    self.current_provider = provider
                    # Reset conversation history for new provider
                    self.conversation.clear_history()
                    self.console.print(
                        f"[green]Switched to {provider.value.title()}[/green]"
                    )
                else:
                    self.console.print(
                        f"[red]No API key for {provider.value.title()}[/red]"
                    )
            except ValueError:
                self.console.print(f"[red]Invalid provider: {provider_name}[/red]")
        else:
            if self.current_provider:
                current = self.current_provider.value.title()
                self.console.print(f"Current provider: {current}")
            else:
                self.console.print("No provider selected. Available providers:")
            self._display_providers_table()

    async def _process_chat_message(self, user_input: str):
        """Process a chat message and get response"""
        try:
            if not self.current_provider:
                self.console.print(
                    Panel(
                        "Please select a provider first",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            provider = self.router.get_available_providers()[self.current_provider]
            model = self.config.get_model_for_provider_and_task(
                self.current_provider, self.router.current_task_type
            )

            messages = self.conversation.get_messages_for_api(
                include_system=True,
                task_type=self.router.current_task_type
            )
            
            # Format user input based on task type
            if self.router.current_task_type == TaskType.WRITE:
                formatted_user_input = f"Please help me polish the following English writing drafts for clarity, grammar, and natural tone. Keep the author's voice as much as possible:\n\n{user_input}"
            elif self.router.current_task_type == TaskType.CODE:
                formatted_user_input = f"Please help me analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code: \n\n{user_input}"
            elif self.router.current_task_type == TaskType.KNOWLEDGE:
                formatted_user_input = f"Please teach me the concept step by step: \n\n{user_input}"
            else:
                formatted_user_input = user_input
                
            messages.append(Message(role="user", content=formatted_user_input))

            if self.stream_mode == "async":
                if provider.supports_async and hasattr(provider, "stream_completion"):
                    await self._stream_response(provider, messages, model, user_input)
                else:
                    self.console.print(
                        "[yellow]Async mode not supported by provider, falling back to sync[/yellow]"
                    )
                    await self._get_basic_response(
                        provider, messages, model, user_input
                    )
            else:
                await self._get_basic_response(provider, messages, model, user_input)

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def _handle_mode_command(self, command: str):
        """Handle /mode command to switch streaming modes"""
        parts = command.split()
        if len(parts) > 1:
            mode = parts[1].lower()
            if mode in ["sync", "async"]:
                self.stream_mode = mode
                self.console.print(f"[green]Switched to {mode} mode[/green]")
            else:
                self.console.print(
                    Panel(
                        f"Invalid mode: {mode} (use sync or async)",
                        title="Error",
                        border_style="red",
                    )
                )
        else:
            self.console.print(f"Current streaming mode: {self.stream_mode}")

    def _handle_multi_command(self, command: str):
        """Handle /multi command to toggle multi-line input mode"""
        parts = command.split()
        if len(parts) > 1:
            mode = parts[1].lower()
            if mode == "on":
                self.multi_mode = True
                self.console.print("[green]Enabled multi-line input mode[/green]")
            elif mode == "off":
                self.multi_mode = False
                self.console.print("[green]Disabled multi-line input mode[/green]")
            else:
                self.console.print(
                    Panel(
                        f"Invalid mode: {mode} (use on or off)",
                        title="Error",
                        border_style="red",
                    )
                )
        else:
            status = "on" if self.multi_mode else "off"
            self.console.print(
                Panel(
                    f"Current multi-line input mode: [bold yellow]{status}[/bold yellow]\n"
                    f"Usage: /multi [on|off] to toggle",
                    title="Multi-line Mode",
                    border_style="blue",
                )
            )

    def _handle_record_command(self, command: str):
        """Handle /record command to toggle response recording"""
        parts = command.split()
        if len(parts) > 1:
            mode = parts[1].lower()
            if mode == "on":
                self.record_enabled = True
                self.console.print("[green]Recording enabled[/green]")
            elif mode == "off":
                self.record_enabled = False
                self.console.print("[green]Recording disabled[/green]")
            else:
                self.console.print(
                    Panel(
                        f"Invalid mode: {mode} (use on or off)",
                        title="Error",
                        border_style="red",
                    )
                )
        else:
            status = "on" if self.record_enabled else "off"
            self.console.print(f"Recording is currently {status}")

    async def _stream_response(self, provider, messages, model, user_input):
        """Handle streaming response with markdown formatting"""
        self.console.print(f"\n[bold blue]Assistant ({model}):[/bold blue]")
        response_text = ""
        usage_data = None

        if self.stream_mode == "async":
            # Async mode - print chunks immediately as plain text
            try:
                response_text = ""
                usage_data = None
                async for chunk in provider.stream_completion(messages, model):
                    if isinstance(chunk, dict):
                        if "text" in chunk:
                            response_text += chunk["text"]
                            self.console.print(chunk["text"], end="", markup=False)
                        if "usage" in chunk:
                            usage_data = chunk["usage"]
                        if "error" in chunk:
                            self.console.print(chunk["error"], style="red")
                    else:
                        self.console.print(
                            f"[red]Invalid chunk format: {type(chunk)}[/red]"
                        )

                self.console.print()
                self._add_to_history(user_input, response_text, model, usage_data)
            except Exception as e:
                self.console.print(
                    Panel(f"Stream error: {str(e)}", title="Error", border_style="red")
                )

    async def _get_basic_response(self, provider, messages, model, user_input):
        """Get non-streaming response"""
        with self.console.status(f"[bold blue]Thinking...[/bold blue]"):
            response = await provider.chat_completion(messages, model)
            self.console.print(f"\n[bold blue]Assistant ({model}):[/bold blue]")

            # Handle different response formats
            if hasattr(response, "content"):
                content = response.content
                usage = response.usage if hasattr(response, "usage") else None
            elif isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("text", "")
                usage = response.get("usage")
            else:
                content = str(response)
                usage = None

            self.console.print(Markdown(content))
            self._add_to_history(user_input, content, model, usage)

    def _add_to_history(
        self, user_input: str, response: str, model: str, usage: Optional[dict] = None
    ):
        """Add conversation to history with API usage data"""
        self.conversation.add_turn(
            user_input,
            response,
            self.current_provider.value.lower() if self.current_provider else "unknown",
            model,
            usage=usage,
        )

        # Save response to file if recording is enabled
        if self.record_enabled:
            try:
                # Create answers directory with date-based subdirectory
                date_str = datetime.now().strftime("%Y%m%d")
                answers_dir = Path("answers") / date_str
                answers_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename: %H%M_{first 12 chars of user question}.md
                time_str = datetime.now().strftime("%H%M")
                safe_question = user_input[:12].replace("/", "_").replace("\\", "_")
                filename = f"{time_str}_{safe_question}.md"
                filepath = answers_dir / filename

                # Write both question and answer in UTF-8 encoding
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Q:\n {user_input}\n\nA:\n {response}\n")
            except Exception as e:
                self.console.print(f"[red]Error saving response: {e}[/red]")

    async def run(self):
        """Run the interactive CLI"""
        self.display_welcome()
        while True:
            try:
                if self.multi_mode:
                    self.console.print(
                        "\n[bold green]You[/bold green] (multi-line mode, press Enter twice to submit):"
                    )
                    buffer = []
                    empty_count = 0

                    while True:
                        line = input()
                        if not line.strip():  # Empty or whitespace line
                            empty_count += 1
                            if (
                                empty_count >= 2 and buffer
                            ):  # Two empty lines with content
                                break
                        else:
                            empty_count = 0
                            if line.strip():  # Only add non-empty lines
                                buffer.append(line.strip())

                    user_input = "\n".join(buffer)
                else:
                    self.console.print("\n[bold green]You[/bold green]:", end=" ")
                    user_input = input().strip()

                if user_input:
                    should_continue = await self.handle_user_input(user_input)
                    if not should_continue:
                        break

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[green]Goodbye! 👋[/green]")
                break


async def async_main(
    config: Optional[Path],
    provider: Optional[str],
    task: Optional[str],
    mode: Optional[str],
    multi: bool,
    record: bool,
):
    """Main async entry point"""
    app_config = Config.from_file(config)

    if not any(app_config.has_api_key(p) for p in Provider):
        raise RuntimeError("No API keys configured")

    cli = ChatCLI(app_config)
    if provider:
        try:
            cli.current_provider = Provider(provider)
        except ValueError:
            pass

    # Set initial task type if provided
    if task:
        if task == "code":
            cli.router.current_task_type = TaskType.CODE
        elif task == "learning":
            cli.router.current_task_type = TaskType.KNOWLEDGE
        elif task == "write":
            cli.router.current_task_type = TaskType.WRITE

    # Set initial streaming mode if provided
    if mode:
        cli.stream_mode = mode

    # Set initial modes from flags
    cli.multi_mode = multi
    cli.record_enabled = record

    await cli.run()


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)

@click.option(
    "--provider",
    "-p",
    type=click.Choice(["claude", "deepseek", "gemini"]),
    help="Preferred provider",
)

@click.option(
    "--task",
    "-t",
    type=click.Choice(["code", "learn", "write"]),
    help="Set the initial task type (code, learn, or write)",
)

@click.option(
    "--mode",
    "-m",
    type=click.Choice(["sync", "async"]),
    help="Set the initial streaming mode (sync or async)",
)

@click.option(
    "--multi",
    "-M",
    type=click.Choice(["on", "off"]),
    help="Enable multi-line input mode at start (on or off)",
)

@click.option(
    "--record",
    "-r",
    type=click.Choice(["on", "off"]),
    help="Enable response recording at start (on or off)",
)

def main(
    config: Optional[Path],
    provider: Optional[str],
    task: Optional[str],
    mode: Optional[str],
    multi: str,
    record: str,
):
    """AI Chat Assistant CLI"""
    # Convert multi/record string values to booleans
    multi_bool = multi == 'on'
    record_bool = record == 'on'
    
    asyncio.run(async_main(config, provider, task, mode, multi_bool, record_bool))


if __name__ == "__main__":
    main()
