"""Interactive CLI for AI Chat Assistant"""

import asyncio
import re
import threading
from typing import Optional
from pathlib import Path
from datetime import datetime
import aiofiles

from .archive import AnswerArchive

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import ApplicationConfig
from .conversation import ConversationHistory
from .task_router import TaskRouter
from .providers.base import Message, ProviderAPIError
from .settings import TaskType, Provider
from .logger import setup_logging, get_logger


class ChatCLI:
    """Interactive chat CLI with rich formatting"""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.console = Console()
        self.logger = get_logger(__name__)  # Get logger for ChatCLI
        self.conversation = ConversationHistory(config.settings.history_window_size)
        self.conversation.clear_history()  # Ensure fresh start
        self.router = TaskRouter(config)
        self.current_task_type = TaskType.KNOWLEDGE  # Default to knowledge tasks
        self.current_provider: Optional[Provider] = None
        self.stream_mode = "sync"  # Default to sync mode
        self.multi_mode = False  # Default to single-line mode
        self._current_stream_task: Optional[asyncio.Task] = None  # Track async tasks
        self.record_enabled = False  # For /record command

        # Initialize answer archive with 14-day retention
        self.archive = AnswerArchive(retention_days=14)

        # Start background archiving thread
        self.archive_thread = threading.Thread(
            target=self.archive.start_periodic_archiving, daemon=True
        )
        self.archive_thread.start()

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
        elif cmd == "/search":
            self._handle_search_command(command)
        elif cmd == "/background":
            self._handle_background_command(command)
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
                self.current_task_type = TaskType.CODE
                self.console.print("[green]Switched to coding tasks[/green]")
            elif task_type in ["learn", "learning", "knowledge"]:
                self.current_task_type = TaskType.KNOWLEDGE
                self.console.print("[green]Switched to learning tasks[/green]")
            elif task_type in ["write", "writing"]:
                self.current_task_type = TaskType.WRITE
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
            current_task = self.current_task_type
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
- /provider [name] - Switch provider (claude/deepseek/gemini/kimi)
- /task [type] - Switch task type (code/learning/write)
- /stat - Show usage statistics
- /mode [sync|async] - Set streaming mode (default: sync)
- /multi [on/off] - Toggle multi-line input mode (default: off)
- /record [on/off] - Save responses to Markdown files (default: off)
- /background [on/off] - Enable/disable asynchronous answer saving (only when /record is on and not for WRITE tasks)
- /search [query] - Search archived answers
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
            self.logger.warning(
                f"Model command failed: No models for {provider.value.title()}"
            )
            return

        model_table = Table(title=f"{provider.value.title()} Models")
        model_table.add_column("Name", style="cyan")
        model_table.add_column("Type", style="magenta")  # New column for model type

        provider_config = self.config.settings.provider_configs.get(provider)

        # Determine current active model based on task type
        active_model = self.config.get_model_for_provider_and_task(
            provider, self.current_task_type
        )

        # Collect all relevant models to display, ensuring uniqueness
        display_models_set = set()
        if provider_config:
            display_models_set.add(provider_config.default_model)
            display_models_set.add(provider_config.knowledge_model)
            display_models_set.add(provider_config.code_model)
            for model_dict in provider_config.models:
                display_models_set.add(model_dict.get("name"))

        # Filter out None and empty strings, then sort
        display_models = sorted([m for m in list(display_models_set) if m])

        for model_name in display_models:
            is_knowledge_model = (
                provider_config and model_name == provider_config.knowledge_model
            )
            is_code_model = provider_config and model_name == provider_config.code_model

            model_type_indicator = []
            if is_knowledge_model:
                model_type_indicator.append("Knowledge")
            if is_code_model:
                model_type_indicator.append("Code")
            if not model_type_indicator:
                model_type_indicator.append(
                    "Default"
                )  # Fallback to Default if not explicitly Knowledge or Code

            row_style = ""
            if model_name == active_model:
                row_style = "bold yellow"

            model_table.add_row(
                model_name, ", ".join(model_type_indicator), style=row_style
            )

        self.console.print(model_table)
        self.console.print(
            f"Currently active model for '{self.current_task_type.value.title()}' tasks: [bold yellow]{active_model}[/bold yellow]"
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
            # Capture provider name here to ensure correct attribution in async tasks
            provider_name = self.current_provider.value.title()
            
            model = self.config.get_model_for_provider_and_task(
                self.current_provider, self.current_task_type
            )

            messages = self.conversation.get_messages_for_api(
                include_system=True, task_type=self.current_task_type
            )

            # Format user input based on task type
            if self.current_task_type == TaskType.WRITE:
                formatted_user_input = f"Please help me polish the following English writing drafts for clarity, grammar, and natural tone. Keep the author's voice as much as possible:\n\n{user_input}"
            elif self.current_task_type == TaskType.CODE:
                formatted_user_input = f"Please help me analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code: \n\n{user_input}"
            elif self.current_task_type == TaskType.KNOWLEDGE:
                formatted_user_input = (
                    f"Please teach me the concept step by step: \n\n{user_input}"
                )
            else:
                formatted_user_input = user_input

            messages.append(Message(role="user", content=formatted_user_input))

            # Get max_tokens from provider config
            provider_config = self.config.settings.provider_configs.get(self.current_provider)
            max_tokens = provider_config.max_tokens if provider_config else None

            if self.config.enable_async_answers:  # Background processing
                self.console.print(
                    "[bold yellow]Request sent to AI. Response will be recorded in the background.[/bold yellow]"
                )
                if self.stream_mode == "async":
                    if provider.supports_async and hasattr(
                        provider, "stream_completion"
                    ):
                        asyncio.create_task(
                            self._stream_response_and_save_background(
                                provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                            )
                        )
                    else:
                        asyncio.create_task(
                            self._get_basic_response_and_save_background(
                                provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                            )
                        )
                else:
                    asyncio.create_task(
                        self._get_basic_response_and_save_background(
                            provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                        )
                    )
            else:  # Foreground processing
                if self.stream_mode == "async":
                    if provider.supports_async and hasattr(
                        provider, "stream_completion"
                    ):
                        await self._stream_response_and_display(
                            provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                        )
                    else:
                        self.console.print(
                            "[yellow]Async mode not supported by provider, falling back to sync[/yellow]"
                        )
                        await self._get_basic_response_and_display(
                            provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                        )
                else:
                    await self._get_basic_response_and_display(
                        provider, messages, model, user_input, provider_name, max_tokens=max_tokens
                    )

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
                # If recording is turned off, also turn off background saving
                if self.config.enable_async_answers:
                    self.config.enable_async_answers = False
                    self.console.print(
                        "[yellow]Background saving also disabled as recording is off[/yellow]"
                    )
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

    def _handle_background_command(self, command: str):
        """Handle /background command to toggle async answer recording"""
        parts = command.split()
        if len(parts) > 1:
            mode = parts[1].lower()
            if mode == "on":
                if not self.record_enabled:
                    self.console.print(
                        "[red]Cannot enable background saving: Recording is not enabled. Use /record on first.[/red]"
                    )
                    return
                if self.current_task_type == TaskType.WRITE:
                    self.console.print(
                        "[red]Cannot enable background saving for 'WRITE' task type.[/red]"
                    )
                    return
                self.config.enable_async_answers = True
                self.console.print("[green]Background saving enabled[/green]")
            elif mode == "off":
                self.config.enable_async_answers = False
                self.console.print("[green]Background saving disabled[/green]")
            else:
                self.console.print(
                    Panel(
                        f"Invalid mode: {mode} (use on or off)",
                        title="Error",
                        border_style="red",
                    )
                )
        else:
            status = "on" if self.config.enable_async_answers else "off"
            self.console.print(f"Background saving is currently {status}")

    async def _stream_response_and_display(self, provider, messages, model, user_input, provider_name, max_tokens=None):
        """Handle streaming response with markdown formatting"""
        self.console.print(f"\n[bold blue]Assistant ({model}):[/bold blue]")
        response_text = ""
        usage_data = None
        
        kwargs = {"task": self.current_task_type}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if self.stream_mode == "async":
            # Async mode - print chunks immediately as plain text
            try:
                response_text = ""
                usage_data = None
                async for chunk in provider.stream_completion(messages, model, **kwargs):
                    if isinstance(chunk, dict):
                        if "text" in chunk:
                            response_text += chunk["text"]
                            self.console.print(chunk["text"], end="", markup=False)
                        if "usage" in chunk:
                            usage_data = chunk["usage"]
                        if "error" in chunk:
                            self.console.print(chunk["error"], style="red")
                            response_text += f"\n[ERROR: {chunk['error']}]"
                    else:
                        self.console.print(
                            f"[red]Invalid chunk format: {type(chunk)}[/red]"
                        )

                self.console.print()
                if self.config.enable_async_answers:
                    task = asyncio.create_task(
                        self._add_to_history(
                            user_input, response_text, model, provider_name, usage_data
                        )
                    )
                    task.add_done_callback(
                        lambda t: (
                            self.console.print(
                                f"[red]Background saving task error: {t.exception()}[/red]"
                            )
                            if t.exception()
                            else None
                        )
                    )
                    self.console.print(
                        "[bold yellow]Response recorded in background.[/bold yellow]"
                    )
                else:
                    await self._add_to_history(
                        user_input, response_text, model, provider_name, usage_data
                    )
            except Exception as e:
                self.console.print(
                    Panel(f"Stream error: {str(e)}", title="Error", border_style="red")
                )

    async def _get_basic_response_and_display(
        self, provider, messages, model, user_input, provider_name, max_tokens=None
    ):
        """Get non-streaming response"""
        kwargs = {"task": self.current_task_type}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        with self.console.status("[bold blue]Thinking...[/bold blue]"):
            response = await provider.chat_completion(messages, model, **kwargs)
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
            if self.config.enable_async_answers:
                task = asyncio.create_task(
                    self._add_to_history(user_input, content, model, provider_name, usage)
                )
                task.add_done_callback(
                    lambda t: (
                        self.console.print(
                            f"[red]Background saving task error: {t.exception()}[/red]"
                        )
                        if t.exception()
                        else None
                    )
                )
                self.console.print(
                    "[bold yellow]Response recorded in background.[/bold yellow]"
                )
            else:
                await self._add_to_history(user_input, content, model, provider_name, usage)

    async def _stream_response_and_save_background(
        self, provider, messages, model, user_input, provider_name, max_tokens=None
    ):
        """Handle streaming response in background and save"""
        response_text = ""
        usage_data = None
        
        kwargs = {"task": self.current_task_type}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            async for chunk in provider.stream_completion(messages, model, **kwargs):
                if isinstance(chunk, dict):
                    if "text" in chunk:
                        response_text += chunk["text"]
                    if "error" in chunk:
                        response_text += f"\n[ERROR: {chunk['error']}]"
                if "usage" in chunk:
                    usage_data = chunk["usage"]
            task = asyncio.create_task(
                self._add_to_history(user_input, response_text, model, provider_name, usage_data)
            )
            task.add_done_callback(
                lambda t: (
                    self.console.print(
                        f"[red]Background saving task error: {t.exception()}[/red]"
                    )
                    if t.exception()
                    else None
                )
            )
        except Exception as e:
            self.console.print(f"[red]Background stream error: {str(e)}[/red]")
            task.add_done_callback(
                lambda t: (
                    self.console.print(
                        f"[red]Background saving task error: {t.exception()}[/red]"
                    )
                    if t.exception()
                    else None
                )
            )
        except Exception as e:
            self.console.print(f"[red]Background stream error: {str(e)}[/red]")

    async def _get_basic_response_and_save_background(
        self, provider, messages, model, user_input, provider_name, max_tokens=None
    ):
        """Get non-streaming response in background and save"""
        kwargs = {"task": self.current_task_type}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            response = await provider.chat_completion(messages, model, **kwargs)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            usage = response.usage if hasattr(response, "usage") else None
            task = asyncio.create_task(
                self._add_to_history(user_input, content, model, provider_name, usage)
            )
            task.add_done_callback(
                lambda t: (
                    self.console.print(
                        f"[red]Background saving task error: {t.exception()}[/red]"
                    )
                    if t.exception()
                    else None
                )
            )
        except Exception as e:
            self.console.print(f"[red]Background API error: {str(e)}[/red]")

    def _clean_filename(self, filename):
        """Remove newlines and trim whitespace from filename"""
        return re.sub(r"\s+", " ", filename).strip()

    def _handle_search_command(self, command: str):
        """Handle /search command to search archived answers and restore to current answers"""
        parts = command.split(maxsplit=1)
        query = parts[1] if len(parts) > 1 else ""

        if not query:
            self.console.print(
                Panel(
                    "Please provide a search query", title="Error", border_style="red"
                )
            )
            return

        results = self.archive.search_answers(query)

        if not results:
            self.console.print("[yellow]No matching results found[/yellow]")
            return

        table = Table(title=f"Search Results for '{query}'")
        table.add_column("ID", style="cyan")
        table.add_column("Date", style="green")
        table.add_column("Question", style="yellow")

        # Prepare to restore answers
        restored_ids = []
        current_date = datetime.now().strftime("%Y%m%d")
        answers_dir = Path("answers") / current_date
        answers_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            answer_id, date, filename, question, answer = result
            # Shorten question for display
            short_question = (question[:50] + "...") if len(question) > 50 else question
            table.add_row(str(answer_id), date, short_question)

            # Generate new filename using the same format as _add_to_history
            time_str = datetime.now().strftime("%H%M%S")  # Include seconds
            safe_question = (
                (question[:12] if question else "restored")
                .replace("/", "_")
                .replace("\\", "_")
            )
            base_filename = f"{time_str}_{safe_question}.md"
            base_filename = self._clean_filename(base_filename)

            # Handle filename conflicts with incremental suffix
            counter = 1
            new_filename = base_filename
            while (answers_dir / new_filename).exists():
                # Remove .md extension for suffixing
                name_without_ext = base_filename[:-3]
                new_filename = f"{name_without_ext}_{counter}.md"
                counter += 1

            filepath = answers_dir / new_filename

            try:
                # Write restored answer to current answers directory
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Q:\n{question}\n\nA:\n{answer}\n")
                restored_ids.append(answer_id)
            except Exception as e:
                self.console.print(
                    f"[red]Error restoring answer {answer_id}: {e}[/red]"
                )

        self.console.print(table)
        self.console.print(f"[bold green]Found {len(results)} results[/bold green]")

        # Delete restored answers from archive
        if restored_ids:
            self.archive.delete_answers(restored_ids)
            self.console.print(
                f"[green]Restored {len(restored_ids)} answers to current answers directory and removed from archive.[/green]"
            )
        else:
            self.console.print("[yellow]No answers were restored[/yellow]")

    async def _add_to_history(
        self, user_input: str, response: str, model: str, provider_name: str, usage: Optional[dict] = None
    ):
        """Add conversation to history with API usage data"""
        self.conversation.add_turn(
            user_input,
            response,
            provider_name.lower(),
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
                time_str = datetime.now().strftime("%H%M%S")
                safe_question = user_input[:12].replace("/", "_").replace("\\", "_")
                filename = f"{time_str}_{safe_question}.md"
                filename = self._clean_filename(filename)
                filepath = answers_dir / filename

                # Prepend the generated by line to the response for file saving
                generated_by_line = f"（本答案由{provider_name}生成。）\n"
                response = generated_by_line + response

                # Write both question and answer in UTF-8 encoding
                file_content = f"Q:\n{user_input}\n\nA:\n{response}\n"
                if self.config.enable_async_answers:
                    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                        await f.write(file_content)
                else:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(file_content)
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
                        line = await asyncio.to_thread(input)
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
                    user_input = (await asyncio.to_thread(input)).strip()

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
    background: bool,
):
    """Main async entry point"""
    app_config = ApplicationConfig.from_file(config)  # Use ApplicationConfig

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
            cli.current_task_type = TaskType.CODE
        elif task == "learning":
            cli.current_task_type = TaskType.KNOWLEDGE
        elif task == "write":
            cli.current_task_type = TaskType.WRITE

    # Set initial streaming mode if provided
    if mode:
        cli.stream_mode = mode

    # Set initial modes from flags
    cli.multi_mode = multi
    cli.record_enabled = record

    # Set initial background saving mode from flag, respecting conditions
    if background:
        if cli.record_enabled and cli.current_task_type != TaskType.WRITE:
            cli.config.enable_async_answers = True
        else:
            # If background is explicitly 'on' but conditions aren't met, log a warning
            if background and (
                not cli.record_enabled or cli.current_task_type == TaskType.WRITE
            ):
                cli.console.print(
                    "[yellow]Warning: --background on ignored. Recording must be enabled and task type cannot be 'WRITE'.[/yellow]"
                )
    else:
        cli.config.enable_async_answers = False

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
    type=click.Choice(["claude", "deepseek", "gemini", "openrouter", "kimi"]),
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
@click.option(
    "--background",
    "-b",
    type=click.Choice(["on", "off"]),
    help="Enable asynchronous answer saving at start (on or off). Requires --record on and not for WRITE tasks.",
)
def main(
    config: Optional[Path],
    provider: Optional[str],
    task: Optional[str],
    mode: Optional[str],
    multi: str,
    record: str,
    background: str,
):
    """AI Chat Assistant CLI"""
    # Convert multi/record/background string values to booleans
    multi_bool = multi == "on"
    record_bool = record == "on"
    background_bool = background == "on"

    asyncio.run(
        async_main(
            config, provider, task, mode, multi_bool, record_bool, background_bool
        )
    )


if __name__ == "__main__":
    main()
