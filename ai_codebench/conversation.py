"""Conversation history management"""

import json
from typing import List, Dict, Any, Optional, Union
from .config import TaskType
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from .providers.base import Message


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""

    timestamp: str
    user_message: str
    assistant_message: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    cached: bool = False


class ConversationHistory:
    """Manages conversation history with configurable window size"""

    def __init__(self, window_size: int = 3, session_file: Optional[Path] = None):
        self.window_size = window_size
        self.session_file = session_file or Path.home() / ".ai_codebench_history.json"
        self.turns: List[ConversationTurn] = []
        self._load_session()

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        provider: str,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        cached: bool = False,
    ):
        """Add a new conversation turn"""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            assistant_message=assistant_message,
            provider=provider,
            model=model,
            usage=usage,
            cached=cached,
        )

        self.turns.append(turn)

        # Keep only the last N turns based on window size
        if len(self.turns) > self.window_size:
            self.turns = self.turns[-self.window_size :]

        self._save_session()

    def get_messages_for_api(
        self, 
        include_system: bool = False, 
        task_type: Union[TaskType, None] = None
    ) -> List[Message]:
        """Convert conversation history to API messages format"""
        messages = []

        if include_system:
            if task_type == TaskType.KNOWLEDGE:
                system_content = "You are a helpful AI assistant for knowledge learning. Teach the concept step by step."
            elif task_type == TaskType.CODE:
                system_content = "You are a helpful AI assistant for code tasks. Analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code. Please respond in Simplified Chinese."
            elif task_type == TaskType.WRITE:
                system_content = "You are a helpful AI assistant for writing instruction, who polishes English drafts for clarity, grammar, and natural tone. Keep the author's voice as much as possible."
            else:
                # Default system prompt for unknown/mixed task types
                system_content = "You are a helpful AI assistant for both knowledge learning and code tasks. For the knowledge learning task, teach the concept step by step. For the code tasks, you only need to analyze the algorithm ideas, algorithm steps and computational complexity, but don't write specific code."
                
            messages.append(Message(role="system", content=system_content))

        # Add conversation history within window, unless in CODE mode
        if task_type != TaskType.CODE:
            for turn in (
                self.turns[:-1] if self.turns else []
            ):  # Exclude current turn if it exists
                messages.append(Message(role="assistant", content=turn.assistant_message))
                messages.append(Message(role="user", content=turn.user_message))

        return messages

    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context"""
        if not self.turns:
            return "No conversation history"

        recent_turns = self.turns[-3:]  # Last 3 turns for context
        summary_parts = []

        for i, turn in enumerate(recent_turns, 1):
            summary_parts.append(f"Turn {i}:")
            summary_parts.append(
                f"  User: {turn.user_message[:100]}{'...' if len(turn.user_message) > 100 else ''}"
            )
            summary_parts.append(
                f"  Assistant ({turn.provider}): {turn.assistant_message[:100]}{'...' if len(turn.assistant_message) > 100 else ''}"
            )
            if turn.usage:
                tokens = turn.usage.get(
                    "total_tokens",
                    turn.usage.get("input_tokens", 0)
                    + turn.usage.get("output_tokens", 0),
                )
                summary_parts.append(
                    f"  Tokens: {tokens}{' (cached)' if turn.cached else ''}"
                )

        return "\n".join(summary_parts)

    def clear_history(self):
        """Clear all conversation history"""
        self.turns = []
        self._save_session()

    def _load_session(self):
        """Load conversation history from file"""
        self.turns = []  # Always start with empty turns
        if self.session_file.exists():
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Only load turns if they have valid usage data
                    self.turns = [
                        turn for turn in [
                            ConversationTurn(**t) for t in data.get("turns", [])
                        ] if turn.usage is not None
                    ]
                    # Apply window size limit to loaded data
                    if len(self.turns) > self.window_size:
                        self.turns = self.turns[-self.window_size :]
            except (json.JSONDecodeError, Exception):
                pass  # Already initialized empty turns

    def _save_session(self):
        """Save conversation history to file"""
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "turns": [asdict(turn) for turn in self.turns],
                        "window_size": self.window_size,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            # Fail silently to not interrupt the chat flow
            print(f"Warning: Could not save session history: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get accurate conversation statistics for CLI display"""
        stats = {
            "total_turns": len(self.turns),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for turn in self.turns:
            if not turn.usage:
                continue
                
            # Handle different provider formats
            usage = turn.usage
            if "input_tokens" in usage:  # DeepSeek format
                stats["prompt_tokens"] += usage.get("input_tokens", 0)
                stats["completion_tokens"] += usage.get("output_tokens", 0)
            elif "prompt_tokens" in usage:  # OpenAI/Claude format
                stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
            elif "usage_metadata" in usage:  # Gemini format
                stats["prompt_tokens"] += usage["usage_metadata"].get("prompt_token_count", 0)
                stats["completion_tokens"] += usage["usage_metadata"].get("candidates_token_count", 0)
            
            stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]

        return stats
