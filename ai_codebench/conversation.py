"""Conversation history management"""

from typing import List, Dict, Any, Optional, Union
from .settings import TaskType, SYSTEM_PROMPTS
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from .providers.base import Message
from .conversation_store import ConversationStore, JsonConversationStore


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
    """Manages conversation history with configurable window size and pluggable storage"""

    def __init__(
        self,
        window_size: int = 3,
        conversation_id: str = "default",
        conversation_store: Optional[ConversationStore] = None,
    ):
        self.window_size = window_size
        self.conversation_id = conversation_id
        self.conversation_store = conversation_store or JsonConversationStore(
            base_dir=Path.home() / ".ai_codebench_history"
        )
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
        self, include_system: bool = False, task_type: Union[TaskType, None] = None
    ) -> List[Message]:
        """Convert conversation history to API messages format"""
        messages = []

        if include_system:
            system_content = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])
            messages.append(Message(role="system", content=system_content))

        # Add conversation history within window, unless in CODE mode
        if task_type != TaskType.CODE:
            for turn in (
                self.turns[:-1] if self.turns else []
            ):  # Exclude current turn if it exists
                messages.append(
                    Message(role="assistant", content=turn.assistant_message)
                )
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
        """Load conversation history from file using the configured store"""
        self.turns = []  # Always start with empty turns
        data = self.conversation_store.load_conversation(self.conversation_id)
        if data:
            self.turns = [
                turn
                for turn in [ConversationTurn(**t) for t in data]
                if turn.usage is not None
            ]
            if len(self.turns) > self.window_size:
                self.turns = self.turns[-self.window_size :]

    def _save_session(self):
        """Save conversation history to file using the configured store"""
        try:
            self.conversation_store.save_conversation(
                self.conversation_id, [asdict(turn) for turn in self.turns]
            )
        except Exception as e:
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
                stats["prompt_tokens"] += usage["usage_metadata"].get(
                    "prompt_token_count", 0
                )
                stats["completion_tokens"] += usage["usage_metadata"].get(
                    "candidates_token_count", 0
                )

            stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]

        return stats
