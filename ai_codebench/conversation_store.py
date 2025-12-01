"""Abstract base class for conversation storage and JSON implementation."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional


class ConversationStore(ABC):
    """Abstract base class for storing and retrieving conversation history."""

    @abstractmethod
    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load a conversation by its ID."""
        pass

    @abstractmethod
    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]):
        """Save a conversation by its ID."""
        pass


class JsonConversationStore(ConversationStore):
    """A ConversationStore implementation that uses JSON files for storage."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, conversation_id: str) -> Path:
        """Get the file path for a given conversation ID."""
        return self.base_dir / f"{conversation_id}.json"

    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON for conversation {conversation_id}"
                )
                return []
        return []

    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]):
        file_path = self._get_file_path(conversation_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4)
