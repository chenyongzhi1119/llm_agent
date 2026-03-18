"""Short-term memory - sliding window conversation history"""

from config.settings import MAX_CONTEXT_MESSAGES


class ShortTermMemory:
    """
    Manages conversation history with a sliding window.
    Keeps system prompt always at index 0, trims oldest messages when limit is reached.
    """

    def __init__(self, max_messages: int = MAX_CONTEXT_MESSAGES):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Add a message to memory"""
        self._messages.append({"role": role, "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return current message history"""
        return list(self._messages)

    def set_system_prompt(self, content: str) -> None:
        """Set or replace the system prompt (always kept at index 0)"""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def clear(self) -> None:
        """Clear all messages except the system prompt"""
        if self._messages and self._messages[0]["role"] == "system":
            system = self._messages[0]
            self._messages = [system]
        else:
            self._messages = []

    def _trim(self) -> None:
        """Trim oldest messages when limit is exceeded, preserving system prompt"""
        if len(self._messages) <= self.max_messages:
            return

        has_system = self._messages and self._messages[0]["role"] == "system"
        if has_system:
            system = self._messages[0]
            # Keep system + last (max_messages - 1) messages
            self._messages = [system] + self._messages[-(self.max_messages - 1):]
        else:
            self._messages = self._messages[-self.max_messages:]

    def __len__(self) -> int:
        return len(self._messages)
