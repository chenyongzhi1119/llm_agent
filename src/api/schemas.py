"""Request/response schemas"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ToolInfo(BaseModel):
    name: str
    description: str


class MessageItem(BaseModel):
    role: str
    content: str


class ChatEvent(BaseModel):
    type: str   # "thought" | "action" | "observation" | "answer" | "error"
    content: str
