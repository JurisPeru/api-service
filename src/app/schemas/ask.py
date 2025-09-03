from typing import Any, Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str
    k: int = Field(default=12)
    temperature: float = Field(default=1.0)
    language: Literal["auto", "spanish", "english"] = Field(default="spanish")


class AskResponse(BaseModel):
    stage: str
    data: str
    contexts: list[dict[str, Any]] | None = Field(default=None)
