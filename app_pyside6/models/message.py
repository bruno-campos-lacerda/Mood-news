from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


@dataclass
class Message:
    content: str
    is_user: bool
    timestamp: datetime
    id: str

    @classmethod
    def create(cls, content: str, is_user: bool) -> "Message":
        return cls(content, is_user, datetime.now(), str(uuid4()))