from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class Audit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    storage_backend: Any = None

    def log_event(self, event: AuditEvent) -> None:
        if self.storage_backend:
            self.storage_backend.store(event)
        else:
            print(f"Audit event: {event.event_type} - {event.status}")

    def get_events(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> list[AuditEvent]:
        if self.storage_backend:
            return self.storage_backend.query(filters, limit)
        return []

    def get_user_events(self, user_id: str, limit: int = 100) -> list[AuditEvent]:
        return self.get_events({"user_id": user_id}, limit)
