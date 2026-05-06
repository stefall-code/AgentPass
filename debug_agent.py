import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.db import SessionLocal
from app.models import AgentRow
with SessionLocal() as db:
    agent = db.get(AgentRow, "external_agent")
    if agent:
        print(f"external_agent: status={agent.status}, reason={agent.status_reason}, role={agent.role}")
    else:
        print("external_agent not found in DB")
