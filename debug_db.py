import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.db import SessionLocal
from app.models import TokenRevocationRow
with SessionLocal() as db:
    rows = db.query(TokenRevocationRow).all()
    print(f"TokenRevocationRow count: {len(rows)}")
    for r in rows:
        print(f"  {r.revoke_type}: {r.revoke_key} - {r.reason}")
