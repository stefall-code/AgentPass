import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.delegation.engine import AUTO_REVOKED_AGENTS, REVOKED_AGENTS, REVOKED_TOKENS, REVOKED_USERS, get_trust_score

print("AUTO_REVOKED_AGENTS:", dict(AUTO_REVOKED_AGENTS))
print("REVOKED_AGENTS:", dict(REVOKED_AGENTS))
print("REVOKED_TOKENS:", set(REVOKED_TOKENS))
print("external_agent trust:", get_trust_score("external_agent"))
