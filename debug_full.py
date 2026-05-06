import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

r = requests.post('http://127.0.0.1:8000/api/delegate/trust/reset')
print("Reset:", r.json().get('status'))

from app.delegation.engine import DelegationEngine, CAPABILITY_AGENTS, AUTO_REVOKED_AGENTS, REVOKED_AGENTS, get_trust_score

engine = DelegationEngine()

print("AUTO_REVOKED_AGENTS:", dict(AUTO_REVOKED_AGENTS))
print("REVOKED_AGENTS:", dict(REVOKED_AGENTS))
print("external_agent trust:", get_trust_score("external_agent"))

root_token = engine.issue_root_token(
    agent_id="doc_agent",
    delegated_user="debug_test",
    capabilities=list(CAPABILITY_AGENTS["doc_agent"]["capabilities"]),
)

del_result = engine.delegate(
    parent_token=root_token,
    target_agent="external_agent",
    action="read:web",
    caller_agent="doc_agent",
)
print("Delegate success:", del_result.success)
print("Delegate reason:", del_result.reason)

if del_result.success:
    check_result = engine.check(token=del_result.token, action="read:web")
    print("Check allowed:", check_result.allowed)
    print("Check reason:", check_result.reason)
