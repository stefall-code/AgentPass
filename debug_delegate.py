import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

from app.delegation.engine import DelegationEngine, CAPABILITY_AGENTS, get_trust_score
from app.config import settings

engine = DelegationEngine()

root_token = engine.issue_root_token(
    agent_id="doc_agent",
    delegated_user="debug_test",
    capabilities=list(CAPABILITY_AGENTS["doc_agent"]["capabilities"]),
)

print("Root token capabilities:", CAPABILITY_AGENTS["doc_agent"]["capabilities"])
print()

del_result = engine.delegate(
    parent_token=root_token,
    target_agent="external_agent",
    action="read:web",
    caller_agent="doc_agent",
)
print("Delegate result:")
print("  success:", del_result.success)
print("  reason:", del_result.reason)
print("  effective_caps:", del_result.effective_caps if hasattr(del_result, 'effective_caps') else 'N/A')
if del_result.success:
    print("  token (first 50):", del_result.token[:50] if del_result.token else 'None')

    check_result = engine.check(token=del_result.token, action="read:web")
    print()
    print("Check result:")
    print("  allowed:", check_result.allowed)
    print("  reason:", check_result.reason)
    print("  capabilities:", check_result.capabilities)
