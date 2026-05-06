import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

r = requests.get('http://127.0.0.1:8000/api/delegate/trust')
d = r.json()
print("Trust scores:")
for agent in ['doc_agent', 'data_agent', 'external_agent']:
    print(f"  {agent}: {d.get(agent, {}).get('trust_score', 'N/A')}")

r2 = requests.get('http://127.0.0.1:8000/api/delegate/auto-revoke/list')
d2 = r2.json()
print(f"Auto-revoked: {json.dumps(d2, ensure_ascii=False)[:200]}")

r3 = requests.post('http://127.0.0.1:8000/api/delegate/demo/escalation-attack')
d3 = r3.json()
print(f"Escalation test: success={d3.get('success')}, blocked={d3.get('blocked')}")
