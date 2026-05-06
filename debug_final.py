import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

r = requests.post('http://127.0.0.1:8000/api/delegate/trust/reset')
print("1. Reset:", r.json().get('status'))

r2 = requests.post('http://127.0.0.1:8000/api/feishu/test', json={'user_id':'final_test','message':'综合分析AI技术趋势'})
d = r2.json()
print("2. Status:", d.get('status'))
print("3. Capability:", d.get('capability'))

content = d.get('content', '')
idx = content.find('external_agent')
if idx > 0:
    section = content[idx:idx+200]
    if 'DENIED' in section:
        print("4. external_agent: DENIED")
    elif 'success' in section.lower() or '✅' in section:
        print("4. external_agent: SUCCESS")
    else:
        print("4. external_agent section:", section[:100])
