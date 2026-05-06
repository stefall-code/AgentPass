import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')
r = requests.post('http://127.0.0.1:8000/api/feishu/test', json={'user_id':'debug2','message':'综合分析AI技术趋势'})
d = r.json()
content = d.get('content','')
idx = content.find('external_agent')
if idx > 0:
    print(content[idx:idx+400])
else:
    print('external_agent not found in content')
    print('Status:', d.get('status'))
    print('Content[:500]:', content[:500])
