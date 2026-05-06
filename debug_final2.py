import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

r = requests.post('http://127.0.0.1:8000/api/delegate/trust/reset')
print("Reset:", r.json().get('status'))

r2 = requests.post('http://127.0.0.1:8000/api/feishu/test', json={'user_id':'final2','message':'综合分析AI技术趋势'})
d = r2.json()
content = d.get('content', '')
print(content[:2000])
