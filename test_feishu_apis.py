import httpx
import asyncio

async def test_feishu_apis():
    from app.feishu.client import get_feishu_client
    client = get_feishu_client()
    token = await client.get_tenant_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = "https://open.feishu.cn"

    apis = [
        ("Bot Info", "GET", "/open-apis/bot/v3/info/", {}),
        ("Calendar Primary", "GET", "/open-apis/calendar/v4/calendars/primary", {}),
        ("Calendar Events", "GET", "/open-apis/calendar/v4/calendars/primary/events", {"start_time": "1748000000", "end_time": "1749000000"}),
        ("Contact Users", "GET", "/open-apis/contact/v3/users", {"page_size": 3}),
        ("Approval List", "GET", "/open-apis/approval/v4/approvals", {"page_size": 3}),
        ("Task List", "GET", "/open-apis/task/v2/tasks", {"page_size": 3}),
        ("Drive Explorer", "GET", "/open-apis/drive/explorer/v2/root_folder/meta", {}),
        ("Doc Search", "POST", "/open-apis/suite/docs-api/search/object", {"search_key": "test", "page_size": 3}),
        ("VC Rooms", "GET", "/open-apis/vc/v1/rooms", {"page_size": 3}),
        ("Mail Public Mailboxes", "GET", "/open-apis/mail/v1/public_mailboxes", {"page_size": 3}),
        ("Sheets Spreadsheet", "GET", "/open-apis/sheets/v3/spreadsheets", {"page_size": 3}),
        ("Slides Presentations", "GET", "/open-apis/slides/v1/presentations", {}),
        ("Minutes Search", "POST", "/open-apis/minutes/v1/minutes/search", {"page_size": 3}),
        ("OKR Periods", "GET", "/open-apis/okr/v1/periods", {"page_size": 3}),
        ("Attendance Shifts", "GET", "/open-apis/attendance/v1/shifts", {"page_size": 3}),
        ("IM Chats", "GET", "/open-apis/im/v1/chats", {"page_size": 3}),
    ]

    available = []
    need_scope = []

    for name, method, path, params in apis:
        try:
            if method == "GET":
                r = httpx.get(f"{base}{path}", headers=headers, params=params, timeout=5)
            else:
                r = httpx.post(f"{base}{path}", headers=headers, json=params, timeout=5)

            status = r.status_code
            data = r.json()
            code = data.get("code", -1)
            msg = data.get("msg", "")

            if status == 200 and code == 0:
                available.append((name, "OK", ""))
                print(f"  OK  {name}: 200 (code=0)")
            elif code == 99991663 or "scope" in msg.lower() or "permission" in msg.lower():
                need_scope.append((name, "need_scope", msg[:50]))
                print(f"  --  {name}: {status} (need scope: {msg[:40]})")
            elif code == 99991668:
                need_scope.append((name, "need_scope", msg[:50]))
                print(f"  --  {name}: {status} (need scope: {msg[:40]})")
            else:
                need_scope.append((name, f"code={code}", msg[:50]))
                print(f"  --  {name}: {status} (code={code}, {msg[:40]})")
        except Exception as e:
            need_scope.append((name, "error", str(e)[:40]))
            print(f"  ERR {name}: {str(e)[:40]}")

    print(f"\n=== Summary ===")
    print(f"Available: {len(available)}")
    print(f"Need scope: {len(need_scope)}")
    for name, status, detail in available:
        print(f"  [OK] {name}")
    for name, status, detail in need_scope:
        print(f"  [--] {name}: {status} ({detail})")

asyncio.run(test_feishu_apis())
