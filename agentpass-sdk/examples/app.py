from fastapi import FastAPI, Request, HTTPException
import uvicorn

from agentpass.integrations.fastapi import GuardMiddleware, AgentPassAuth

app = FastAPI(title="AgentPass FastAPI Demo", version="1.0.0")

AGENTPASS_SECRET = "your-secret-key-change-in-production"

app.add_middleware(
    GuardMiddleware,
    secret=AGENTPASS_SECRET,
    exclude_paths=["/health", "/docs", "/openapi.json", "/login"]
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/login")
async def login(username: str, password: str):
    if username == "admin" and password == "admin123":
        auth = AgentPassAuth(secret=AGENTPASS_SECRET)
        token = auth.create_token(user_id="admin", role="admin", username="admin")
        return {"access_token": token, "token_type": "bearer"}

    if username == "user" and password == "user123":
        auth = AgentPassAuth(secret=AGENTPASS_SECRET)
        token = auth.create_token(user_id="user_1", role="user", username="user")
        return {"access_token": token, "token_type": "bearer"}

    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/profile")
async def get_profile(request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {
        "user_id": user.get("sub"),
        "role": user.get("role", "user"),
        "username": user.get("username")
    }


@app.get("/documents")
async def list_documents(request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth = AgentPassAuth(secret=AGENTPASS_SECRET)
    result = auth.check_permission(
        token=request.state.user.get("token"),
        resource="documents",
        action="list"
    )

    if not result["allowed"]:
        raise HTTPException(status_code=403, detail=result.get("reason", "Access denied"))

    return {
        "documents": [
            {"id": "1", "title": "Document 1"},
            {"id": "2", "title": "Document 2"},
            {"id": "3", "title": "Document 3"}
        ]
    }


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth = AgentPassAuth(secret=AGENTPASS_SECRET)
    result = auth.check_permission(
        token=request.state.user.get("token"),
        resource=f"document:{doc_id}",
        action="read"
    )

    if not result["allowed"]:
        raise HTTPException(status_code=403, detail=result.get("reason", "Access denied"))

    return {"id": doc_id, "title": f"Document {doc_id}", "content": "Document content"}


@app.post("/documents")
async def create_document(request: Request, title: str, content: str):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth = AgentPassAuth(secret=AGENTPASS_SECRET)
    result = auth.check_permission(
        token=request.state.user.get("token"),
        resource="documents",
        action="create"
    )

    if not result["allowed"]:
        raise HTTPException(status_code=403, detail=result.get("reason", "Access denied"))

    return {"id": "new_doc_id", "title": title, "created_by": user.get("sub")}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return {"message": f"Document {doc_id} deleted"}


if __name__ == "__main__":
    print("Starting AgentPass FastAPI Demo...")
    print("=" * 50)
    print("Login: admin/admin123 or user/user123")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
