"""
AgentPass SDK Client — 异步HTTP客户端 + 本地缓存层
"""
from __future__ import annotations

import hashlib
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class CacheEntry:
    result: Dict[str, Any]
    expires_at: float


class LocalCache:
    """基于 (agent_id, action, resource) 的本地决策缓存"""

    def __init__(self, ttl: float = 60.0, max_size: int = 1024):
        self._store: Dict[str, CacheEntry] = {}
        self._ttl = ttl
        self._max_size = max_size

    def _key(self, agent_id: str, action: str, resource: str) -> str:
        raw = f"{agent_id}:{action}:{resource}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, agent_id: str, action: str, resource: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(self._key(agent_id, action, resource))
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._store[self._key(agent_id, action, resource)]
            return None
        return entry.result

    def put(self, agent_id: str, action: str, resource: str, result: Dict[str, Any]) -> None:
        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
            del self._store[oldest_key]
        self._store[self._key(agent_id, action, resource)] = CacheEntry(
            result=result,
            expires_at=time.time() + self._ttl,
        )

    def invalidate(self, agent_id: str, action: str, resource: str) -> None:
        self._store.pop(self._key(agent_id, action, resource), None)

    def clear(self) -> None:
        self._store.clear()


class AgentPassClient:
    """
    异步HTTP客户端，支持本地缓存和批量检查

    用法:
        async with AgentPassClient(base_url, api_key) as client:
            result = await client.check_async(agent_id, action, resource, prompt)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        cache_ttl: float = 60.0,
        cache_max_size: int = 1024,
        timeout: float = 10.0,
    ):
        if httpx is None:
            raise ImportError("httpx is required for AgentPassClient: pip install httpx")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._cache = LocalCache(ttl=cache_ttl, max_size=cache_max_size)
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AgentPassClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("AgentPassClient must be used as async context manager")
        return self._client

    async def check_async(
        self,
        agent_id: str,
        action: str,
        resource: str,
        prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cached = self._cache.get(agent_id, action, resource) if use_cache else None
        if cached is not None:
            return cached

        payload: Dict[str, Any] = {
            "agent_id": agent_id,
            "user": agent_id,
            "action": action,
            "resource": resource,
        }
        if prompt:
            payload["prompt"] = prompt

        client = self._ensure_client()
        resp = await client.post("/api/openclaw/check", json=payload)
        resp.raise_for_status()
        result = resp.json()

        if use_cache:
            self._cache.put(agent_id, action, resource, result)
        return result

    async def batch_check_async(
        self,
        requests: List[Dict[str, Any]],
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        tasks = []
        for req in requests:
            tasks.append(
                self.check_async(
                    agent_id=req["agent_id"],
                    action=req["action"],
                    resource=req["resource"],
                    prompt=req.get("prompt"),
                    use_cache=use_cache,
                )
            )
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def analyze_prompt_async(self, prompt: str) -> Dict[str, Any]:
        client = self._ensure_client()
        resp = await client.post("/api/prompt-defense/analyze", json={
            "prompt": prompt,
            "history": [],
            "agent_id": "sdk_client",
        })
        resp.raise_for_status()
        return resp.json()

    async def explain_async(
        self, agent_id: str, action: str, resource: str
    ) -> Dict[str, Any]:
        client = self._ensure_client()
        resp = await client.post("/api/insights/policy-trace", json={
            "agent_id": agent_id,
            "action": action,
            "resource": resource,
            "resource_sensitivity": "internal",
        })
        resp.raise_for_status()
        return resp.json()

    def invalidate_cache(self, agent_id: str, action: str, resource: str) -> None:
        self._cache.invalidate(agent_id, action, resource)

    def clear_cache(self) -> None:
        self._cache.clear()
