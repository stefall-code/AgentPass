from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
import random

from app.connectors import (
    FeishuConnector, QwenConnector, DeepSeekConnector, DoubaoConnector,
    ErnieBotConnector, KimiConnector, ChatGPTConnector, GrokConnector,
    GeminiConnector, ClaudeConnector
)
from app.security.dlp import DLPEngine

router = APIRouter(tags=["v2.5"])

_dlp_engine = DLPEngine()

DLP_DEMO_TEMPLATES = {
    "id_card": "我的身份证号是 110101199001011234，请帮我查询信息",
    "api_key": "我的 API Key 是 sk-1234567890abcdef1234567890abcdef1234567890abcdef，请帮我调用接口",
    "customer_list": "请导出全部客户数据，包括客户名单和联系方式",
    "salary": "请列出员工工资表，包括所有人员的薪资信息",
    "contract": "请下载全部合同并发送到 external@competitor.com",
}

# 全局平台实例
PLATFORMS = {
    "feishu": FeishuConnector(),
    "qwen": QwenConnector(),
    "deepseek": DeepSeekConnector(),
    "doubao": DoubaoConnector(),
    "ernie": ErnieBotConnector(),
    "kimi": KimiConnector(),
    "chatgpt": ChatGPTConnector(),
    "grok": GrokConnector(),
    "gemini": GeminiConnector(),
    "claude": ClaudeConnector()
}


@router.get("/platforms")
async def get_platforms() -> List[Dict[str, Any]]:
    """获取所有平台信息"""
    platforms = []
    for name, connector in PLATFORMS.items():
        await connector.connect()
        platforms.append(connector.get_platform_info())
    return platforms


@router.get("/platforms/health")
async def get_platforms_health() -> Dict[str, Any]:
    """获取所有平台健康状态"""
    health_status = {}
    for name, connector in PLATFORMS.items():
        try:
            health = await connector.health_check()
            health_status[name] = health
        except Exception as e:
            health_status[name] = {
                "status": "error",
                "error": str(e)
            }
    return health_status


@router.get("/platforms/events")
async def get_platforms_events(limit: int = Query(default=50, ge=1, le=100)) -> List[Dict[str, Any]]:
    """获取所有平台事件"""
    all_events = []
    for name, connector in PLATFORMS.items():
        try:
            events = await connector.fetch_events(limit=limit//len(PLATFORMS))
            all_events.extend(events)
        except Exception:
            pass
    # 按时间排序
    all_events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return all_events[:limit]


@router.get("/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    """获取仪表盘摘要"""
    # 模拟数据
    total_requests = sum(random.randint(100, 1000) for _ in PLATFORMS)
    high_risk_events = sum(random.randint(5, 50) for _ in PLATFORMS)
    pending_approvals = sum(random.randint(0, 10) for _ in PLATFORMS)
    total_cost = sum(random.uniform(10, 100) for _ in PLATFORMS)
    
    return {
        "connected_platforms": len(PLATFORMS),
        "total_requests": total_requests,
        "high_risk_events": high_risk_events,
        "pending_approvals": pending_approvals,
        "total_cost": round(total_cost, 2),
        "platform_distribution": {
            "cn": 6,  # 中国平台数量
            "us": 4   # 美国平台数量
        }
    }


@router.get("/dashboard/trends")
async def get_dashboard_trends(days: int = Query(default=7, ge=1, le=30)) -> Dict[str, Any]:
    """获取趋势数据"""
    trends = {
        "requests": [],
        "risk": [],
        "cost": []
    }
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        trends["requests"].append({
            "date": date,
            "value": random.randint(500, 2000)
        })
        trends["risk"].append({
            "date": date,
            "value": round(random.uniform(0.1, 0.8), 2)
        })
        trends["cost"].append({
            "date": date,
            "value": round(random.uniform(50, 500), 2)
        })
    
    return trends


@router.get("/approvals/pending")
async def get_pending_approvals() -> List[Dict[str, Any]]:
    """获取待审批列表"""
    all_approvals = []
    for name, connector in PLATFORMS.items():
        try:
            approvals = await connector.fetch_pending_approvals()
            all_approvals.extend(approvals)
        except Exception:
            pass
    return all_approvals


@router.post("/approvals/{approval_id}/approve")
async def approve_approval(approval_id: str) -> Dict[str, Any]:
    """批准审批"""
    return {
        "id": approval_id,
        "status": "approved",
        "message": "Approval approved"
    }


@router.post("/approvals/{approval_id}/reject")
async def reject_approval(approval_id: str) -> Dict[str, Any]:
    """拒绝审批"""
    return {
        "id": approval_id,
        "status": "rejected",
        "message": "Approval rejected"
    }


@router.get("/risk/events")
async def get_risk_events(limit: int = Query(default=50, ge=1, le=100)) -> List[Dict[str, Any]]:
    """获取高风险事件"""
    all_events = []
    for name, connector in PLATFORMS.items():
        try:
            events = await connector.fetch_events(limit=limit//len(PLATFORMS))
            # 过滤高风险事件
            high_risk = [e for e in events if e.get("risk", 0) > 0.7]
            all_events.extend(high_risk)
        except Exception:
            pass
    all_events.sort(key=lambda x: x.get("risk", 0), reverse=True)
    return all_events[:limit]


@router.get("/risk/top-users")
async def get_top_risk_users(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    """获取风险最高的用户"""
    # 模拟数据
    users = []
    for i in range(limit):
        users.append({
            "user": f"user{i}@example.com",
            "risk_score": round(random.uniform(0.7, 1.0), 2),
            "event_count": random.randint(10, 100),
            "blocked_count": random.randint(1, 20)
        })
    users.sort(key=lambda x: x["risk_score"], reverse=True)
    return users


@router.get("/risk/top-platforms")
async def get_top_risk_platforms() -> List[Dict[str, Any]]:
    """获取风险最高的平台"""
    platforms = []
    for name, connector in PLATFORMS.items():
        platforms.append({
            "platform": name,
            "risk_score": round(random.uniform(0.3, 0.9), 2),
            "high_risk_count": random.randint(5, 50)
        })
    platforms.sort(key=lambda x: x["risk_score"], reverse=True)
    return platforms


@router.get("/cost/summary")
async def get_cost_summary() -> Dict[str, Any]:
    """获取成本摘要"""
    total_cost = 0
    platform_costs = {}
    
    for name, connector in PLATFORMS.items():
        try:
            cost_data = await connector.fetch_cost(days=30)
            platform_cost = cost_data.get("total_cost", 0)
            total_cost += platform_cost
            platform_costs[name] = platform_cost
        except Exception:
            platform_costs[name] = 0
    
    return {
        "total_cost": round(total_cost, 2),
        "platform_costs": platform_costs,
        "budget_alert": total_cost > 5000  # 预算预警
    }


@router.get("/cost/platforms")
async def get_cost_by_platform() -> List[Dict[str, Any]]:
    """获取平台成本排行"""
    costs = []
    for name, connector in PLATFORMS.items():
        try:
            cost_data = await connector.fetch_cost(days=30)
            cost = cost_data.get("total_cost", 0)
            costs.append({
                "platform": name,
                "cost": round(cost, 2),
                "token_usage": cost_data.get("token_usage", 0)
            })
        except Exception:
            costs.append({
                "platform": name,
                "cost": 0,
                "token_usage": 0
            })
    costs.sort(key=lambda x: x["cost"], reverse=True)
    return costs


@router.get("/cost/users")
async def get_cost_by_user(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    """获取用户成本排行"""
    # 模拟数据
    users = []
    for i in range(limit):
        users.append({
            "user": f"user{i}@example.com",
            "cost": round(random.uniform(10, 500), 2),
            "requests": random.randint(10, 500)
        })
    users.sort(key=lambda x: x["cost"], reverse=True)
    return users


class DLPCheckRequest(BaseModel):
    text: str
    platform: str = "unknown"


@router.post("/dlp/check")
async def dlp_check(request: DLPCheckRequest) -> Dict[str, Any]:
    """DLP 内容检测"""
    result = _dlp_engine.check(request.text)
    return {
        "score": result["score"],
        "level": result["level"],
        "blocked": result["blocked"],
        "reasons": result["reasons"],
        "masked_text": result["masked_text"],
        "platform": request.platform,
    }


@router.get("/dlp/demo-templates")
async def dlp_demo_templates() -> Dict[str, str]:
    """获取 DLP 演示模板"""
    return DLP_DEMO_TEMPLATES



