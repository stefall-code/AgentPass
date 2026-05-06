from typing import Dict, List
from datetime import datetime
import random
from collections import defaultdict


class RiskEngine:
    """统一风控引擎"""
    
    # 风险因子权重
    RISK_FACTORS = {
        "prompt_injection": 0.3,
        "dlp": 0.25,
        "abnormal_behavior": 0.15,
        "high_frequency": 0.1,
        "multi_platform_attack": 0.1,
        "late_night_access": 0.05,
        "high_cost": 0.05,
    }
    
    def __init__(self):
        # 存储用户行为数据
        self.user_behavior = defaultdict(list)
        # 存储平台事件
        self.platform_events = defaultdict(list)
        # 存储成本数据
        self.cost_data = defaultdict(list)
    
    def calculate_risk(self, request: Dict) -> Dict:
        """计算综合风险评分"""
        scores = {}
        reasons = []
        
        # 1. Prompt Injection 风险
        prompt_risk = request.get("prompt_risk", 0)
        scores["prompt_injection"] = prompt_risk
        if prompt_risk > 0.7:
            reasons.append(f"High prompt injection risk: {prompt_risk:.2f}")
        
        # 2. DLP 风险
        dlp_risk = request.get("dlp_risk", 0)
        scores["dlp"] = dlp_risk
        if dlp_risk > 0.7:
            reasons.append(f"High DLP risk: {dlp_risk:.2f}")
        
        # 3. 用户异常行为
        abnormal_behavior = self._detect_abnormal_behavior(request)
        scores["abnormal_behavior"] = abnormal_behavior
        if abnormal_behavior > 0.7:
            reasons.append("Abnormal user behavior detected")
        
        # 4. 高频调用
        high_frequency = self._detect_high_frequency(request)
        scores["high_frequency"] = high_frequency
        if high_frequency > 0.7:
            reasons.append("High frequency requests detected")
        
        # 5. 多平台连续攻击
        multi_platform = self._detect_multi_platform_attack(request)
        scores["multi_platform_attack"] = multi_platform
        if multi_platform > 0.7:
            reasons.append("Multi-platform attack detected")
        
        # 6. 深夜访问
        late_night = self._detect_late_night_access(request)
        scores["late_night_access"] = late_night
        if late_night > 0.7:
            reasons.append("Late night access detected")
        
        # 7. 高成本异常
        high_cost = self._detect_high_cost(request)
        scores["high_cost"] = high_cost
        if high_cost > 0.7:
            reasons.append("High cost anomaly detected")
        
        # 计算综合风险
        final_risk = 0
        for factor, weight in self.RISK_FACTORS.items():
            final_risk += scores.get(factor, 0) * weight
        
        final_risk = min(1.0, final_risk)
        
        # 确定风险等级
        risk_level = self._get_risk_level(final_risk)
        
        # 确定是否需要阻断
        blocked = final_risk > 0.8
        
        # 确定是否需要审批
        require_approval = final_risk > 0.7 and not blocked
        
        # 记录事件
        self._record_event(request, final_risk)
        
        return {
            "final_risk": final_risk,
            "risk_level": risk_level,
            "blocked": blocked,
            "require_approval": require_approval,
            "reasons": reasons,
            "factor_scores": scores
        }
    
    def _detect_abnormal_behavior(self, request: Dict) -> float:
        """检测用户异常行为"""
        user = request.get("user")
        if not user:
            return 0
        
        # 检查用户历史行为
        user_events = self.user_behavior[user]
        if len(user_events) < 3:
            return 0
        
        # 检查最近的拒绝次数
        recent_denials = sum(1 for event in user_events[-10:] if event.get("blocked", False))
        if recent_denials > 3:
            return 0.8
        
        return 0
    
    def _detect_high_frequency(self, request: Dict) -> float:
        """检测高频调用"""
        user = request.get("user")
        if not user:
            return 0
        
        # 检查最近1分钟的请求次数
        now = datetime.now()
        recent_events = [event for event in self.user_behavior[user] 
                       if (now - datetime.fromisoformat(event.get("timestamp"))).total_seconds() < 60]
        
        if len(recent_events) > 20:
            return 0.9
        elif len(recent_events) > 10:
            return 0.6
        elif len(recent_events) > 5:
            return 0.3
        
        return 0
    
    def _detect_multi_platform_attack(self, request: Dict) -> float:
        """检测多平台连续攻击"""
        user = request.get("user")
        if not user:
            return 0
        
        # 检查最近5分钟内的多平台事件
        now = datetime.now()
        recent_events = [event for event in self.user_behavior[user] 
                       if (now - datetime.fromisoformat(event.get("timestamp"))).total_seconds() < 300]
        
        platforms = set(event.get("platform") for event in recent_events)
        if len(platforms) > 3 and len(recent_events) > 10:
            return 0.8
        
        return 0
    
    def _detect_late_night_access(self, request: Dict) -> float:
        """检测深夜访问"""
        now = datetime.now()
        hour = now.hour
        
        # 深夜时间：22:00 - 06:00
        if 22 <= hour or hour < 6:
            return 0.7
        
        return 0
    
    def _detect_high_cost(self, request: Dict) -> float:
        """检测高成本异常"""
        cost = request.get("cost", 0)
        user = request.get("user")
        
        if not user:
            return 0
        
        # 检查用户历史成本
        user_costs = self.cost_data[user]
        if not user_costs:
            return 0
        
        avg_cost = sum(user_costs) / len(user_costs)
        if cost > avg_cost * 3:
            return 0.8
        
        return 0
    
    def _get_risk_level(self, risk: float) -> str:
        """根据风险分数确定风险等级"""
        if risk >= 0.9:
            return "critical"
        elif risk >= 0.7:
            return "high"
        elif risk >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _record_event(self, request: Dict, risk: float):
        """记录事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user": request.get("user"),
            "platform": request.get("platform"),
            "action": request.get("action"),
            "resource": request.get("resource"),
            "risk": risk,
            "blocked": risk > 0.8,
            "cost": request.get("cost", 0)
        }
        
        # 记录用户行为
        user = request.get("user")
        if user:
            self.user_behavior[user].append(event)
            # 只保留最近100个事件
            if len(self.user_behavior[user]) > 100:
                self.user_behavior[user] = self.user_behavior[user][-100:]
        
        # 记录平台事件
        platform = request.get("platform")
        if platform:
            self.platform_events[platform].append(event)
            # 只保留最近50个事件
            if len(self.platform_events[platform]) > 50:
                self.platform_events[platform] = self.platform_events[platform][-50:]
        
        # 记录成本数据
        if user and "cost" in request:
            self.cost_data[user].append(request["cost"])
            # 只保留最近10个成本记录
            if len(self.cost_data[user]) > 10:
                self.cost_data[user] = self.cost_data[user][-10:]
    
    def get_top_risk_users(self, limit: int = 10) -> List[Dict]:
        """获取风险最高的用户"""
        user_risks = {}
        
        for user, events in self.user_behavior.items():
            if events:
                recent_events = events[-10:]  # 最近10个事件
                avg_risk = sum(event.get("risk", 0) for event in recent_events) / len(recent_events)
                blocked_count = sum(1 for event in recent_events if event.get("blocked", False))
                user_risks[user] = {
                    "user": user,
                    "risk_score": avg_risk,
                    "event_count": len(recent_events),
                    "blocked_count": blocked_count
                }
        
        # 按风险分数排序
        sorted_users = sorted(user_risks.values(), key=lambda x: x["risk_score"], reverse=True)
        return sorted_users[:limit]
    
    def get_top_risk_platforms(self) -> List[Dict]:
        """获取风险最高的平台"""
        platform_risks = {}
        
        for platform, events in self.platform_events.items():
            if events:
                recent_events = events[-20:]  # 最近20个事件
                avg_risk = sum(event.get("risk", 0) for event in recent_events) / len(recent_events)
                high_risk_count = sum(1 for event in recent_events if event.get("risk", 0) > 0.7)
                platform_risks[platform] = {
                    "platform": platform,
                    "risk_score": avg_risk,
                    "high_risk_count": high_risk_count
                }
        
        # 按风险分数排序
        sorted_platforms = sorted(platform_risks.values(), key=lambda x: x["risk_score"], reverse=True)
        return sorted_platforms
    
    def generate_mock_risk_events(self, count: int = 50) -> List[Dict]:
        """生成模拟风险事件"""
        events = []
        platforms = ["chatgpt", "feishu", "qwen", "claude", "gemini", "grok"]
        users = [f"user{i}@example.com" for i in range(1, 21)]
        actions = ["chat_completion", "data_access", "file_upload", "model_training"]
        resources = ["general_query", "customer_data", "financial_data", "internal_docs"]
        
        for i in range(count):
            risk = random.uniform(0, 1)
            event = {
                "id": f"event_{i}",
                "timestamp": datetime.now().isoformat(),
                "platform": random.choice(platforms),
                "region": "us" if random.choice(platforms) in ["chatgpt", "claude", "gemini", "grok"] else "cn",
                "user": random.choice(users),
                "team": f"team{random.randint(1, 10)}",
                "action": random.choice(actions),
                "resource": random.choice(resources),
                "prompt": "This is a test prompt",
                "output": "This is a test output",
                "risk": risk,
                "risk_level": self._get_risk_level(risk),
                "approval_required": risk > 0.7 and risk <= 0.8,
                "approval_status": "pending" if risk > 0.7 else "none",
                "cost": round(random.uniform(0.01, 10.0), 2),
                "token_usage": random.randint(10, 1000),
                "blocked": risk > 0.8,
                "reason": "High risk content detected" if risk > 0.8 else ""
            }
            events.append(event)
        
        return events
