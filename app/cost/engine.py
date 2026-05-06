from typing import Dict, List
from datetime import datetime, timedelta
import random
from collections import defaultdict


class CostEngine:
    """统一成本治理引擎"""
    
    # 平台成本系数（美元）
    PLATFORM_COST_FACTORS = {
        "chatgpt": 0.002,      # $0.002 per 1K tokens
        "claude": 0.0015,       # $0.0015 per 1K tokens
        "gemini": 0.001,        # $0.001 per 1K tokens
        "grok": 0.0025,         # $0.0025 per 1K tokens
        "feishu": 0.001,         # $0.001 per 1K tokens
        "qwen": 0.0008,          # $0.0008 per 1K tokens
        "deepseek": 0.0007,      # $0.0007 per 1K tokens
        "doubao": 0.0009,        # $0.0009 per 1K tokens
        "ernie": 0.0012,         # $0.0012 per 1K tokens
        "kimi": 0.0011,          # $0.0011 per 1K tokens
    }
    
    # 汇率
    CNY_TO_USD = 7.2
    
    def __init__(self):
        # 存储成本数据
        self.cost_records = []
        # 预算设置
        self.monthly_budget = 5000  # 默认月度预算 $5000
    
    def calculate_cost(self, request: Dict) -> Dict:
        """计算单次请求成本"""
        platform = request.get("platform", "unknown")
        token_usage = request.get("token_usage", 0)
        
        # 获取平台成本系数
        cost_factor = self.PLATFORM_COST_FACTORS.get(platform, 0.001)
        
        # 计算美元成本
        usd_cost = (token_usage / 1000) * cost_factor
        
        # 计算人民币成本
        cny_cost = usd_cost * self.CNY_TO_USD
        
        # 创建成本记录
        record = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "user": request.get("user", "anonymous"),
            "team": request.get("team", "default"),
            "action": request.get("action", "unknown"),
            "resource": request.get("resource", "unknown"),
            "token_usage": token_usage,
            "usd_cost": round(usd_cost, 4),
            "cny_cost": round(cny_cost, 2),
            "request_id": request.get("request_id", f"req_{len(self.cost_records)}")
        }
        
        self.cost_records.append(record)
        
        return record
    
    def get_cost_summary(self, days: int = 30) -> Dict:
        """获取成本摘要"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        recent_records = [r for r in self.cost_records if r["timestamp"] >= cutoff_time]
        
        total_usd = sum(r["usd_cost"] for r in recent_records)
        total_cny = sum(r["cny_cost"] for r in recent_records)
        total_tokens = sum(r["token_usage"] for r in recent_records)
        total_requests = len(recent_records)
        
        # 平台成本分布
        platform_costs = defaultdict(float)
        for r in recent_records:
            platform_costs[r["platform"]] += r["usd_cost"]
        
        # 用户成本分布
        user_costs = defaultdict(float)
        for r in recent_records:
            user_costs[r["user"]] += r["usd_cost"]
        
        # 团队成本分布
        team_costs = defaultdict(float)
        for r in recent_records:
            team_costs[r["team"]] += r["usd_cost"]
        
        # 检查预算
        monthly_cost = self.get_period_cost(30)
        budget_alert = monthly_cost > self.monthly_budget
        
        return {
            "total_requests": total_requests,
            "total_token_usage": total_tokens,
            "total_usd_cost": round(total_usd, 2),
            "total_cny_cost": round(total_cny, 2),
            "platform_costs": dict(platform_costs),
            "user_costs": dict(user_costs),
            "team_costs": dict(team_costs),
            "budget_alert": budget_alert,
            "monthly_budget": self.monthly_budget,
            "monthly_cost": round(monthly_cost, 2)
        }
    
    def get_period_cost(self, days: int) -> float:
        """获取指定天数的成本"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        recent_records = [r for r in self.cost_records if r["timestamp"] >= cutoff_time]
        return sum(r["usd_cost"] for r in recent_records)
    
    def get_cost_by_platform(self, days: int = 30) -> List[Dict]:
        """获取平台成本排行"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        recent_records = [r for r in self.cost_records if r["timestamp"] >= cutoff_time]
        
        platform_stats = defaultdict(lambda: {"cost": 0, "token_usage": 0, "requests": 0})
        for r in recent_records:
            platform_stats[r["platform"]]["cost"] += r["usd_cost"]
            platform_stats[r["platform"]]["token_usage"] += r["token_usage"]
            platform_stats[r["platform"]]["requests"] += 1
        
        # 转换为列表并排序
        result = []
        for platform, stats in platform_stats.items():
            result.append({
                "platform": platform,
                "cost": round(stats["cost"], 2),
                "token_usage": stats["token_usage"],
                "requests": stats["requests"]
            })
        
        # 按成本排序
        result.sort(key=lambda x: x["cost"], reverse=True)
        return result
    
    def get_cost_by_user(self, days: int = 30, limit: int = 10) -> List[Dict]:
        """获取用户成本排行"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        recent_records = [r for r in self.cost_records if r["timestamp"] >= cutoff_time]
        
        user_stats = defaultdict(lambda: {"cost": 0, "requests": 0})
        for r in recent_records:
            user_stats[r["user"]]["cost"] += r["usd_cost"]
            user_stats[r["user"]]["requests"] += 1
        
        # 转换为列表并排序
        result = []
        for user, stats in user_stats.items():
            result.append({
                "user": user,
                "cost": round(stats["cost"], 2),
                "requests": stats["requests"]
            })
        
        # 按成本排序
        result.sort(key=lambda x: x["cost"], reverse=True)
        return result[:limit]
    
    def get_cost_by_team(self, days: int = 30) -> List[Dict]:
        """获取团队成本排行"""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        recent_records = [r for r in self.cost_records if r["timestamp"] >= cutoff_time]
        
        team_stats = defaultdict(lambda: {"cost": 0, "requests": 0})
        for r in recent_records:
            team_stats[r["team"]]["cost"] += r["usd_cost"]
            team_stats[r["team"]]["requests"] += 1
        
        # 转换为列表并排序
        result = []
        for team, stats in team_stats.items():
            result.append({
                "team": team,
                "cost": round(stats["cost"], 2),
                "requests": stats["requests"]
            })
        
        # 按成本排序
        result.sort(key=lambda x: x["cost"], reverse=True)
        return result
    
    def get_daily_trend(self, days: int = 7) -> List[Dict]:
        """获取每日成本趋势"""
        trend = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            start_time = (datetime.now() - timedelta(days=i+1)).isoformat()
            end_time = (datetime.now() - timedelta(days=i)).isoformat()
            
            day_records = [r for r in self.cost_records 
                         if start_time <= r["timestamp"] < end_time]
            
            total_cost = sum(r["usd_cost"] for r in day_records)
            total_tokens = sum(r["token_usage"] for r in day_records)
            total_requests = len(day_records)
            
            trend.append({
                "date": date,
                "cost": round(total_cost, 2),
                "token_usage": total_tokens,
                "requests": total_requests
            })
        
        # 反转顺序，使最早的日期在前
        trend.reverse()
        return trend
    
    def set_monthly_budget(self, budget: float):
        """设置月度预算"""
        self.monthly_budget = budget
    
    def generate_mock_cost_data(self, count: int = 1000):
        """生成模拟成本数据"""
        platforms = list(self.PLATFORM_COST_FACTORS.keys())
        users = [f"user{i}@example.com" for i in range(1, 21)]
        teams = [f"team{i}" for i in range(1, 11)]
        actions = ["chat_completion", "data_access", "file_upload", "model_training"]
        resources = ["general_query", "customer_data", "financial_data", "internal_docs"]
        
        for i in range(count):
            # 生成过去30天内的随机时间
            days_ago = random.randint(0, 29)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            timestamp = (datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)).isoformat()
            
            platform = random.choice(platforms)
            token_usage = random.randint(10, 10000)
            cost_factor = self.PLATFORM_COST_FACTORS[platform]
            usd_cost = (token_usage / 1000) * cost_factor
            cny_cost = usd_cost * self.CNY_TO_USD
            
            record = {
                "timestamp": timestamp,
                "platform": platform,
                "user": random.choice(users),
                "team": random.choice(teams),
                "action": random.choice(actions),
                "resource": random.choice(resources),
                "token_usage": token_usage,
                "usd_cost": round(usd_cost, 4),
                "cny_cost": round(cny_cost, 2),
                "request_id": f"mock_req_{i}"
            }
            
            self.cost_records.append(record)
    
    def get_today_cost(self) -> Dict:
        """获取今日成本"""
        today = datetime.now().strftime("%Y-%m-%d")
        start_time = f"{today}T00:00:00"
        end_time = f"{today}T23:59:59"
        
        today_records = [r for r in self.cost_records 
                       if start_time <= r["timestamp"] <= end_time]
        
        total_usd = sum(r["usd_cost"] for r in today_records)
        total_cny = sum(r["cny_cost"] for r in today_records)
        total_tokens = sum(r["token_usage"] for r in today_records)
        total_requests = len(today_records)
        
        return {
            "date": today,
            "total_requests": total_requests,
            "total_token_usage": total_tokens,
            "total_usd_cost": round(total_usd, 2),
            "total_cny_cost": round(total_cny, 2)
        }
