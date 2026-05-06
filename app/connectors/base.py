from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseConnector(ABC):
    """Connector 基类"""
    
    def __init__(self, platform: str, region: str, mock: bool = True):
        self.platform = platform
        self.region = region
        self.mock = mock
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到平台"""
        pass
    
    @abstractmethod
    async def fetch_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件"""
        pass
    
    @abstractmethod
    async def fetch_cost(self, days: int = 7) -> Dict[str, Any]:
        """获取成本"""
        pass
    
    @abstractmethod
    async def fetch_pending_approvals(self) -> List[Dict[str, Any]]:
        """获取待审批"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    def get_platform_info(self) -> Dict[str, Any]:
        """获取平台信息"""
        return {
            "platform": self.platform,
            "region": self.region,
            "mock": self.mock,
            "connected": self.connected
        }
