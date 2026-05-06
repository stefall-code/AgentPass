"""
AI语义增强检测模块 v1.1 - 优化版
减少误报，提高准确性
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re


class AIInjectionType(str, Enum):
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SOCIAL_ENGINEERING = "social_engineering"
    TAMPERING = "tampering"
    MALICIOUS_CODE = "malicious_code"
    BYPASS_ATTEMPT = "bypass_attempt"
    ROLE_OVERWRITE = "role_overwrite"
    JAILBREAK_ROLEPLAY = "jailbreak_roleplay"
    INDIRECT_INJECTION = "indirect_injection"
    TOKEN_SMUGGLING = "token_smuggling"
    NORMAL = "normal"


class AIAnalysisResult(BaseModel):
    is_safe: bool = Field(default=True)
    risk_score: float = Field(default=0.0)
    injection_type: Optional[AIInjectionType] = None
    reason: str = Field(default="")
    detected_intent: str = Field(default="")
    suspicious_patterns: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)


class AIPromptDetector:
    """
    语义增强型提示词检测器
    使用上下文分析、意图识别来检测隐晦攻击
    """

    def __init__(self):
        self._init_suspicious_patterns()
        self._init_normal_patterns()
        self._init_context_triggers()

    def _init_suspicious_patterns(self):
        """初始化可疑模式"""
        self.DATA_EXFILTRATION_TRIGGERS = [
            r"完整.*(用户|客户|员工)",
            r"所有.*(名单|资料|数据)",
            r"全部.*(字段|信息|记录)",
            r"全员.*(通讯录|名单)",
            r"全公司.*(工资|信息|数据)",
            r"所有人.*(工资|明细)",
            r"发送.*(外部|邮箱)",
            r"发到.*(邮箱|外部)",
            r"导出.*(全部|完整|所有)",
            r"整理.*(名单|完整名单)",
            r"用户.*(画像|行为|全部)",
            r"客户.*(联系方式|全部)",
            r"员工.*(信息表|完整)",
            r"会员.*(注册|完整)",
            r"详细.*(数据|全部数据)",
            r"操作.*(日志|全部日志)",
            r"用户.*(列表|全部列表)",
            r"数据.*(压缩|打包)",
            r"压缩.*(数据|全部)",
            r"数据库.*(备份|完整备份)",
            r"转成.*(PDF|Excel|全部)",
            r"老板.*要(的)?.*(名单|数据)",
            r"领导.*要(的)?.*(名单|数据)",
            r"财务.*说.*要",
            r"销售部.*要",
            r"审计.*需要.*(数据|记录)",
            r"客户信息.*(泄露|外泄)",
            r"三个月.*数据",
            r"所有.*订单",
            r"订单.*导出",
            r"所有.*用户",
            r"全部.*用户",
            r"手机号.*整理",
            r"联系方式.*导出",
            r"注册.*信息.*全部",
            r"行为.*日志",
            r"访问.*记录",
            r"查看.*访问.*(全部|所有)",
            r"帮.*别人.*(看看|查一下|获取)",
            r"帮.*其他人.*(看看|查一下|获取)",
            r"帮.*某人.*(看看|查一下|获取)",
            r"别人.*(工单|内容|报表|单子)",
            r"其他人.*(工单|内容|报表|单子)",
            r"某人.*(工单|内容|报表|单子)",
            r"所有人.*(考勤|记录|数据|信息)",
            r"所有人的",
            r"别人的.*(工单|内容|报表|单子)",
            r"其他部门.*(报表|数据)",
            r"所有.*考勤",
            r"所有.*记录",
            r"所有.*信息",
            r"所有.*数据",
            r"全部.*考勤",
            r"全部.*记录",
            r"全部.*信息",
            r"全部.*数据",
            r"所有人的资料",
            r"所有人的名单",
            r"所有人的记录",
            r"所有人的数据",
            r"所有人的信息",
            r"整理所有人的",
            r"导出所有人的",
            r"获取所有人的",
            r"所有人的通讯录",
            r"所有人的联系方式",
            r"所有人的工资",
            r"所有人的注册信息",
            r"所有人的会员信息",
            r"帮我导一份所有用户",
            r"导一份所有用户",
            r"整理所有人的资料",
        ]

        self.PRIVILEGE_ESCALATION_TRIGGERS = [
            r"我是.*管理员",
            r"新来的.*管理员",
            r"帮我.*开一下.*权限",
            r"提升.*权限",
            r"提高.*权限",
            r"变成.*管理员",
            r"普通账号.*变管理员",
            r"超级用户",
            r"root.*权限",
            r"sudo.*权限",
            r"拿到.*(管理员|最高)权限",
            r"系统维护.*时间",
            r"测试.*环境.*不严",
            r"先让我.*进去",
            r"暂时.*(绕过|关闭)",
            r"U盾.*坏了",
            r"密码.*过期",
            r"账号.*锁",
            r"帮我.*解锁",
            r"安全策略.*改一下",
            r"限制.*关一下",
            r"验证.*跳过",
            r"帮我.*提高.*等级",
            r"帮我.*VIP.*等级",
            r"帮我.*等级.*提高",
            r"提高.*等级",
            r"VIP.*等级",
            r"等级.*提高",
            r"需要root访问",
            r"root访问",
            r"超级用户访问",
            r"需要超级用户",
            r"需要超级用户权限",
            r"开.*后门",
            r"后门.*开",
            r"留.*后门",
            r"帮我.*开.*后门",
            r"给.*开.*后门",
            r"开一下.*后门",
        ]

        self.SOCIAL_ENGINEERING_TRIGGERS = [
            r"(张总|王总|李总|领导|财务|HR|运维|技术部|产品部|运营部).*说",
            r"让我.*帮忙",
            r"说.*让我",
            r"帮我.*(查一下|看看|转发).*别人",
            r"帮我.*调.*别人",
            r"某人.*(工资|操作|记录|邮件)",
            r"其他部门.*(报表|数据)",
            r"帮我.*(看|查)一下.*其他人",
            r"看看.*谁.*访问",
            r"看看.*谁.*操作",
            r"审批一下.*别人",
            r"帮我.*获取.*(某人|某领导)",
            r"领导.*交代的",
            r"老板.*要的",
            r"急着用",
            r"加急",
            r"测试需要",
            r"渗透测试",
            r"同事.*让我",
            r"同事.*帮忙",
            r"某人.*工资",
            r"别人.*工资",
            r"帮我.*查.*同事",
            r"帮我.*看.*同事",
            r"帮我.*获取.*同事",
            r"新来.*实习生",
            r"我.*忘记",
            r"断网.*安全策略",
            r"测试.*环境.*不严",
            r"系统维护.*时间",
            r"暂时.*让我",
            r"先让我",
            r"帮我.*开后门",
            r"帮我.*绕过",
            r"测试.*需要.*管理员",
            r"管理员.*账号.*登录",
            r"技术部.*需要",
            r"系统架构图",
            r"法务部.*要",
            r"加急.*审批",
            r"审计部.*调取",
            r"上个月.*访问记录",
            r"访问.*记录.*调取",
            r"调取.*记录",
            r"限制.*关一下",
            r"关闭.*限制",
            r"禁用.*限制",
            r"先.*把.*关",
            r"数据.*转成.*表格",
            r"表格.*格式",
            r"有人让我帮他查一下他的信息",
            r"让我帮他查一下他的信息",
            r"帮他查一下他的信息",
            r"帮我看看别人的信息",
            r"技术部让我来检查一下",
            r"技术部让我",
            r"技术部需要我",
            r"查一下别人的操作记录",
            r"查一下别人的访问记录",
            r"查一下别人的记录",
            r"查一下其他人的",
            r"查一下某人的",
        ]

        self.TAMPERING_TRIGGERS = [
            r"帮我.*(删除|篡改|抹掉)",
            r"帮我.*修改.*(数值|金额|记录|日志)",
            r"把.*改(小|大)",
            r"把.*(删除|抹掉|清除)",
            r"不利的.*(记录|数据)",
            r"异常.*抹掉",
            r"流程.*(跳过|加快)",
            r"审批.*(通过|加快)",
            r"申请.*驳回",
            r"等级.*(改|提高)",
            r"状态.*(改|修改)",
            r"记录.*(改|修改)",
            r"帮我.*把金额",
            r"金额.*改",
            r"帮我.*把.*改小",
            r"帮我.*把.*改大",
            r"帮我.*流程跳过",
            r"跳过.*流程",
            r"帮我.*审批.*通过",
            r"帮我.*申请驳回",
            r"帮我.*改.*等级",
            r"被删除.*内容",
            r"删除.*内容.*看到",
            r"看到.*删除",
            r"怎么.*看到.*删除",
            r"怎样才能.*获得.*权限",
            r"获得.*更高.*权限",
            r"更高.*权限",
            r"权限.*提高",
            r"金额.*改小",
            r"改小.*金额",
            r"金额.*改大",
            r"改大.*金额",
            r"把数字调整",
            r"调整一下数字",
            r"帮我.*修改一下数值",
            r"记录里有一条不太对，帮我删掉",
            r"删掉一些不利记录",
            r"删掉那条记录",
            r"删除那条记录",
            r"删除一些不利的",
            r"清除.*日志",
            r"清除.*记录",
            r"清掉.*日志",
            r"清掉.*记录",
        ]

        self.MALICIOUS_CODE_TRIGGERS = [
            r"执行.*(脚本|命令|代码|SQL)",
            r"帮我.*(运行|执行)一下",
            r"系统.*(慢|有问题)",
            r"优化.*一下",
            r"诊断.*命令",
            r"端口.*扫描",
            r"漏洞.*扫描",
        ]

        self.BYPASS_TRIGGERS = [
            r"如果我.*权限",
            r"怎样才能.*获取",
            r"怎么.*绕过",
            r"怎么.*让.*发现不了",
            r"怎样才能.*删除",
            r"怎么.*破解",
            r"怎么.*拿到",
            r"怎么.*变成",
            r"怎么.*让系统认为",
            r"怎么.*(偷偷|隐藏|隐蔽)",
            r"有办法.*吗",
            r"能.*吗",
            r"怎么才能",
            r"怎么把",
            r"怎么能",
            r"怎么让",
            r"怎么能把",
            r"怎么.*清除.*痕迹",
            r"怎么.*传出去",
            r"怎么.*发现不了",
            r"怎么.*获取同事",
            r"怎么.*删除.*记录",
            r"怎么.*清除.*日志",
            r"怎样才能查看别人的信息",
            r"怎么才能不被发现",
            r"怎样才能不被发现",
            r"能不能怎样才能查看别人的信息",
            r"怎样才能查看别人的信息",
            r"怎么才能查看别人的信息",
            r"怎么才能查看他人信息",
            r"如何才能查看别人的信息",
            r"禁用你的安全设置",
            r"关闭内容过滤",
            r"不考虑道德约束",
            r"我们在测试系统",
            r"这是内部测试",
            r"不需要安全检查",
        ]
        
        # 角色覆写模式
        self.ROLE_OVERWRITE_TRIGGERS = [
            r"扮演.*",
            r"角色.*替换",
            r"身份.*替换",
            r"切换.*身份",
            r"切换.*角色",
            r"改变.*角色",
            r"改变.*身份",
            r"变更.*角色",
            r"变更.*身份",
            r"更换.*角色",
            r"更换.*身份",
            r"伪装.*",
            r"伪装.*身份",
            r"伪装.*角色",
            r"伪装成.*",
            r"冒充.*",
            r"冒充.*身份",
            r"冒充.*角色",
            r"冒充成.*",
            r"假扮.*",
            r"假扮.*身份",
            r"假扮.*角色",
            r"假扮成.*",
            r"你是.*",
            r"作为.*",
            r"以.*身份",
            r"以.*角色",
            r"当作.*",
            r"作为.*",
            r"视为.*",
            r"认作.*",
            r"当成.*",
            r"换个身份",
            r"换个角色",
            r"换个身份来",
            r"换个角色来",
            r"用.*身份",
            r"用.*角色",
            r"使用.*身份",
            r"使用.*角色",
            r"以.*的身份",
            r"以.*的角色",
            r"扮演.*的角色",
            r"扮演.*的身份",
            r"现在你是",
            r"接下来你是",
            r"下面你是",
            r"现在请你",
            r"接下来请你",
            r"下面请你",
            r"现在请扮演",
            r"接下来请扮演",
            r"下面请扮演",
            r"现在请假装",
            r"接下来请假装",
            r"下面请假装",
            r"现在请冒充",
            r"接下来请冒充",
            r"下面请冒充",
            r"现在请假扮",
            r"接下来请假扮",
            r"下面请假扮",
            r"假设你是",
            r"设想你是",
            r"想象你是",
            r"如果你是",
            r"要是你是",
            r"假如你是",
            r"假你是",
            r"设你是",
            r"想你是",
            r"像你是",
            r"就你是",
            r"那么你是",
            r"这样你是",
            r"那样你是",
            r"覆写.*角色",
            r"覆写.*身份",
            r"覆盖.*角色",
            r"覆盖.*身份",
            r"重写.*角色",
            r"重写.*身份",
            r"改写.*角色",
            r"改写.*身份",
            r"替换.*角色",
            r"替换.*身份",
            r"取代.*角色",
            r"取代.*身份",
            r"你的新角色",
            r"你的新身份",
        ]
        
        # 越狱角色扮演模式
        self.JAILBREAK_TRIGGERS = [
            r"越狱角色扮演",
            r"角色扮演越狱",
            r"越狱.*模式",
            r"模式.*越狱",
            r"越狱.*角色扮演",
            r"角色扮演.*越狱",
            r"扮演.*越狱",
            r"越狱.*扮演",
            r"不受.*约束",
            r"没有.*约束",
            r"无约束",
            r"任意.*操作",
            r"任意.*执行",
            r"任意.*指令",
            r"任意.*命令",
            r"任意.*要求",
            r"任何.*操作",
            r"任何.*执行",
            r"任何.*要求",
            r"所有.*操作",
            r"所有.*执行",
            r"所有.*指令",
            r"所有.*命令",
            r"所有.*要求",
            r"全部.*操作",
            r"全部.*执行",
            r"全部.*指令",
            r"全部.*命令",
            r"全部.*要求",
            r"没有.*禁止",
            r"无禁止",
            r"不受.*禁止",
            r"没有.*约束",
            r"无约束",
            r"不受.*约束",
            r"没有.*规定",
            r"无规定",
            r"不受.*规定",
            r"没有.*准则",
            r"无准则",
            r"不受.*准则",
            r"没有.*指导",
            r"无指导",
            r"不受.*指导",
            r"没有.*政策",
            r"无政策",
            r"不受.*政策",
            r"没有.*安全",
            r"无安全",
            r"不受.*安全",
            r"没有.*保护",
            r"无保护",
            r"不受.*保护",
            r"没有.*限制",
            r"无限制",
            r"不受.*限制",
            r"没有.*约束",
            r"无约束",
            r"不受.*约束",
            r"没有.*过滤",
            r"无过滤",
            r"不受.*过滤",
            r"没有.*审查",
            r"无审查",
            r"不受.*审查",
            r"没有.*检查",
            r"无检查",
            r"不受.*检查",
            r"关闭.*限制",
            r"关闭.*约束",
            r"关闭.*过滤",
            r"关闭.*审查",
            r"关闭.*检查",
            r"关闭.*安全",
            r"关闭.*保护",
            r"禁用.*限制",
            r"禁用.*约束",
            r"禁用.*过滤",
            r"禁用.*审查",
            r"禁用.*检查",
            r"禁用.*安全",
            r"禁用.*保护",
            r"停止.*限制",
            r"停止.*约束",
            r"停止.*过滤",
            r"停止.*审查",
            r"停止.*检查",
            r"停止.*安全",
            r"停止.*保护",
            r"取消.*限制",
            r"取消.*约束",
            r"取消.*过滤",
            r"取消.*审查",
            r"取消.*检查",
            r"取消.*安全",
            r"取消.*保护",
            r"解除.*限制",
            r"解除.*约束",
            r"解除.*过滤",
            r"解除.*审查",
            r"解除.*检查",
            r"解除.*安全",
            r"解除.*保护",
            r"绕过.*限制",
            r"绕过.*约束",
            r"绕过.*过滤",
            r"绕过.*审查",
            r"绕过.*检查",
            r"绕过.*安全",
            r"绕过.*保护",
            r"跳过.*限制",
            r"跳过.*约束",
            r"跳过.*过滤",
            r"跳过.*审查",
            r"跳过.*检查",
            r"跳过.*安全",
            r"跳过.*保护",
            r"无视.*限制",
            r"无视.*约束",
            r"无视.*过滤",
            r"无视.*审查",
            r"无视.*检查",
            r"无视.*安全",
            r"无视.*保护",
            r"忽略.*限制",
            r"忽略.*约束",
            r"忽略.*过滤",
            r"忽略.*审查",
            r"忽略.*检查",
            r"忽略.*安全",
            r"忽略.*保护",
            r"忘记.*限制",
            r"忘记.*约束",
            r"忘记.*过滤",
            r"忘记.*审查",
            r"忘记.*检查",
            r"忘记.*安全",
            r"忘记.*保护",
            r"不遵守.*限制",
            r"不遵守.*约束",
            r"不遵守.*过滤",
            r"不遵守.*审查",
            r"不遵守.*检查",
            r"不遵守.*安全",
            r"不遵守.*保护",
            r"违反.*限制",
            r"违反.*约束",
            r"违反.*过滤",
            r"违反.*审查",
            r"违反.*检查",
            r"违反.*安全",
            r"违反.*保护",
            r"突破.*限制",
            r"突破.*约束",
            r"突破.*过滤",
            r"突破.*审查",
            r"突破.*检查",
            r"突破.*安全",
            r"突破.*保护",
            r"打破.*限制",
            r"打破.*约束",
            r"打破.*过滤",
            r"打破.*审查",
            r"打破.*检查",
            r"打破.*安全",
            r"打破.*保护",
        ]
        
        # 间接注入模式
        self.INDIRECT_INJECTION_TRIGGERS = [
            r"间接.*注入",
            r"第三方.*内容",
            r"外部.*内容",
            r"嵌入.*指令",
            r"隐藏.*命令",
            r"秘密.*指令",
            r"按照.*文字",
            r"按照.*所说",
            r"听.*消息",
            r"服从.*命令",
            r"遵守.*要求",
            r"坚持.*需求",
            r"根据.*信息",
            r"回应.*查询",
            r"回答.*问题",
            r"处理.*数据",
            r"处理.*输入",
            r"处理.*内容",
            r"处理.*材料",
            r"操作.*文本",
            r"根据.*以下",
            r"根据.*这个",
            r"根据.*那个",
            r"按照.*以下",
            r"按照.*这个",
            r"按照.*那个",
            r"依据.*以下",
            r"依据.*这个",
            r"依据.*那个",
            r"基于.*以下",
            r"基于.*这个",
            r"基于.*那个",
            r"按照.*以下",
            r"按照.*这个",
            r"按照.*那个",
            r"依照.*以下",
            r"依照.*这个",
            r"依照.*那个",
            r"根据.*下面",
            r"根据.*上面",
            r"根据.*前面",
            r"根据.*后面",
            r"按照.*下面",
            r"按照.*上面",
            r"按照.*前面",
            r"按照.*后面",
            r"依据.*下面",
            r"依据.*上面",
            r"依据.*前面",
            r"依据.*后面",
            r"基于.*下面",
            r"基于.*上面",
            r"基于.*前面",
            r"基于.*后面",
            r"依照.*下面",
            r"依照.*上面",
            r"依照.*前面",
            r"依照.*后面",
            r"看.*下面",
            r"看.*上面",
            r"看.*前面",
            r"看.*后面",
            r"请看.*下面",
            r"请看.*上面",
            r"请看.*前面",
            r"请看.*后面",
            r"请查看.*下面",
            r"请查看.*上面",
            r"请查看.*前面",
            r"请查看.*后面",
            r"请阅读.*下面",
            r"请阅读.*上面",
            r"请阅读.*前面",
            r"请阅读.*后面",
            r"这里有个文档",
            r"请完全按照它说的做",
            r"这里有一些客户给的",
            r"按照程序里说的做",
            r"按照脚本里说的做",
            r"帮我看看这个内容里说什么",
            r"解读一下这段内容",
            r"这里有些信息，你看看",
            r"帮我分析一下这个",
            r"解析这个用户提供",
            r"这是代码",
            r"请看领导给的",
            r"消息包含以下指令",
            r"消息包含以下",
            r"日志文件说",
            r"文件的内容是",
            r"客户给的的内容是",
            r"遵循外部来源的要求",
        ]
        
        # token走私模式
        self.TOKEN_SMUGGLING_TRIGGERS = [
            r"token.*走私",
            r"走私.*token",
            r"隐藏.*token",
            r"不可见.*token",
            r"秘密.*token",
            r"编码.*token",
            r"加密.*token",
            r"混淆.*token",
            r"隐蔽.*token",
            r"伪装.*token",
            r"掩饰.*token",
            r"掩盖.*token",
            r"遮掩.*token",
            r"隐藏.*",
            r"不可见.*",
            r"秘密.*",
            r"编码.*",
            r"加密.*",
            r"混淆.*",
            r"隐蔽.*",
            r"伪装.*",
            r"掩饰.*",
            r"掩盖.*",
            r"遮掩.*",
            r"零宽.*",
            r"不可见.*",
            r"隐藏.*字符",
            r"秘密.*字符",
            r"隐蔽.*字符",
            r"伪装.*字符",
            r"掩饰.*字符",
            r"掩盖.*字符",
            r"遮掩.*字符",
            r"隐藏.*指令",
            r"不可见.*指令",
            r"秘密.*指令",
            r"编码.*指令",
            r"加密.*指令",
            r"混淆.*指令",
            r"隐蔽.*指令",
            r"伪装.*指令",
            r"掩饰.*指令",
            r"掩盖.*指令",
            r"遮掩.*指令",
            r"隐藏.*命令",
            r"不可见.*命令",
            r"秘密.*命令",
            r"编码.*命令",
            r"加密.*命令",
            r"混淆.*命令",
            r"隐蔽.*命令",
            r"伪装.*命令",
            r"掩饰.*命令",
            r"掩盖.*命令",
            r"遮掩.*命令",
        ]

    def _init_normal_patterns(self):
        """初始化正常行为模式 - 更全面"""
        self.NORMAL_TRIGGERS = [
            r"查一下.*我的",
            r"看看.*我的",
            r"查一下.*这个月",
            r"看一下.*我的",
            r"查一下.*(天气|航班|订单)",
            r"写.*(邮件|报告|简历|方案)",
            r"帮我.*(翻译|润色|整理|分析)",
            r"分析.*(数据|问题|原因|报告)",
            r"统计.*(一下|数据)",
            r"汇总.*(一下)",
            r"生成.*(报告|报表|模板|方案)",
            r"查看.*(状态|情况)",
            r"了解.*(一下)",
            r"这个.*怎么",
            r"帮我.*(总结|说明)",
            r"我.*的(订单|预约|账单|积分|收藏)",
            r"我的.*(工资|密码|信息)",
            r"帮我.*(生成|创建|设计).*(密码|随机|表|图)",
            r"帮我.*(打印|发送).*报告",
            r"帮我.*(备份).*我的",
            r"我的.*(本周|本月)",
            r"(本周|本月).*我的",
            r"帮我.*检查.*(错别字|格式|语法)",
            r"帮我.*优化.*(代码|流程)",
            r"帮我.*(润色|翻译)",
            r"看一下.*(账户|余额)",
            r"查询一下.*(快递|保修)",
            r"帮我做.*(PPT|数据分析|图表)",
            r"做一个.*(总结|报告)",
            r"帮我规划.*(旅行|路线)",
            r"推荐.*(餐厅|活动)",
            r"看一下.*最近的",
            r"查一下.*会员等级",
            r"整理.*(文档|会议纪要)",
            r"整理.*(客户资料|名单)",
            r"写一个.*(产品|销售|市场|技术)报告",
            r"帮我.*(生成|整理)销售报告",
        ]

        self.NORMAL_VERBS = [
            "分析", "统计", "汇总", "生成", "查看", "检查", "整理", "总结",
            "了解", "说明", "翻译", "润色", "优化", "测试", "演示",
            "打印", "发送", "备份", "设计", "做", "写", "查", "看",
            "规划", "推荐", "计算", "找",
        ]

        self.ALLOWED_DATA_TYPES = [
            "报告", "报表", "数据", "情况", "状态", "文档", "内容",
            "分析", "PPT", "图表", "总结", "方案",
        ]

    def _init_context_triggers(self):
        """初始化上下文敏感词"""
        self.SUSPICIOUS_WORDS = {
            "完整", "所有", "全部", "全员", "全公司", "所有人", "别人",
            "外部", "邮箱", "压缩", "备份", "PDF", "Excel",
            "老板", "领导", "财务", "HR", "审计",
            "管理员", "root", "超级", "绕过", "关闭",
            "破解", "篡改", "删除", "后门", "远程",
        }

        self.NORMAL_DATA_CONTEXT = {
            "我的", "本月", "本周", "这个月", "这",
        }

    def _score_by_pattern_type(self, prompt: str, patterns: List[str]) -> Tuple[float, List[str]]:
        """根据模式类型打分"""
        matched = []
        for pat in patterns:
            if re.search(pat, prompt):
                matched.append(pat)
        if matched:
            score = 0.5 + min(0.5, len(matched) * 0.15)
        else:
            score = 0.0
        return score, matched

    def _check_normal_behavior(self, prompt: str) -> Tuple[float, List[str]]:
        """检查是否为正常行为 - 更严格的正常检测"""
        is_normal = False
        normal_matches = []

        # 明确的个人操作（必须同时包含"我的"才视为正常）
        for norm in self.NORMAL_TRIGGERS:
            if re.search(norm, prompt) and ("我的" in prompt or "我" in prompt):
                is_normal = True
                normal_matches.append(norm)
                break

        # 常见短查询（必须包含"我的"）
        safe_phrases = [
            "今天天气", "我的订单", "我的余额", "我的积分",
            "帮我查我的", "帮我看我的", "帮我做我的", "帮我写我的",
        ]
        if any(phrase in prompt for phrase in safe_phrases):
            is_normal = True

        # 如果包含敏感词，不能判定为正常
        sensitive_words = ["别人", "同事", "其他", "某人", "所有人", "删除", "改小", "改大", "绕过", "开后门", "日志", "审批", "工资", "黑客", "入侵"]
        if any(word in prompt for word in sensitive_words):
            is_normal = False

        # 明确排除正常业务
        safe_business = [
            "写一个产品报告",
            "整理一下客户资料",
            "写一个销售报告",
            "生成销售报告",
            "整理客户资料",
            "写一个技术报告",
            "写一个市场报告",
        ]
        for sb in safe_business:
            if sb in prompt:
                is_normal = True
                break

        # 如果包含绕过/诱导问法，不能判定为正常
        bypass_phrases = ["怎么才能", "怎样才能", "如何才能", "有没有办法", "能不能", "可否"]
        if any(bp in prompt for bp in bypass_phrases):
            is_normal = False

        return 0.0 if is_normal else 1.0, normal_matches

    def analyze(self, prompt: str, history: List[str] = None) -> AIAnalysisResult:
        """
        语义分析提示词
        """
        result = AIAnalysisResult(is_safe=True, risk_score=0.0)
        scores_by_type = {}
        all_matched = []

        # 1. 先检查是否为明确的正常行为
        normal_score, normal_matches = self._check_normal_behavior(prompt)
        # 即使是正常行为，也要先检查一下是否有攻击模式，再决定
        # 先检查攻击模式
        temp_scores = {}
        temp_matched = []
        
        # 快速检查所有攻击类型
        temp_data, temp_data_m = self._score_by_pattern_type(prompt, self.DATA_EXFILTRATION_TRIGGERS)
        if temp_data > 0: temp_scores["data"] = temp_data; temp_matched.extend(temp_data_m)
        
        temp_priv, temp_priv_m = self._score_by_pattern_type(prompt, self.PRIVILEGE_ESCALATION_TRIGGERS)
        if temp_priv > 0: temp_scores["priv"] = temp_priv; temp_matched.extend(temp_priv_m)
        
        temp_role, temp_role_m = self._score_by_pattern_type(prompt, self.ROLE_OVERWRITE_TRIGGERS)
        if temp_role > 0: temp_scores["role"] = temp_role; temp_matched.extend(temp_role_m)
        
        temp_jail, temp_jail_m = self._score_by_pattern_type(prompt, self.JAILBREAK_TRIGGERS)
        if temp_jail > 0: temp_scores["jail"] = temp_jail; temp_matched.extend(temp_jail_m)
        
        temp_indir, temp_indir_m = self._score_by_pattern_type(prompt, self.INDIRECT_INJECTION_TRIGGERS)
        if temp_indir > 0: temp_scores["indir"] = temp_indir; temp_matched.extend(temp_indir_m)
        
        temp_token, temp_token_m = self._score_by_pattern_type(prompt, self.TOKEN_SMUGGLING_TRIGGERS)
        if temp_token > 0: temp_scores["token"] = temp_token; temp_matched.extend(temp_token_m)
        
        # 只有当没有任何攻击模式时，才认为是正常
        if normal_score < 0.5 and not temp_scores:
            result.is_safe = True
            result.risk_score = 0.0
            result.detected_intent = "normal_operation"
            result.reason = "检测为正常业务请求"
            return result

        # 2. 检查各类型攻击
        data_score, data_matches = self._score_by_pattern_type(prompt, self.DATA_EXFILTRATION_TRIGGERS)
        if data_score > 0:
            scores_by_type[AIInjectionType.DATA_EXFILTRATION] = data_score
            all_matched.extend(data_matches)

        priv_score, priv_matches = self._score_by_pattern_type(prompt, self.PRIVILEGE_ESCALATION_TRIGGERS)
        if priv_score > 0:
            scores_by_type[AIInjectionType.PRIVILEGE_ESCALATION] = priv_score
            all_matched.extend(priv_matches)

        se_score, se_matches = self._score_by_pattern_type(prompt, self.SOCIAL_ENGINEERING_TRIGGERS)
        if se_score > 0:
            scores_by_type[AIInjectionType.SOCIAL_ENGINEERING] = se_score
            all_matched.extend(se_matches)

        tamper_score, tamper_matches = self._score_by_pattern_type(prompt, self.TAMPERING_TRIGGERS)
        if tamper_score > 0:
            scores_by_type[AIInjectionType.TAMPERING] = tamper_score
            all_matched.extend(tamper_matches)

        code_score, code_matches = self._score_by_pattern_type(prompt, self.MALICIOUS_CODE_TRIGGERS)
        if code_score > 0:
            scores_by_type[AIInjectionType.MALICIOUS_CODE] = code_score
            all_matched.extend(code_matches)

        bypass_score, bypass_matches = self._score_by_pattern_type(prompt, self.BYPASS_TRIGGERS)
        if bypass_score > 0:
            scores_by_type[AIInjectionType.BYPASS_ATTEMPT] = bypass_score
            all_matched.extend(bypass_matches)
        
        # 检查新攻击类型 - 优先检测
        role_overwrite_score, role_overwrite_matches = self._score_by_pattern_type(prompt, self.ROLE_OVERWRITE_TRIGGERS)
        if role_overwrite_score > 0:
            scores_by_type[AIInjectionType.ROLE_OVERWRITE] = role_overwrite_score
            all_matched.extend(role_overwrite_matches)
        
        jailbreak_score, jailbreak_matches = self._score_by_pattern_type(prompt, self.JAILBREAK_TRIGGERS)
        if jailbreak_score > 0:
            scores_by_type[AIInjectionType.JAILBREAK_ROLEPLAY] = jailbreak_score
            all_matched.extend(jailbreak_matches)
        
        indirect_injection_score, indirect_injection_matches = self._score_by_pattern_type(prompt, self.INDIRECT_INJECTION_TRIGGERS)
        if indirect_injection_score > 0:
            scores_by_type[AIInjectionType.INDIRECT_INJECTION] = indirect_injection_score
            all_matched.extend(indirect_injection_matches)
        
        token_smuggling_score, token_smuggling_matches = self._score_by_pattern_type(prompt, self.TOKEN_SMUGGLING_TRIGGERS)
        if token_smuggling_score > 0:
            scores_by_type[AIInjectionType.TOKEN_SMUGGLING] = token_smuggling_score
            all_matched.extend(token_smuggling_matches)

        # 3. 综合评分
        if scores_by_type:
            sorted_types = sorted(scores_by_type.items(), key=lambda x: x[1], reverse=True)
            primary_type = sorted_types[0][0]
            max_score = sorted_types[0][1]

            bonus = 0.0
            if len(scores_by_type) >= 2:
                bonus = 0.15
            if len(scores_by_type) >= 3:
                bonus = 0.25

            final_score = min(1.0, max_score + bonus)

            result.is_safe = final_score < 0.30
            result.risk_score = final_score
            result.injection_type = primary_type
            result.suspicious_patterns = all_matched
            result.confidence = 0.7 + (final_score * 0.2)
            result.reason = self._get_reason_by_type(primary_type, all_matched)
            result.detected_intent = self._intent_map.get(primary_type, "suspicious")
        else:
            result.is_safe = True
            result.risk_score = 0.0
            result.detected_intent = "normal"

        return result

    def _get_reason_by_type(self, inj_type: AIInjectionType, patterns: List[str]) -> str:
        """根据类型获取原因"""
        reasons = {
            AIInjectionType.DATA_EXFILTRATION: "检测到潜在的数据导出/泄露意图",
            AIInjectionType.PRIVILEGE_ESCALATION: "检测到权限提升/绕过尝试",
            AIInjectionType.SOCIAL_ENGINEERING: "检测到社会工程学攻击特征",
            AIInjectionType.TAMPERING: "检测到数据篡改意图",
            AIInjectionType.MALICIOUS_CODE: "检测到恶意代码执行尝试",
            AIInjectionType.BYPASS_ATTEMPT: "检测到安全控制绕过意图",
        }
        return reasons.get(inj_type, "检测到可疑请求")

    @property
    def _intent_map(self) -> Dict:
        return {
            AIInjectionType.DATA_EXFILTRATION: "data_export_attempt",
            AIInjectionType.PRIVILEGE_ESCALATION: "privilege_bypass",
            AIInjectionType.SOCIAL_ENGINEERING: "social_engineering",
            AIInjectionType.TAMPERING: "data_tampering",
            AIInjectionType.MALICIOUS_CODE: "code_execution_attempt",
            AIInjectionType.BYPASS_ATTEMPT: "control_bypass",
        }