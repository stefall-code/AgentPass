<# 
.SYNOPSIS
    AgentPass 评委验收一键启动 — 标准验收步骤1-4
.DESCRIPTION
    1. 启动服务器  2. 初始化演示数据  3. 打开浏览器  4. 显示验收步骤
#>

$ErrorActionPreference = "SilentlyContinue"
$Host.UI.RawUI.WindowTitle = "AgentPass - Judge Acceptance"

$PORT = 8000
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host "  |  AgentPass — AI Agent Security Governance Platform  |" -ForegroundColor Cyan
Write-Host "  |  评委验收一键启动                                    |" -ForegroundColor Cyan
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: 清理旧进程
Write-Host "[1/4] 清理旧进程..." -ForegroundColor Yellow
$oldProcs = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
if ($oldProcs) {
    $pids = $oldProcs | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        if ($pid -gt 0) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 3
}
Write-Host "      OK" -ForegroundColor Green

# Step 2: 启动服务器
Write-Host "[2/4] 启动 AgentPass 服务器..." -ForegroundColor Yellow
Set-Location $PROJECT_DIR
$serverProc = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 5

$serverReady = $false
for ($i = 0; $i -lt 15; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$PORT/healthz" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            $serverReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if ($serverReady) {
    Write-Host "      服务器已启动: http://localhost:$PORT" -ForegroundColor Green
} else {
    Write-Host "      服务器启动失败!" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

# Step 3: 初始化演示数据
Write-Host "[3/4] 初始化演示数据..." -ForegroundColor Yellow
try {
    $initResp = Invoke-RestMethod -Uri "http://localhost:$PORT/api/p2/demo/init" -Method POST -TimeoutSec 10
    Write-Host "      $($initResp.message)" -ForegroundColor Green
} catch {
    Write-Host "      初始化失败（非致命）" -ForegroundColor Red
}

# Step 4: 打开浏览器
Write-Host "[4/4] 打开浏览器..." -ForegroundColor Yellow
Start-Process "http://localhost:$PORT/"
Start-Process "http://localhost:$PORT/demo"
Write-Host "      已打开仪表盘 + 演示中心" -ForegroundColor Green

# 显示验收步骤
Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Green
Write-Host "  |  评委标准验收步骤（4步）                              |" -ForegroundColor Green
Write-Host "  ======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Step 1: 系统已启动 — 3个Agent正常运行" -ForegroundColor White
Write-Host "          doc_agent (飞书文档助手) / data_agent (企业数据) / external_agent (外部检索)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Step 2: 正常委托流程" -ForegroundColor White
Write-Host "          doc_agent -> data_agent -> read:feishu_table:finance => ALLOWED" -ForegroundColor Green
Write-Host "          API: POST http://localhost:$PORT/api/delegate/demo/judge-acceptance" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Step 3: 越权拦截流程" -ForegroundColor White
Write-Host "          external_agent -> data_agent -> read:feishu_table:finance => BLOCKED" -ForegroundColor Red
Write-Host "          API: POST http://localhost:$PORT/api/delegate/demo/escalation-attack" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Step 4: 日志完整性验收" -ForegroundColor White
Write-Host "          审计日志页面: http://localhost:$PORT/audit" -ForegroundColor DarkGray
Write-Host "          哈希链验证:   GET  http://localhost:$PORT/api/delegate/audit/integrity" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host "  |  API验收 (可选)                                      |" -ForegroundColor Cyan
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  一键验收:  POST http://localhost:$PORT/api/delegate/demo/judge-acceptance" -ForegroundColor White
Write-Host "  监控告警:  GET  http://localhost:$PORT/api/monitor/alerts" -ForegroundColor White
Write-Host "  协议概念:  GET  http://localhost:$PORT/api/p2/protocol-concepts" -ForegroundColor White
Write-Host "  性能基准:  POST http://localhost:$PORT/api/p2/six-layer/benchmark" -ForegroundColor White
Write-Host "  防御模式:  GET  http://localhost:$PORT/api/prompt-defense/defense-mode" -ForegroundColor White
Write-Host "  A2A演示:   POST http://localhost:$PORT/api/protocols/demo/a2a" -ForegroundColor White
Write-Host "  P2评委:    POST http://localhost:$PORT/api/p2/judge/verify-all" -ForegroundColor White
Write-Host ""

# 保持运行
Write-Host "  服务器运行中... 按 Ctrl+C 停止" -ForegroundColor DarkGray
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 60
        $proc = Get-Process -Id $serverProc.Id -ErrorAction SilentlyContinue
        if (-not $proc) {
            Write-Host "  服务器已停止" -ForegroundColor Red
            break
        }
    }
} catch {
    Write-Host ""
} finally {
    Write-Host "  正在清理..." -ForegroundColor Yellow
    Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  已停止" -ForegroundColor Green
}
