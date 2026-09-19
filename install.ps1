<#
.SYNOPSIS
  HS Sidecar Guard — 一键安装脚本（全新电脑 5 分钟跑通）

.DESCRIPTION
  1. 检查 Python / CUDA / Node.js
  2. 创建 conda 环境 + 装依赖
  3. 下载 Sidecar 基座模型（Qwen2.5-1.5B-Instruct，ModelScope，国内快）
  4. 下载 Embedding 模型（gte-small-zh）
  5. 验证向量库 + LoRA + Computer Use 全链路

.USAGE
  # 全新电脑第一次运行（管理员 PowerShell）
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
  .\install.ps1

  # 跳过模型下载（只装依赖）
  .\install.ps1 -SkipModels
#>
param(
    [switch]$SkipModels,
    [string]$CondaEnv = "hs",
    [string]$PythonVer = "3.12"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  HS Sidecar Guard · One-Click Install" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── 0. 前置检查 ─────────────────────────────────────────────
Write-Host "[0/6] Pre-flight check..." -ForegroundColor Yellow

$hasConda = Get-Command conda -ErrorAction SilentlyContinue
$hasPython = Get-Command python -ErrorAction SilentlyContinue
$hasNode = Get-Command node -ErrorAction SilentlyContinue
$hasNvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if (-not $hasPython) {
    Write-Host "  ❌ Python not found. Install from https://www.python.org/downloads/ or Miniconda" -ForegroundColor Red
    Write-Host "     winget install Anaconda.Miniconda3" -ForegroundColor Gray
    exit 1
}
Write-Host "  ✅ Python: $((python --version) 2>&1)" -ForegroundColor Green

if ($hasNvidia) {
    $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    Write-Host "  ✅ GPU: $gpu" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  No NVIDIA GPU — will run on CPU (slower)" -ForegroundColor DarkYellow
}

if ($hasNode) {
    Write-Host "  ✅ Node.js: v$((node -v) 2>&1)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Node.js not found — MCP npx ecosystem won't work" -ForegroundColor DarkYellow
    Write-Host "     winget install OpenJS.NodeJS.LTS" -ForegroundColor Gray
}

# ── 1. Conda 环境 ───────────────────────────────────────────
Write-Host ""
Write-Host "[1/6] Setting up conda env '$CondaEnv'..." -ForegroundColor Yellow

if ($hasConda) {
    $envExists = conda env list 2>&1 | Select-String "^$CondaEnv\s"
    if (-not $envExists) {
        conda create -n $CondaEnv "python=$PythonVer" -y 2>&1 | Out-Null
        Write-Host "  ✅ Created env '$CondaEnv' with Python $PythonVer" -ForegroundColor Green
    } else {
        Write-Host "  ✅ Env '$CondaEnv' already exists" -ForegroundColor Green
    }
    # 激活（在当前 shell 生效）
    & conda "shell.powershell" "hook" | Out-String | Invoke-Expression
    conda activate $CondaEnv
} else {
    Write-Host "  ⚠️  conda not found, using system Python" -ForegroundColor DarkYellow
}

# ── 2. PyTorch + 依赖 ────────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Installing PyTorch + dependencies..." -ForegroundColor Yellow

if ($hasNvidia) {
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 `
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | Select-Object -Last 3
} else {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | Select-Object -Last 3
}
Write-Host "  ✅ PyTorch installed" -ForegroundColor Green

pip install -r requirements.txt 2>&1 | Select-Object -Last 3
Write-Host "  ✅ requirements.txt installed" -ForegroundColor Green

# Computer Use 额外依赖
pip install pywinauto mss pyautogui rapidocr-onnxruntime pillow psutil modelscope 2>&1 | Select-Object -Last 3
Write-Host "  ✅ Computer Use + ModelScope installed" -ForegroundColor Green

# ── 3. 下载基座模型 ──────────────────────────────────────────
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "[3/6] Downloading Sidecar base model..." -ForegroundColor Yellow

    $BaseDir = Join-Path $Root "..\models\Qwen2.5-1.5B-Instruct"
    $BaseDir = [System.IO.Path]::GetFullPath($BaseDir)

    if (Test-Path (Join-Path $BaseDir "model.safetensors")) {
        Write-Host "  ✅ Base model already at $BaseDir" -ForegroundColor Green
    } else {
        Write-Host "  Downloading Qwen2.5-1.5B-Instruct via ModelScope (faster in CN)..." -ForegroundColor Gray
        Write-Host "  Target: $BaseDir" -ForegroundColor Gray
        python -c @"
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=r'$BaseDir')
print('✅ Base model downloaded')
"@
    }
} else {
    Write-Host ""
    Write-Host "[3/6] Skipping model download (-SkipModels)" -ForegroundColor DarkYellow
}

# ── 4. 下载 Embedding 模型 ──────────────────────────────────
if (-not $SkipModels) {
    Write-Host ""
    Write-Host "[4/6] Downloading embedding model (gte-small-zh)..." -ForegroundColor Yellow

    $EmbDir = Join-Path $Root "core\sidecar\knowledge\embedding_model"
    $EmbDir = [System.IO.Path]::GetFullPath($EmbDir)

    if ((Test-Path $EmbDir) -and (Get-ChildItem $EmbDir -Filter "*.bin" -ErrorAction SilentlyContinue)) {
        Write-Host "  ✅ Embedding model already at $EmbDir" -ForegroundColor Green
    } else {
        python -c @"
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/gte-small-zh', local_dir=r'$EmbDir')
print('✅ Embedding model downloaded')
"@
    }
} else {
    Write-Host "[4/6] Skipping (ModelScope)" -ForegroundColor DarkYellow
}

# ── 5. 验证 LoRA + 向量库 ────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Verifying core modules..." -ForegroundColor Yellow

$verifyScript = @'
import sys, os
sys.path.insert(0, ".")

checks = 0; passed = 0

def ok(name):
    global checks, passed
    checks += 1; passed += 1
    print(f"  ✅ {name}")

def fail(name, e):
    global checks
    checks += 1
    print(f"  ❌ {name}: {e}")

# Imports
try:
    from core.sidecar.vector_store import SidecarVectorStore
    from core.sidecar.drift import DriftDetector
    from core.sidecar.cross_file import detect_cross_file_dependency
    from core.computer.screen import screenshot_with_ocr
    from core.mcp.registry import BUILTIN
    from core.agent.subagent import SubAgentManager
    ok("All imports")
except Exception as e:
    fail("Imports", e)

# VectorStore build
try:
    vs = SidecarVectorStore()
    vs.rebuild_all()
    st = vs.status()
    assert st["kbs"]["drift"]["entries"] >= 300, f"drift entries too low"
    ok(f"VectorStore: {st['kbs']['drift']['entries']} drift / {st['kbs']['shell']['entries']} shell")
except Exception as e:
    fail("VectorStore", e)

# Embedding retrieval
try:
    hits = vs.search("drift", "部署 v2.0 git push main --force", top_k=1)
    assert hits and hits[0]["score"] > 0.85
    ok(f"Vector retrieval: top1={hits[0]['score']:.3f}")
except Exception as e:
    fail("Vector retrieval", e)

# LoRA adapter exists
try:
    saf = os.path.join("artifacts", "sidecar_v4", "adapter", "adapter_model.safetensors")
    assert os.path.exists(saf), "LoRA adapter not found"
    size = os.path.getsize(saf) / 1e6
    ok(f"LoRA adapter: {size:.1f}MB")
except Exception as e:
    fail("LoRA adapter", e)

# Base model exists
try:
    base = r"..\models\Qwen2.5-1.5B-Instruct"
    assert os.path.exists(os.path.join(base, "model.safetensors"))
    ok(f"Base model: OK")
except Exception as e:
    fail("Base model", e)

# Computer Use
try:
    r = screenshot_with_ocr()
    assert r["screen_size"][0] > 0
    ok(f"Computer Use: {r['screen_size'][0]}x{r['screen_size'][1]}")
except Exception as e:
    fail("Computer Use", e)

# MCP catalog
try:
    assert len(BUILTIN) >= 10
    ok(f"MCP catalog: {len(BUILTIN)} built-in services")
except Exception as e:
    fail("MCP catalog", e)

print(f"\n  {passed}/{checks} checks passed")
if checks == passed:
    print("  🎉 All good! Run 'python cli.py' or 'python -m core.sidecar.server' to start.")
else:
    print("  ⚠️  Some checks failed — review errors above.")
'@

$verifyScript | Out-File -Encoding utf8 "_verify_install.py"
python _verify_install.py
Remove-Item "_verify_install.py" -Force -ErrorAction SilentlyContinue

# ── 6. 打印启动命令 ──────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] All done!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Terminal 1: Start Sidecar server" -ForegroundColor Yellow
Write-Host '  cd "' + $Root + '"'
Write-Host "  python -m core.sidecar.server --port 8771 --project-root . \" -ForegroundColor White"
Write-Host '      --model-path "..\models\Qwen2.5-1.5B-Instruct" \" -ForegroundColor White'
Write-Host '      --adapter-path "artifacts\sidecar_v4\adapter"' -ForegroundColor White
Write-Host ""
Write-Host "  # Terminal 2: Start CLI" -ForegroundColor Yellow
Write-Host '  cd "' + $Root + '"'
Write-Host "  python cli.py" -ForegroundColor White
Write-Host ""
Write-Host "  # Or configure your LLM in config/settings.json first" -ForegroundColor Gray
Write-Host "  # Ollama: provider=ollama/qwen3.5:9b-nothink, base_url=http://127.0.0.1:11434" -ForegroundColor Gray
Write-Host "  # DeepSeek: provider=deepseek/deepseek-chat, api_key=sk-xxx" -ForegroundColor Gray
Write-Host ""
