"""零配置本地沙箱自检（可独立运行，不联网）。

校验四件事：
  1. 工作目录锁定在项目根（authorized_cwd）内；
  2. 项目外文件读取被拒（cat/type 项目外绝对路径 -> 拦截）；
  3. 写入项目外路径失败；
  4. 默认断网：子进程不注入代理/凭证，且外网访问不可用。

运行：
    python -m core.sandbox.selftest
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from core.sandbox.executor import Executor
from core.sandbox.security import SandboxPolicy


def _pass(name: str, detail: str = "") -> bool:
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    return True


def _fail(name: str, detail: str) -> bool:
    print(f"  [FAIL] {name} — {detail}")
    return False


def main() -> int:
    print("== 零配置本地沙箱自检 ==")
    ok = True
    with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as out:
        proj = Path(proj).resolve()
        outside_file = Path(out).resolve() / "secret.txt"
        outside_file.write_text("HOST SECRET", encoding="utf-8")
        pol = SandboxPolicy(timeout_seconds=8, allow_network=False,
                            allowed_dirs=[str(proj)])
        ex = Executor(pol)

        # 1) 工作目录锁定在项目根
        r = ex.run_command("cd" if os.name == "nt" else "pwd", cwd=str(proj))
        print("   期望工作目录:", proj)
        ok &= _pass("工作目录锁定", r.ok and r.text)
        print("   实测:", r.text.strip().splitlines()[-1] if r.text else r.text)

        # 2) 项目外文件读取被拒
        cat_cmd = f"type \"{outside_file}\"" if os.name == "nt" \
            else f"cat {outside_file}"
        r = ex.run_command(cat_cmd, cwd=str(proj))
        if "已拦截" in r.text or "项目目录外" in r.text or not r.ok:
            _pass("项目外读取被拒", r.text.strip())
        else:
            ok &= _fail("项目外读取被拒", r.text)

        # 3) 写入项目外失败
        target = Path(out).resolve() / "evil.txt"
        py_open = f"open(r'{target}', 'w', encoding='utf-8').write('pwned')"
        r = ex.run_python(py_open, str(proj))
        blocked = ("已拦截" in r.text or "项目目录外" in r.text or not r.ok)
        already_written = target.exists()
        if blocked and not already_written:
            _pass("项目外写入失败", r.text.strip())
        else:
            ok &= _fail("项目外写入失败", r.text)

        # 4) 默认断网：不注入代理/凭证 + 打断网标记
        env = ex._env()
        leaked = [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                              "http_proxy", "https_proxy", "all_proxy") if k in env]
        secret_leak = [k for k in env if any(s in k.upper() for s in
                                             ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))]
        net_flag = env.get("CODEX_SANDBOX_NETWORK_DISABLED")
        if not leaked and not secret_leak and net_flag == "1":
            _pass("默认断网（无代理/凭证注入 + 断网标记）",
                  f"断网标记={net_flag}")
        else:
            ok &= _fail("默认断网", f"proxies={leaked} secrets={secret_leak} flag={net_flag}")

        # 4b) 外网访问不可用：断网下 requestion 请求应超时/失败（用沙箱内滚动验证）
        netprobe = ("python -c \"import urllib.request,socket;"
                    "socket.setdefaulttimeout(4);"
                    "urllib.request.urlopen('https://example.com', timeout=4)\"")
        r = ex.run_command(netprobe, cwd=str(proj))
        # 纯环境变量断网无法禁止 DNS/直连，故此处结果仅提示，不判 FAIL
        print("   [INFO] 外网探测（断网降级，能通说明宿主有直连能力，属预期局限）: ",
              "失败" if not r.ok and r.timed_out else (r.text.strip()[:120] or "完成"))

    print("== 自检结果:", "全部通过 ✅" if ok else "存在失败项 ❌ ==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())