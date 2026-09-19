"""窗口与应用：列出可见窗口、切到目标窗口、打开应用/文件/链接。"""
from __future__ import annotations


def list_windows() -> str:
    try:
        from pywinauto import Desktop
    except Exception as e:  # noqa: BLE001
        return f"错误：未安装 pywinauto - {e}"
    desc = Desktop(backend="uia")
    out = []
    for w in desc.windows():
        try:
            if not w.is_visible():
                continue
            t = w.window_text() or ""
            if not t:
                continue
            r = w.rectangle()
            out.append(f"{t}  rect({int(r.left)},{int(r.top)},{int(r.right)},{int(r.bottom)})")
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(out) if out else "(无可视窗口)"


def focus_window(name: str) -> str:
    """模糊匹配窗口标题并前置到前台。"""
    try:
        from pywinauto import Desktop
    except Exception as e:  # noqa: BLE001
        return f"错误：未安装 pywinauto - {e}"
    desc = Desktop(backend="uia")
    for w in desc.windows():
        try:
            if w.is_visible() and name.lower() in (w.window_text() or "").lower():
                w.set_focus()
                try:
                    w.restore()
                except Exception:  # noqa: BLE001
                    pass
                return f"已切换到窗口: {w.window_text()}"
        except Exception:  # noqa: BLE001
            continue
    return f"未找到标题包含「{name}」的窗口（可用 computer_windows 查看）"


def open_app(cmd: str) -> str:
    """打开应用/文件/URL（使用默认程序，经 os.startfile 隔离子进程）。"""
    import os
    import threading
    cmd = str(cmd).strip()
    if not cmd:
        return "错误：需要给出要打开的应用名或路径"
    try:
        # os.startfile 会阻塞直到启动完成，丢到后台线程避免卡住 agent
        t = threading.Thread(target=os.startfile, args=(cmd,), daemon=True)
        t.start()
        return f"已用默认程序打开: {cmd}"
    except Exception as e:  # noqa: BLE001
        # 退回 Win+R 方式：尝试 Shen/系统启动
        return f"打开失败: {e}"