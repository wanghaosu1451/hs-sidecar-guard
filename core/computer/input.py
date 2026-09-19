# -*- coding: utf-8 -*-
"""桌面输入：鼠标点击/拖拽/滚动 + 键盘输入/快捷键。基于 pyautogui。

升级亮点：
- DPI 感知：click/drag 的 x/y 逻辑坐标（与 pywinauto 一致）自动转物理像素喂给 pyautogui
- 新增 drag / scroll / right_click / hotkey / wait 等便捷函数
- 旧函数（click / double_click / type_text / key / move）签名保持向后兼容
"""
from __future__ import annotations

import time

import ctypes


# === DPI 感知（与 screen._screen_scale 同逻辑，避免循环导入） ===
def _screen_scale() -> float:
    """Windows DPI 缩放因子（1.0=100%, 1.25=125%, 1.5=150%）。"""
    try:
        hwnd = ctypes.windll.user32.GetDesktopWindow()
        hdc = ctypes.windll.user32.GetWindowDC(hwnd)
        LOGPIXELSX = 88
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        ctypes.windll.user32.ReleaseDC(hwnd, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


# === pyautogui 懒加载 ===
def _pg():
    try:
        import pyautogui
        pyautogui.FAILSAFE = True   # 鼠标甩到 (0,0) 自动停
        pyautogui.PAUSE = 0.1      # 每个操作间隔
        return pyautogui
    except Exception as e:
        raise RuntimeError(f"未安装 pyautogui - {e}")


def _to_physical(x: float, y: float) -> tuple[int, int]:
    """逻辑坐标 → 物理像素坐标。"""
    scale = _screen_scale()
    return int(x * scale), int(y * scale)


# ===================================================================
# 新版 API：显式坐标 + DPI 感知
# ===================================================================

def click(x: int | None = None, y: int | None = None,
          name: str | None = None,
          button: str = "left", clicks: int = 1) -> str:
    """点击（DPI 感知）。

    - 给 x,y：按逻辑坐标点击（自动 *scale 转物理像素）
    - 给 name：先在屏幕上按文字定位（OCR+UI 双引擎），再点击其中心
    - 同时给 name 和 x,y：优先 name
    """
    pyautogui = _pg()

    # 名称解析
    if name:
        try:
            from .screen import find_by_text
            hits = find_by_text(name, use_ocr=True)
            if hits:
                cx, cy = hits[0]["center"]
            else:
                return f"未找到包含「{name}」的文字/控件"
        except Exception as e:
            return f"按名称定位失败：{e}"
    else:
        if x is None or y is None:
            return "错误：必须提供 x,y 坐标或有效的 name"
        cx, cy = x, y

    px, py = _to_physical(cx, cy)
    pyautogui.click(px, py, button=button, clicks=clicks)
    return f"已点击 ({cx},{cy})→物理({px},{py}) [{button}×{clicks}]"


def double_click(x: int | None = None, y: int | None = None,
                 name: str | None = None) -> str:
    """双击（DPI 感知）。"""
    return click(x=x, y=y, name=name, clicks=2)


def right_click(x: int | None = None, y: int | None = None,
                name: str | None = None) -> str:
    """右键点击（DPI 感知）。"""
    return click(x=x, y=y, name=name, button="right")


def drag(x1: int, y1: int, x2: int, y2: int,
         duration: float = 0.5) -> str:
    """拖拽（DPI 感知）。从 (x1,y1) 拖到 (x2,y2)。"""
    pyautogui = _pg()
    px1, py1 = _to_physical(x1, y1)
    px2, py2 = _to_physical(x2, y2)
    pyautogui.moveTo(px1, py1)
    pyautogui.drag(px2 - px1, py2 - py1, duration=duration)
    return f"已拖拽 ({x1},{y1})→({x2},{y2})"


def move(x: int, y: int, duration: float = 0.2) -> str:
    """鼠标移动（DPI 感知）。"""
    pyautogui = _pg()
    px, py = _to_physical(x, y)
    pyautogui.moveTo(px, py, duration=duration)
    return f"鼠标移动到 ({x},{y})→物理({px},{py})"


def scroll(clicks: int, direction: str = "down") -> str:
    """滚动。clicks 正数=向下滚，负数=向上滚。direction 可覆盖。"""
    pyautogui = _pg()
    amount = abs(clicks) * (-1 if direction == "up" else 1)
    pyautogui.scroll(amount)
    return f"已滚动 {direction} {abs(clicks)} 次"


def type_text(text: str, interval: float = 0.02) -> str:
    """输入文本。interval 为每个字符之间的间隔（秒）。"""
    pyautogui = _pg()
    # pyautogui.typewrite 只支持 ASCII；write 是 typewrite 的别名，同样有限制
    # 对中文走 clipboard 粘贴更稳
    try:
        pyautogui.typewrite(text, interval=interval)
    except Exception:
        # fallback 走剪贴板
        try:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            return f"输入失败：{e}"
    return f"已输入 {len(text)} 字符"


def hotkey(*keys: str) -> str:
    """按下组合键，如 hotkey("ctrl","shift","t")。"""
    pyautogui = _pg()
    pyautogui.hotkey(*keys)
    return f"已按下 {'+'.join(keys)}"


def key(combo: str) -> str:
    """按快捷键字符串，如 \"ctrl+s\"、\"enter\"、\"ctrl+shift+esc\"。

    兼容旧接口。与 hotkey() 不同：接收一个字符串，内部解析 + 分隔。
    """
    pyautogui = _pg()
    parts = [p.strip().lower() for p in str(combo).split("+") if p.strip()]
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)
    return f"已按键: {combo}"


def wait(seconds: float) -> str:
    """暂停指定秒数（让前台应用有时间响应）。"""
    time.sleep(seconds)
    return f"已等待 {seconds}s"
