# -*- coding: utf-8 -*-
"""屏幕感知：把真实桌面转成文本快照或结构化 JSON，让无视觉模型也能理解屏幕。

升级亮点：
- DPI 感知（_screen_scale），解决 pywinauto 逻辑像素 vs pyautogui 物理像素的偏差
- 可选 RapidOCR：OCR + UI Automation 双引擎融合
- 截图自动存到 hs-sidecar-guard/computer_capture/ 下，timestamp 命名
- 旧函数（screenshot_text / save_screenshot / find_element_coords）保持向后兼容
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from typing import Any

# === DPI 感知 ===
import ctypes


def _screen_scale() -> float:
    """Windows DPI 缩放因子（1.0=100%, 1.25=125%, 1.5=150%）。

    pywinauto 返回逻辑像素坐标，pyautogui/mss 在不同 DPI 下行为不一致——
    统一用本函数获取缩放比，在需要物理像素时做 scale 转换。
    """
    try:
        hwnd = ctypes.windll.user32.GetDesktopWindow()
        hdc = ctypes.windll.user32.GetWindowDC(hwnd)
        LOGPIXELSX = 88
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        ctypes.windll.user32.ReleaseDC(hwnd, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


def _screen_size_physical() -> tuple[int, int]:
    """返回主显示器物理像素尺寸 (w, h)。"""
    try:
        sm_cxscreen = 0
        sm_cyscreen = 1
        w = ctypes.windll.user32.GetSystemMetrics(sm_cxscreen)
        h = ctypes.windll.user32.GetSystemMetrics(sm_cyscreen)
        scale = _screen_scale()
        # GetSystemMetrics 在 DPI 感知进程里返回物理像素；在未感知进程返回逻辑像素
        # 保险起见，按 scale 补一次
        if scale > 1.0:
            return int(w * scale), int(h * scale)
        return w, h
    except Exception:
        return 1920, 1080


# === pywinauto 控件抽取 ===
_CONTROL_KEEP = {"Button", "Edit", "ListItem", "TabItem", "MenuItem",
                 "CheckBox", "RadioButton", "ComboBox", "Hyperlink",
                 "TreeItem", "Document", "TitleBar", "Text", "Window"}


def _el(x, n):
    return getattr(x, n, None) or ""


def _center(elm) -> tuple[int, int] | None:
    try:
        r = elm.rectangle()
        return (int((r.left + r.right) / 2), int((r.top + r.bottom) / 2))
    except Exception:
        return None


def _rect_repr(elm) -> str:
    try:
        r = elm.rectangle()
        return f"({int(r.left)},{int(r.top)},{int(r.right)},{int(r.bottom)})"
    except Exception:
        return ""


def _is_real(elm) -> bool:
    try:
        r = elm.rectangle()
        return (r.right - r.left) > 20 and (r.bottom - r.top) > 20
    except Exception:
        return False


def _extract_controls(max_elements: int = 150,
                      scale: float = 1.0,
                      only_focus: bool = True) -> list[dict]:
    """用 pywinauto 抽 UI 控件，返回结构化列表。

    每项: {"type": str, "name": str, "center": [cx, cy], "rect": [l,t,r,b]}
    """
    try:
        from pywinauto import Desktop
    except Exception:
        return []

    controls: list[dict] = []
    try:
        desc = Desktop(backend="uia")
        wins = [w for w in desc.windows() if w.is_visible() and _is_real(w)]
    except Exception:
        return []

    picked = None
    if only_focus:
        for w in wins:
            try:
                if w.has_focus() or w.is_active():
                    picked = w
                    break
            except Exception:
                continue
        if picked is None and wins:
            picked = wins[0]

    if picked is not None:
        try:
            for child in picked.descendants():
                try:
                    ctype = _el(child, "control_type")
                    if ctype not in _CONTROL_KEEP:
                        continue
                    name = (_el(child, "friendly_class_name")
                            or str(_el(child, "class_name")) or "")
                    try:
                        name = child.window_text() or name
                    except Exception:
                        pass
                    c = _center(child)
                    if not c:
                        continue
                    try:
                        r = child.rectangle()
                        rect = [int(r.left), int(r.top), int(r.right), int(r.bottom)]
                    except Exception:
                        rect = [0, 0, 0, 0]
                    controls.append({
                        "type": ctype,
                        "name": str(name),
                        "center": [c[0], c[1]],
                        "rect": rect,
                    })
                    if len(controls) >= max_elements:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # 附加其他顶层窗口标题
    for w in wins:
        if w is picked:
            continue
        try:
            if not w.is_visible():
                continue
            t = w.window_text() or ""
            if t:
                c = _center(w)
                if c:
                    controls.append({
                        "type": "Window",
                        "name": t,
                        "center": [c[0], c[1]],
                        "rect": [0, 0, 0, 0],
                    })
                    if len(controls) >= max_elements:
                        break
        except Exception:
            continue

    return controls


# === 保存路径（hs-sidecar-guard/computer_capture/） ===
_CAPTURE_DIR = Path(__file__).resolve().parent.parent.parent / "computer_capture"


def _capture_path(ext: str = ".png") -> Path:
    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    return _CAPTURE_DIR / f"screenshot_{int(time.time())}{ext}"


# === 融合 OCR 的截图 ===
def screenshot_with_ocr(dst: str | None = None,
                        use_rapidocr: bool = True,
                        max_elements: int = 150) -> dict:
    """截图 + OCR + UI Automation 三合一。

    返回 dict:
        image_path:   str  截图保存路径
        ocr_text:     list[{"text","box","confidence"}]
        ui_controls:  list[{"type","name","center","rect"}]
        scale:        float  DPI 缩放因子
        screen_size:  [w, h] 物理像素
    """
    scale = _screen_scale()

    # a) mss 截全屏
    img = None
    try:
        import mss
        from PIL import Image
        pw, ph = _screen_size_physical()
        monitor = {"top": 0, "left": 0, "width": pw, "height": ph}
        with mss.mss() as sct:
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as e:
        img = None

    # b) OCR (rapidocr-onnxruntime)
    ocr_results: list[dict] = []
    if use_rapidocr and img is not None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
            result, _ = ocr(img)
            if result:
                for item in result:
                    box, text, conf = item
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    ocr_results.append({
                        "text": str(text),
                        "box": [min(xs), min(ys), max(xs), max(ys)],
                        "confidence": float(conf),
                    })
        except Exception as e:
            ocr_results = [{
                "text": f"[OCR_FAIL: {type(e).__name__}]",
                "box": [0, 0, 0, 0],
                "confidence": 0.0,
            }]

    # c) UI Automation
    ui_controls = _extract_controls(max_elements, scale)

    # d) 保存
    image_path = ""
    if img is not None:
        if dst is None:
            dst = str(_capture_path(".png"))
        else:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(dst)
            image_path = dst
        except Exception:
            image_path = ""

    pw, ph = _screen_size_physical()
    return {
        "image_path": image_path,
        "ocr_text": ocr_results,
        "ui_controls": ui_controls,
        "scale": scale,
        "screen_size": [pw, ph],
    }


# === 融合 OCR + UI 的文本快照（给纯文本模型用） ===
def screenshot_snapshot(max_elements: int = 100,
                        include_ocr: bool = True) -> str:
    """返回融合后的文本快照：UI 控件 + OCR 文字区域 + 坐标。"""
    result = screenshot_with_ocr(use_rapidocr=include_ocr,
                                 max_elements=max_elements)
    lines: list[str] = []
    lines.append(
        f"[SCREEN] size={result['screen_size']} scale={result['scale']:.2f}"
    )

    # UI Automation 控件（优先，更结构化）
    for c in result["ui_controls"]:
        ctr = c.get("center", [0, 0])
        lines.append(
            f"  UI {c['type']} \"{c['name']}\" center=({ctr[0]},{ctr[1]})"
        )

    # OCR 文字（补 UI Automation 没抓到的）
    for o in result["ocr_text"][:80]:
        box = o.get("box") or []
        if box and len(box) == 4:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            conf = o.get("confidence", 0)
            lines.append(
                f"  OCR \"{o['text']}\" center=({cx},{cy}) conf={conf:.2f}"
            )

    return "\n".join(lines)


# === 按文字查找（优先 OCR，fallback UI） ===
def find_by_text(query: str, use_ocr: bool = True) -> list[dict]:
    """在屏幕上查找包含 query 的文字区域（OCR + UI Automation 双引擎）。

    返回:
        [{"source": "ocr"|"ui", "text": str, "center": [x,y], ...}, ...]
    """
    result = screenshot_with_ocr(use_rapidocr=use_ocr, max_elements=200)
    hits: list[dict] = []
    q = query.lower()

    for o in result["ocr_text"]:
        if q in o["text"].lower():
            box = o["box"]
            hits.append({
                "source": "ocr",
                "text": o["text"],
                "center": [int((box[0] + box[2]) / 2),
                           int((box[1] + box[3]) / 2)],
                "confidence": o["confidence"],
            })

    for c in result["ui_controls"]:
        if q in c["name"].lower():
            hits.append({
                "source": "ui",
                "text": c["name"],
                "center": c["center"],
                "type": c["type"],
            })

    return hits


# ===================================================================
# 旧函数 —— 保持向后兼容，内部复用新实现
# ===================================================================

def screenshot_text(max_elements: int = 300) -> str:
    """返回前台焦点窗口的文本快照（兼容旧接口）。

    现在调用 screenshot_snapshot 融合 OCR + UI Automation。
    """
    return screenshot_snapshot(max_elements=max_elements, include_ocr=True)


def save_screenshot(dst: str | None = None) -> str:
    """用 mss 截全屏存为 png，返回路径（兼容旧接口）。"""
    result = screenshot_with_ocr(dst=dst, use_rapidocr=False)
    if result["image_path"]:
        return f"已截图: {result['image_path']}"
    return "截图失败：无可用图像"


def find_element_coords(name: str,
                        control_type: str | None = None) -> str:
    """按控件名/文字查找，返回中心坐标（兼容旧接口）。"""
    hits = find_by_text(name, use_ocr=True)
    if not hits:
        return (
            f"未找到包含「{name}」的元素"
            "（可先用 screenshot_snapshot 查看屏幕再试）"
        )
    h = hits[0]
    src = h.get("source", "?")
    center = h["center"]
    extra = ""
    if control_type and h.get("type"):
        if control_type.lower() != h["type"].lower():
            # 继续找匹配类型的
            for hh in hits[1:]:
                if hh.get("type", "").lower() == control_type.lower():
                    h = hh
                    src = h.get("source", "?")
                    center = h["center"]
                    if h.get("type"):
                        extra = f" 类型={h['type']}"
                    break
    if h.get("type") and control_type and h["type"].lower() != control_type.lower():
        return (
            f"已找到但控件类型不匹配：{h['type']} 含「{name}」"
            f" center={center}"
        )
    return (
        f"[find:{src}] \"{h.get('text','')}\" center={center}{extra}"
        f" 类型={h.get('type','ocr-text')}"
    )
