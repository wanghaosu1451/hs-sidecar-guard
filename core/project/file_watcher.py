"""文件变动监听（轻量：轮询 mtime + 大小）。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .workspace import Workspace


class FileWatcher:
    def __init__(self, workspace: Workspace, interval: float = 1.0):
        self.workspace = workspace
        self.interval = interval
        self._state: dict[str, tuple[float, int]] = {}
        self.on_change: Callable[[Path], None] | None = None

    def snapshot(self) -> None:
        self._state = self._scan()

    def _scan(self) -> dict[str, tuple[float, int]]:
        state: dict[str, tuple[float, int]] = {}
        for p in self.workspace.iter_files():
            try:
                st = p.stat()
                state[str(p)] = (st.st_mtime, st.st_size)
            except OSError:
                pass
        return state

    def poll(self) -> None:
        current = self._scan()
        for path, info in current.items():
            prev = self._state.get(path)
            if prev is not None and prev != info and self.on_change:
                self.on_change(Path(path))
        # 记录新增文件
        for path in current:
            if path not in self._state and self.on_change:
                self.on_change(Path(path))
        self._state = current

    def run(self, stop: Callable[[], bool] | None = None, poll_ticks: int = 0) -> None:
        """阻塞轮询。stop 返回 True 时退出；poll_ticks<=0 表示无限。"""
        ticks = 0
        while not (stop and stop()):
            self.poll()
            ticks += 1
            if 0 <= poll_ticks <= ticks:
                break
            time.sleep(self.interval)