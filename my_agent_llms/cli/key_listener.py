"""run 期间的 esc 打断监听:后台线程 raw-mode 读 stdin,捕获 esc → set event。
非 tty / 不支持 termios 时 no-op。with 块退出时恢复终端 + 收线程。"""
from __future__ import annotations

import sys
import threading

try:
    import termios
    import tty
    import select
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


class EscListener:
    def __init__(self):
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._old = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def __enter__(self) -> "EscListener":
        if not (_HAS_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()):
            return self
        try:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self._old = None
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception:
                return
            if r:
                try:
                    ch = sys.stdin.read(1)
                except Exception:
                    return
                if ch == "\x1b":
                    self._event.set()
                    return

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
            self._thread = None
        if self._old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None
