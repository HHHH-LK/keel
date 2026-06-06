"""run 期间的 esc 打断监听:后台线程 raw-mode 读 stdin,捕获 esc → set event。
非 tty / 不支持 termios 时 no-op。with 块退出时恢复终端 + 收线程。

pause()/resume():审批框(prompt_toolkit Application)会独占 stdin,期间必须
暂停本监听的读取,否则两个 reader 抢同一 fd → 审批按键被本线程吃掉、审批框卡住。
"""
from __future__ import annotations

import logging
import sys
import threading

try:
    import termios
    import tty
    import select
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

logger = logging.getLogger(__name__)


class EscListener:
    def __init__(self):
        self._event = threading.Event()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        self._old = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def pause(self) -> None:
        """暂停读取 stdin —— 审批框等独占 stdin 的交互前调用,避免抢 fd。"""
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

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
            if self._paused.is_set():          # 审批等交互独占 stdin 期间不读
                self._stop.wait(0.05)
                continue
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception:
                logger.debug("EscListener select 失败,监听退出", exc_info=True)
                return
            if r:
                try:
                    ch = sys.stdin.read(1)
                except Exception:
                    logger.debug("EscListener read 失败,监听退出", exc_info=True)
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
