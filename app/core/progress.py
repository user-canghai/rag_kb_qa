"""统一的终端进度输出：阶段横幅、步骤计时、进度条与流式速率。"""
import os
import shutil
import sys
import time
import unicodedata

_ENABLED = os.getenv("RAG_PROGRESS", "1") != "0"


def _init_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_init_stdout()

_IS_TTY = bool(getattr(sys.stdout, "isatty", lambda: False)()) and _ENABLED
_COLOR = _IS_TTY and not os.getenv("NO_COLOR") and os.getenv("TERM", "") != "dumb"
_WIDTH = max(60, shutil.get_terminal_size((100, 24)).columns)


def _c(code, text):
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else text


def bold(text):
    return _c("1", text)


def dim(text):
    return _c("2", text)


def green(text):
    return _c("32", text)


def yellow(text):
    return _c("33", text)


def cyan(text):
    return _c("36", text)


def fmt_seconds(seconds):
    """把秒数格式化成易读的耗时。"""
    if seconds < 1:
        return "%.0fms" % (seconds * 1000)
    if seconds < 60:
        return "%.2fs" % seconds
    minutes, rest = divmod(seconds, 60)
    return "%dm%04.1fs" % (int(minutes), rest)


def fmt_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024


def _width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width):
    return text + " " * max(0, width - _width(text))


def _flush():
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _write(text, end="\n"):
    if not _ENABLED:
        return
    sys.stdout.write(text + end)
    _flush()


def banner(title, subtitle=""):
    """整段运行的开始横幅。"""
    line = "=" * min(_WIDTH - 1, 78)
    _write(cyan(line))
    _write(cyan("  " + title))
    if subtitle:
        _write(dim("  " + subtitle))
    _write(cyan(line))


def section(title):
    _write("")
    _write(bold(cyan(">> " + title)))


def log(message):
    _write("%s %s" % (dim(time.strftime("%H:%M:%S")), message))


def info(message):
    _write("   " + message)


def ok(message):
    _write("   " + green("[OK] " + message))


def warn(message):
    _write("   " + yellow("[!] " + message))


def kv(pairs, indent=2):
    """对齐打印键值对，中文按两列宽计算。"""
    width = max((_width(str(k)) for k, _ in pairs), default=0)
    for key, value in pairs:
        _write(" " * indent + dim(_pad(str(key), width)) + "  " + str(value))


class Steps:
    """一次请求的分步计时：begin/end 成对使用，finish 打印总耗时。"""

    def __init__(self, title):
        self.title = title
        self.started = time.time()
        self.index = 0
        self._open = None
        _write("")
        _write(bold(cyan("+-- " + title)))

    def begin(self, name, detail=""):
        self.index += 1
        self._open = (self.index, name, time.time())
        line = "| [%d] %s" % (self.index, name)
        if detail:
            line += dim("  " + detail)
        _write(line)
        return self._open[2]

    def note(self, message):
        _write("|     " + dim(message))

    def end(self, note=""):
        if not self._open:
            return 0.0
        index, name, start = self._open
        elapsed = time.time() - start
        self._open = None
        line = "|     +-- " + green(fmt_seconds(elapsed))
        if note:
            line += "  " + note
        _write(line)
        return elapsed

    def finish(self, summary=""):
        total = time.time() - self.started
        line = "+-- 总耗时 " + bold(fmt_seconds(total))
        if summary:
            line += "  " + summary
        _write(cyan(line))
        _write("")
        return total


def _draw_bar(desc, done, total, elapsed, unit, final=False):
    if total:
        ratio = done / float(total)
        width = max(12, min(28, _WIDTH - 46))
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        speed = "  %.1f %s/s" % (done / elapsed, unit) if elapsed > 0.3 else ""
        line = "  %s [%s] %d/%d %3.0f%%  %s%s" % (
            desc, bar, done, total, ratio * 100, fmt_seconds(elapsed), speed)
    else:
        line = "  %s %d %s  %s" % (desc, done, unit, fmt_seconds(elapsed))
    if final:
        sys.stdout.write("\r" + _pad(line, _WIDTH - 1) + "\n")
    else:
        sys.stdout.write("\r" + _pad(line, _WIDTH - 1))
    _flush()


def track(iterable, total=None, desc="处理", unit="项"):
    """带进度条的循环：TTY 原地刷新，日志里按 10% 步进逐行打印。"""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    started = time.time()
    done = 0
    last_draw = 0.0
    next_mark = 0.1
    if _IS_TTY and desc:
        _draw_bar(desc, 0, total, 0.0, unit)
    for item in iterable:
        yield item
        done += 1
        now = time.time()
        if _IS_TTY:
            if now - last_draw >= 0.1:
                _draw_bar(desc, done, total, now - started, unit)
                last_draw = now
        elif _ENABLED and total and done / float(total) >= next_mark:
            info("%s %d/%d %s（%.0f%%，已用 %s）" % (
                desc, done, total, unit, done * 100.0 / total, fmt_seconds(now - started)))
            next_mark += 0.1
    elapsed = time.time() - started
    if _IS_TTY:
        _draw_bar(desc, done, total, elapsed, unit, final=True)
    elif _ENABLED and total:
        ok("%s完成 %d/%d %s，耗时 %s" % (desc, done, total, unit, fmt_seconds(elapsed)))


class StreamMeter:
    """流式生成的字数与速率统计；TTY 下按固定间隔刷新同一行。"""

    def __init__(self, desc="生成中", interval=0.6):
        self.desc = desc
        self.interval = interval
        self.chars = 0
        self.chunks = 0
        self.started = time.time()
        self._last = 0.0

    def update(self, count=1):
        self.chars += count
        self.chunks += 1
        if not _IS_TTY:
            return
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            self._draw()

    def _draw(self):
        elapsed = time.time() - self.started
        if elapsed >= 0.1:
            speed = "%.1f 字/秒" % (self.chars / elapsed)
        else:
            speed = "-- 字/秒"
        line = "  %s ... 已生成 %d 字  %s  %s" % (
            self.desc, self.chars, speed, fmt_seconds(elapsed))
        sys.stdout.write("\r" + _pad(line, _WIDTH - 1))
        _flush()

    def finish(self, note=""):
        elapsed = time.time() - self.started
        rate = self.chars / elapsed if elapsed > 0 else 0.0
        if _IS_TTY:
            sys.stdout.write("\r" + " " * (_WIDTH - 1) + "\r")
        line = "  生成完成：%d 字，用时 %s，平均 %.1f 字/秒" % (
            self.chars, fmt_seconds(elapsed), rate)
        if note:
            line += "，" + note
        _write(green(line))
        return elapsed
