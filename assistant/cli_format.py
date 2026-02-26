from __future__ import annotations

import json
import re
import sys
from typing import Any


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def _fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


# VS Code Dark+ inspired colors
MD_TEXT = _fg(212, 212, 212)
MD_TITLE_1 = _fg(86, 156, 214)
MD_TITLE_2 = _fg(78, 201, 176)
MD_TITLE_3 = _fg(197, 134, 192)
MD_TITLE_N = _fg(220, 220, 170)
MD_CODE = _fg(206, 145, 120)
MD_LINK = _fg(79, 193, 255)
MD_URL = _fg(156, 220, 254)
MD_QUOTE = _fg(106, 153, 85)
MD_LIST_MARK = _fg(220, 220, 170)
MD_BOLD = _fg(220, 220, 170)
MD_ITALIC = _fg(197, 134, 192)
MD_FENCE = _fg(128, 128, 128)
TOOL_LABEL = _fg(206, 145, 120)


class MarkdownAnsiRenderer:
    def __init__(self) -> None:
        self.in_code_block = False

    @staticmethod
    def _format_inline(text: str, base_color: str) -> str:
        out = text
        out = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: f"{MD_LINK}{m.group(1)}{RESET}{base_color} {MD_URL}({m.group(2)}){RESET}{base_color}",
            out,
        )
        out = re.sub(
            r"`([^`]+)`",
            lambda m: f"{MD_CODE}`{m.group(1)}`{RESET}{base_color}",
            out,
        )
        out = re.sub(
            r"\*\*([^*\n]+)\*\*",
            lambda m: f"{MD_BOLD}{BOLD}{m.group(1)}{RESET}{base_color}",
            out,
        )
        out = re.sub(
            r"(?<!\*)\*([^*\n]+)\*(?!\*)",
            lambda m: f"{MD_ITALIC}{m.group(1)}{RESET}{base_color}",
            out,
        )
        return out

    def render_line(self, line: str) -> str:
        stripped = line.lstrip()

        if stripped.startswith("```"):
            self.in_code_block = not self.in_code_block
            return f"{DIM}{MD_FENCE}{line}{RESET}"

        if self.in_code_block:
            return f"{MD_CODE}{line}{RESET}"

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if level == 1:
                color = MD_TITLE_1
            elif level == 2:
                color = MD_TITLE_2
            elif level == 3:
                color = MD_TITLE_3
            else:
                color = MD_TITLE_N
            title = self._format_inline(title, color)
            return f"{BOLD}{color}{title}{RESET}"

        quote = re.match(r"^\s*>\s?(.*)$", line)
        if quote:
            body = self._format_inline(quote.group(1), MD_QUOTE)
            return f"{MD_QUOTE}> {body}{RESET}"

        listing = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if listing:
            indent, marker, body = listing.groups()
            body = self._format_inline(body, MD_TEXT)
            return f"{MD_TEXT}{indent}{MD_LIST_MARK}{marker}{RESET}{MD_TEXT} {body}{RESET}"

        return f"{MD_TEXT}{self._format_inline(line, MD_TEXT)}{RESET}"

    def render_text(self, text: str) -> str:
        if not text:
            return ""
        parts: list[str] = []
        for raw_line in text.splitlines(keepends=True):
            if raw_line.endswith("\n"):
                line = raw_line[:-1]
                parts.append(self.render_line(line) + "\n")
            else:
                parts.append(self.render_line(raw_line))
        return "".join(parts)


def _split_thinking_and_answer(response: str) -> tuple[str, str]:
    think_chunks = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL | re.IGNORECASE)
    thinking = "\n".join(chunk.strip() for chunk in think_chunks if chunk.strip()).strip()
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    answer = answer.replace("<think>", "").replace("</think>", "").strip()
    return thinking, answer


def _render_markdown(text: str) -> str:
    return MarkdownAnsiRenderer().render_text(text)


def extract_answer_text(response: str) -> str:
    _, answer = _split_thinking_and_answer(response)
    return answer


def _print_markdown(text: str) -> None:
    rendered = _render_markdown(text)
    if not rendered:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    sys.stdout.write(rendered)
    if not rendered.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_answer_only(answer: str) -> None:
    _print_markdown(answer)


def print_formatted_output(response: str) -> None:
    thinking, answer = _split_thinking_and_answer(response)

    if thinking:
        print(f"{BOLD}{CYAN}Thinking...{RESET}")
        for line in thinking.splitlines() or [""]:
            print(f"{DIM}{BLUE}{line}{RESET}")
        print(f"{BOLD}{CYAN}...done thinking.{RESET}")
        print("")

    _print_markdown(answer)


class StreamRenderer:
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._pending_think_end = False
        self._buffer = ""
        self._started = False
        self._last_char = ""
        self._answer_buffer = ""
        self._md = MarkdownAnsiRenderer()
        self.has_output = False
        self.has_answer_output = False

    def _write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self.has_output = True
        self._last_char = text[-1]

    def _ensure_newline(self) -> None:
        if self.has_output and self._last_char != "\n":
            self._write("\n")

    def _start_answer(self) -> None:
        if self._started:
            return
        self._started = True
        self.has_answer_output = True

    def _write_answer(self, text: str) -> None:
        if not text:
            return
        self._start_answer()
        self._answer_buffer += text
        while "\n" in self._answer_buffer:
            line, self._answer_buffer = self._answer_buffer.split("\n", 1)
            rendered = self._md.render_line(line)
            self._write(rendered + "\n")

    def _flush_answer(self) -> None:
        if self._answer_buffer:
            rendered = self._md.render_line(self._answer_buffer)
            self._write(rendered)
            self._answer_buffer = ""

    def _end_answer(self) -> None:
        if self._started:
            self._flush_answer()
            self._started = False

    def _start_think(self) -> None:
        if self._in_think:
            return
        self._end_answer()
        self._write(f"{BOLD}{CYAN}Thinking...{RESET}\n{DIM}{BLUE}")
        self._in_think = True
        self._pending_think_end = False

    def _end_think(self) -> None:
        if not self._in_think:
            return
        self._write(f"{RESET}\n{BOLD}{CYAN}...done thinking.{RESET}\n\n")
        self._in_think = False
        self._started = False
        self._pending_think_end = False

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk

        while self._buffer:
            if self._buffer.startswith(self.THINK_CLOSE):
                self._buffer = self._buffer[len(self.THINK_CLOSE) :]
                continue

            if self._pending_think_end:
                stripped = self._buffer.lstrip()
                if not stripped:
                    return
                if stripped.startswith(self.THINK_OPEN):
                    lead_ws = len(self._buffer) - len(stripped)
                    self._buffer = self._buffer[lead_ws + len(self.THINK_OPEN) :]
                    self._pending_think_end = False
                    continue
                if stripped.startswith(self.THINK_CLOSE):
                    lead_ws = len(self._buffer) - len(stripped)
                    self._buffer = self._buffer[lead_ws + len(self.THINK_CLOSE) :]
                    continue
                self._end_think()
                continue

            if self._in_think:
                idx = self._buffer.find(self.THINK_CLOSE)
                if idx == -1:
                    safe = max(0, len(self._buffer) - (len(self.THINK_CLOSE) - 1))
                    if safe == 0:
                        return
                    cleaned = self._buffer[:safe].replace(self.THINK_OPEN, "").replace(self.THINK_CLOSE, "")
                    self._write(cleaned)
                    self._buffer = self._buffer[safe:]
                    return
                cleaned = self._buffer[:idx].replace(self.THINK_OPEN, "").replace(self.THINK_CLOSE, "")
                self._write(cleaned)
                self._buffer = self._buffer[idx + len(self.THINK_CLOSE) :]
                self._pending_think_end = True
                continue

            idx = self._buffer.find(self.THINK_OPEN)
            if idx == -1:
                safe = max(0, len(self._buffer) - (len(self.THINK_OPEN) - 1))
                if safe == 0:
                    return
                cleaned = self._buffer[:safe].replace(self.THINK_CLOSE, "")
                if cleaned:
                    self._write_answer(cleaned)
                self._buffer = self._buffer[safe:]
                return

            before = self._buffer[:idx].replace(self.THINK_CLOSE, "")
            if before:
                self._write_answer(before)
            self._buffer = self._buffer[idx + len(self.THINK_OPEN) :]
            self._start_think()

    def finish(self) -> None:
        self.prepare_tool_output()
        self._ensure_newline()

    def prepare_tool_output(self) -> None:
        if self._pending_think_end:
            self._end_think()

        if self._buffer:
            if self._in_think:
                cleaned = self._buffer.replace(self.THINK_OPEN, "").replace(self.THINK_CLOSE, "")
                self._write(cleaned)
            else:
                cleaned = self._buffer.replace(self.THINK_CLOSE, "")
                if cleaned:
                    self._write_answer(cleaned)
            self._buffer = ""

        if self._in_think:
            self._end_think()

        self._end_answer()
        self._ensure_newline()


def print_tool_event(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    args_text = json.dumps(args, ensure_ascii=False)
    result_text = json.dumps(result, ensure_ascii=False)
    if len(result_text) > 420:
        result_text = result_text[:417] + "..."
    print(f"{BOLD}{MAGENTA}tool>{RESET} {TOOL_LABEL}{name}{RESET}")
    print(f"{DIM}{BLUE}args:{RESET} {args_text}")
    print(f"{DIM}{BLUE}result:{RESET} {result_text}")
