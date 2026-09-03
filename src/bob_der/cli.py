from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .ollama import OllamaError, OllamaRunner
from .prompts import DEFAULT_SUBAGENT_SYSTEM_PROMPT
from .tools import EYES_MODEL

PRIMARY_MODEL = "nemotron-3-super:cloud"
SUBAGENT_MODEL = "gpt-oss:20b-cloud"


class Console:
    def __init__(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.live = sys.stdout.isatty() and not quiet
        self._stream_buffer = ""
        self._stream_text = ""
        self._streamed_final = False
        self._status_visible = False

    def paint(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def clear_status(self) -> None:
        if self._status_visible:
            print("\r\033[2K", end="", flush=True)
            self._status_visible = False

    def answer(self, message: str) -> None:
        self.clear_status()
        if self._streamed_final:
            print()
            self._streamed_final = False
            self._stream_text = ""
            return
        print(self.paint("1", message))

    def event(self, event: str, data: dict[str, object]) -> None:
        if self.quiet or event in {"thinking", "final"}:
            return
        if event == "model_start":
            self.clear_status()
            self._stream_buffer = ""
            self._stream_text = ""
            self._streamed_final = False
            if self.live:
                print(self.paint("2;35", "✻ Thinking…"), end="", flush=True)
                self._status_visible = True
        elif event == "model_chunk":
            if not self.live:
                return
            self._stream_buffer += str(data.get("chunk", ""))
            message = partial_final_message(self._stream_buffer)
            if message is None:
                return
            if not self._streamed_final:
                self.clear_status()
                self._streamed_final = True
            if len(message) > len(self._stream_text):
                print(message[len(self._stream_text) :], end="", flush=True)
                self._stream_text = message
        elif event == "tool_start":
            self.clear_status()
            tool = str(data["tool"])
            args = data.get("args", {})
            thought = str(data.get("thought", ""))
            if self.verbose and thought:
                print(self.paint("2", f"  {thought}"))
            if tool == "shell" and isinstance(args, dict):
                label = f"Bash({args.get('command', '')})"
            elif tool == "subagent" and isinstance(args, dict):
                task = str(args.get("task", ""))
                suffix = "…" if len(task) > 100 else ""
                label = f"Task({task[:100]}{suffix})"
            elif tool == "describe_image" and isinstance(args, dict):
                target = args.get("path") or args.get("url") or "images"
                label = f"Eyes({target})"
            elif isinstance(args, dict):
                target = args.get("path", "")
                pretty = {
                    "read_file": "Read",
                    "write_file": "Write",
                    "replace_in_file": "Edit",
                    "list_files": "Glob",
                    "grep": "Grep",
                    "find_files": "Find",
                    "git_diff": "Diff",
                    "test_run": "Test",
                    "todo_scan": "Todos",
                    "size_report": "Size",
                }.get(tool, tool)
                label = f"{pretty}({target})"
            else:
                label = tool
            print(self.paint("35", f"● {label}"), flush=True)
        elif event == "fallback":
            self.clear_status()
            print(
                self.paint(
                    "33",
                    f"↻ {data.get('from_model')} was slow; "
                    f"{data.get('to_model')} is taking over",
                ),
                flush=True,
            )
        elif event == "tool_end":
            result = data.get("result", {})
            if isinstance(result, dict):
                if not result.get("ok"):
                    print(self.paint("31", f"  ⎿ Error: {result.get('error', 'tool failed')}"))
                elif data.get("tool") == "subagent":
                    print(
                        self.paint(
                            "2",
                            f"  ⎿ {result.get('model')} completed in {result.get('steps')} steps",
                        )
                    )
                elif self.verbose and "output" in result:
                    output = str(result.get("output", "")).rstrip()
                    if output:
                        print(self.paint("2", "  ⎿ ") + output.replace("\n", "\n    "))


def partial_final_message(response: str) -> str | None:
    """Decode the generated portion of a final action's JSON message string."""
    response = action_part(response)
    if not response:
        return None
    tool = re.search(r'"tool"\s*:\s*"final"', response)
    if tool is None:
        return None
    message = re.search(r'"message"\s*:\s*"', response[tool.end() :])
    if message is None:
        return None
    raw = response[tool.end() + message.end() :]
    output: list[str] = []
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == '"':
            break
        if char != "\\":
            output.append(char)
            index += 1
            continue
        if index + 1 >= len(raw):
            break
        escaped = raw[index + 1]
        if escaped == "u":
            digits = raw[index + 2 : index + 6]
            if len(digits) < 4 or not all(value in "0123456789abcdefABCDEF" for value in digits):
                break
            output.append(chr(int(digits, 16)))
            index += 6
            continue
        output.append(escapes.get(escaped, escaped))
        index += 2
    return "".join(output)


def action_part(response: str) -> str:
    """Return only the action portion after Ollama's thinking stream."""
    if "Thinking..." not in response:
        return response
    end = response.rfind("...done thinking.")
    if end < 0:
        return ""
    return response[end + len("...done thinking.") :].lstrip("\r\n")


def partial_thinking(response: str) -> str:
    """Return all reasoning generated so far, without Ollama's markers."""
    start = response.find("Thinking...")
    if start < 0:
        return ""
    content = response[start + len("Thinking...") :].lstrip("\r\n")
    end = content.find("...done thinking.")
    if end >= 0:
        content = content[:end]
    return content.rstrip("\r\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="bob-der",
        description="Autonomous Ollama-powered coding agent for the terminal.",
    )
    result.add_argument("task", nargs="*", help="task to complete; omit for interactive mode")
    result.add_argument("--subagent-steps", type=int, default=25)
    result.add_argument(
        "--subagent-prompt",
        default=DEFAULT_SUBAGENT_SYSTEM_PROMPT,
        help="custom system instructions for the fixed gpt-oss subagent",
    )
    modes = result.add_mutually_exclusive_group()
    modes.add_argument(
        "--plan",
        action="store_true",
        help="one-shot plan mode: inspect with read-only tools without editing files",
    )
    modes.add_argument(
        "--deep",
        action="store_true",
        help="unbounded reason, plan, revise, then execute workflow",
    )
    result.add_argument("-C", "--directory", default=".", help="working directory")
    result.add_argument("--max-steps", type=int, default=50, help="0 means unlimited (default: 50)")
    result.add_argument("--model-timeout", type=int, default=0, help="seconds; 0 means unlimited (default)")
    result.add_argument(
        "--fallback-after",
        type=float,
        default=0.0,
        help="optional seconds before subagent takeover; 0 disables (default)",
    )
    result.add_argument("--ollama", default="ollama", help="path to the Ollama executable")
    result.add_argument("-v", "--verbose", action="store_true", help="show thoughts and tool output")
    result.add_argument("-q", "--quiet", action="store_true", help="only print the final response")
    updates = result.add_mutually_exclusive_group()
    updates.add_argument(
        "--check-update", action="store_true", help="check GitHub for a newer release"
    )
    updates.add_argument(
        "--update",
        action="store_true",
        help="download and verify the latest Ubuntu package",
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return result


def make_agent(args: argparse.Namespace, console: Console) -> Agent:
    workspace = Path(args.directory).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Working directory does not exist: {workspace}")
    if shutil.which(args.ollama) is None:
        raise OllamaError(f"Ollama executable not found: {args.ollama}")
    runner = OllamaRunner(
        PRIMARY_MODEL,
        args.ollama,
        args.model_timeout,
        first_token_timeout=max(0, args.fallback_after),
    )
    subagent_runner = OllamaRunner(SUBAGENT_MODEL, args.ollama, args.model_timeout)
    return Agent(
        runner,
        workspace,
        args.max_steps,
        console.event,
        subagent_runner=subagent_runner,
        subagent_max_steps=args.subagent_steps,
        subagent_system_prompt=args.subagent_prompt,
    )


def basic_interactive(agent: Agent, console: Console) -> int:
    width = 62
    print(console.paint("1;35", "╭" + "─" * width + "╮"))
    print(
        console.paint("1;35", "│")
        + console.paint("1", "  bob-der")
        + " " * (width - 9)
        + console.paint("1;35", "│")
    )
    print(
        console.paint("1;35", "│")
        + f"  Primary   {agent.runner.model}".ljust(width)
        + console.paint("1;35", "│")
    )
    subagent = agent.subagent_runner.model if agent.subagent_runner else "disabled"
    print(
        console.paint("1;35", "│")
        + f"  Subagent  {subagent}".ljust(width)
        + console.paint("1;35", "│")
    )
    print(
        console.paint("1;35", "│")
        + f"  Eyes      {EYES_MODEL}".ljust(width)
        + console.paint("1;35", "│")
    )
    print(
        console.paint("1;35", "│")
        + f"  {agent.workspace}".ljust(width)
        + console.paint("1;35", "│")
    )
    print(console.paint("1;35", "╰" + "─" * width + "╯"))
    print(console.paint("2", "  /help for commands · Ctrl+C interrupts the current task"))
    while True:
        try:
            request = input("\n" + console.paint("1;35", "❯ ")).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
            continue
        if not request:
            continue
        if request in {"/exit", "/quit"}:
            return 0
        if request == "/clear":
            agent.clear()
            print("Conversation cleared.")
            continue
        if request == "/help":
            print("/clear  clear conversation context")
            print("/model  show the primary and subagent models")
            print("/exit   leave bob-der")
            print("\nEnter any coding task to start an autonomous run.")
            continue
        if request == "/model":
            print(f"Primary:  {agent.runner.model}")
            print(
                f"Subagent: {agent.subagent_runner.model if agent.subagent_runner else 'disabled'}"
            )
            print(f"Eyes:     {EYES_MODEL}")
            continue
        try:
            answer = agent.run(request)
            console.answer(answer)
        except KeyboardInterrupt:
            console.clear_status()
            print("\nTask interrupted.", file=sys.stderr)
        except (OllamaError, RuntimeError) as exc:
            console.clear_status()
            print(console.paint("31", f"Error: {exc}"), file=sys.stderr)


def interactive(agent: Agent, console: Console) -> int:
    try:
        from .tui import run_tui
    except ImportError:
        return basic_interactive(agent, console)
    return run_tui(agent)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    console = Console(verbose=args.verbose, quiet=args.quiet)
    if args.check_update or args.update:
        from .updater import UpdateError, fetch_update, format_outcome

        try:
            outcome = fetch_update(download=args.update)
        except UpdateError as exc:
            print(f"bob-der: {exc}", file=sys.stderr)
            return 1
        print(format_outcome(outcome))
        return 0
    try:
        agent = make_agent(args, console)
        if not args.task:
            return interactive(agent, console)
        answer = agent.run(
            " ".join(args.task),
            plan_mode=args.plan,
            deep_mode=args.deep,
        )
        console.answer(answer)
        return 0
    except KeyboardInterrupt:
        console.clear_status()
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (OllamaError, RuntimeError, ValueError) as exc:
        console.clear_status()
        print(f"bob-der: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
