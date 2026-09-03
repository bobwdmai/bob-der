from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import beta_tools

EYES_MODEL = "gemma4:31b-cloud"

# These beta actions require CLI integration or would create an additional model,
# so they intentionally remain unavailable in bob-der's one-main/one-subagent design.
INCOMPATIBLE_BETA_TOOLS = {
    "ask_user",
    "compact_conversation",
    "sub_ai",
}

PLAN_READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "list_dir",
        "grep",
        "find_files",
        "git_diff",
        "todo_scan",
        "size_report",
        "browse",
        "secret_scan",
        "diff_files",
        "hash_file",
        "gamepad",
        "git_log",
        "git_blame",
        "ps_list",
        "disk_usage",
        "journalctl",
        "dns_lookup",
        "ping",
        "port_scan",
        "docker_ps",
        "docker_logs",
        "clipboard_get",
        "head_file",
        "tail_file",
        "json_query",
        "base64_encode",
        "base64_decode",
        "uuid_gen",
        "math_eval",
        "watch_file",
        "archive_list",
        "env_get",
        "regex_test",
        "csv_query",
        "image_info",
        "network_info",
        "uptime_info",
        "ssl_check",
        "whois_lookup",
        "top_procs",
        "cron_list",
        "ast_analyze",
    }
)


def available_tool_schemas() -> list[dict[str, Any]]:
    """Return the complete compatible beta catalog, including dynamic tools."""
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema in beta_tools.TOOL_SCHEMAS:
        if not isinstance(schema, dict):
            continue
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in INCOMPATIBLE_BETA_TOOLS
            or name in seen
        ):
            continue
        parameters = function.get("parameters", {})
        if parameters is not None and not isinstance(parameters, dict):
            continue
        schemas.append(schema)
        seen.add(name)
    return schemas


def compact_tool_catalog() -> str:
    """Build a small prompt catalog so the large tool set does not add much latency."""
    lines: list[str] = []
    for schema in available_tool_schemas():
        function = schema.get("function", {})
        name = str(function.get("name", ""))
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if not isinstance(properties, dict):
            properties = {}
        required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
        rendered = ", ".join(
            f"{key}{'*' if key in required else ''}" for key in properties
        )
        lines.append(f"- {name}({rendered})")
    return "\n".join(lines)


class ToolError(RuntimeError):
    pass


@dataclass
class ToolExecutor:
    workspace: Path
    max_output_chars: int = 40_000

    def __post_init__(self) -> None:
        self.workspace = self.workspace.expanduser().resolve()
        self.max_output_chars = max(1_024, int(self.max_output_chars))

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool without allowing tool failures to escape the boundary."""
        if not isinstance(name, str) or not name.strip():
            return self._failure("Tool name must be a non-empty string")
        if not isinstance(args, dict):
            return self._failure("Tool arguments must be a JSON object")
        name = name.strip()
        methods = {
            "shell": self.shell,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "list_files": self.list_files,
            "grep": self.grep,
            "find_files": self.find_files,
            "git_diff": self.git_diff,
            "test_run": self.test_run,
            "todo_scan": self.todo_scan,
            "size_report": self.size_report,
        }
        method = methods.get(name)
        if method is None:
            schemas = {
                str(schema.get("function", {}).get("name", "")): schema
                for schema in available_tool_schemas()
            }
            schema = schemas.get(name)
            if schema is None:
                return self._failure(f"Unknown tool: {name}")
            parameters = schema.get("function", {}).get("parameters", {})
            required = parameters.get("required", []) if isinstance(parameters, dict) else []
            missing = [key for key in required if key not in args]
            if missing:
                return self._failure(
                    f"{name} is missing required argument(s): {', '.join(missing)}"
                )
            try:
                config = None
                if name == "describe_image":
                    # Vision is a fixed capability, not a recursively autonomous
                    # subagent.
                    from types import SimpleNamespace

                    config = SimpleNamespace(
                        vision_model=EYES_MODEL,
                        vision_max_px=1024,
                        vision_cache=True,
                        ollama_host="http://localhost:11434",
                    )
                    args = {**args, "model": EYES_MODEL}
                result = beta_tools.dispatch(
                    name, args, str(self.workspace), cfg=config
                )
                if not isinstance(result, dict):
                    return self._failure(f"Tool {name} returned invalid output")
                return self._normalize_result(result)
            except Exception as exc:
                return self._failure(f"{type(exc).__name__}: {exc}")
        try:
            result = method(**args)
            if not isinstance(result, dict):
                return self._failure(f"Tool {name} returned invalid output")
            return self._normalize_result({"ok": True, **result})
        except Exception as exc:
            # Tools are an isolation boundary: bad model arguments, optional
            # dependency failures, and malformed files must be reported back to
            # the model, never terminate the autonomous run.
            return self._failure(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def is_plan_safe(name: str, args: dict[str, Any]) -> bool:
        if name in {"shell", "bash"}:
            return True
        if name == "http_request":
            return str(args.get("method", "GET")).upper() in {"GET", "HEAD", "OPTIONS"}
        if name == "describe_image":
            return not bool(args.get("save_to"))
        return name in PLAN_READ_ONLY_TOOLS

    def execute_read_only(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a plan-mode tool while preventing filesystem modification."""
        if not isinstance(args, dict):
            return self._failure("Tool arguments must be a JSON object")
        if not self.is_plan_safe(name, args):
            return self._failure(
                f"Tool '{name}' is disabled in plan mode because it may modify files or state"
            )
        if name in {"shell", "bash"}:
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return self._failure(f"{name} requires a non-empty command")
            try:
                return self.read_only_shell(
                    command,
                    cwd=args.get("cwd") if name == "shell" else None,
                    timeout=args.get("timeout", 0),
                )
            except Exception as exc:
                return self._failure(f"{type(exc).__name__}: {exc}")
        return self.execute(name, args)

    def _failure(self, message: str) -> dict[str, Any]:
        cleaned, truncated = self._trim(str(message))
        result: dict[str, Any] = {"ok": False, "error": cleaned}
        if truncated:
            result["truncated"] = True
        return result

    def _json_safe(
        self,
        value: Any,
        *,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> Any:
        """Convert arbitrary beta-tool output to bounded JSON-safe values."""
        if value is None or isinstance(value, (str, bool, int)):
            if isinstance(value, str):
                return self._trim(value)[0]
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, bytes):
            return self._trim(value.decode("utf-8", errors="replace"))[0]
        if isinstance(value, Path):
            return str(value)
        if depth >= 12:
            return "<maximum nesting depth reached>"
        seen = seen if seen is not None else set()
        identity = id(value)
        if identity in seen:
            return "<recursive value>"
        if isinstance(value, dict):
            seen.add(identity)
            items = list(value.items())
            safe = {
                str(key): self._json_safe(item, depth=depth + 1, seen=seen)
                for key, item in items[:500]
            }
            if len(items) > 500:
                safe["_omitted_keys"] = len(items) - 500
            seen.remove(identity)
            return safe
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(identity)
            items = list(value)
            safe_items = [
                self._json_safe(item, depth=depth + 1, seen=seen)
                for item in items[:500]
            ]
            if len(items) > 500:
                safe_items.append(f"< {len(items) - 500} items omitted >")
            seen.remove(identity)
            return safe_items
        return self._trim(str(value))[0]

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        normalized = self._json_safe(result)
        if not isinstance(normalized, dict):
            return self._failure("Tool returned an invalid result")
        normalized["ok"] = bool(normalized.get("ok", False))
        try:
            encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return self._failure(f"Tool returned non-serializable output: {exc}")
        if len(encoded) <= self.max_output_chars:
            return normalized

        # Preserve the most useful status fields and provide a bounded snapshot
        # instead of allowing one noisy tool to crowd out the conversation.
        compact: dict[str, Any] = {
            "ok": normalized["ok"],
            "truncated": True,
        }
        for key in ("error", "path", "exit_code", "count", "model", "mode"):
            if key in normalized:
                compact[key] = normalized[key]
        primary_key = next(
            (
                key
                for key in ("output", "content", "diff", "description", "body")
                if isinstance(normalized.get(key), str)
            ),
            None,
        )
        if primary_key is not None:
            compact[primary_key] = self._trim_to(
                str(normalized[primary_key]), max(512, self.max_output_chars - 2_000)
            )[0]
        else:
            compact["result_snapshot"] = self._trim_to(
                encoded, max(512, self.max_output_chars - 500)
            )[0]
        return compact

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)

    def _trim(self, text: str) -> tuple[str, bool]:
        return self._trim_to(text, self.max_output_chars)

    @staticmethod
    def _trim_to(text: str, limit: int) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        half = max(1, limit // 2)
        omitted = len(text) - limit
        return f"{text[:half]}\n\n... {omitted} characters omitted ...\n\n{text[-half:]}", True

    def _atomic_write_text(self, target: Path, content: str) -> int:
        """Replace a text file atomically and preserve its existing mode."""
        if target.exists() and not target.is_file():
            raise ToolError(f"Target is not a file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists():
                shutil.copymode(target, temporary)
            os.replace(temporary, target)
            temporary = None
            return len(content.encode("utf-8"))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def shell(self, command: str, cwd: str | None = None, timeout: int = 0) -> dict[str, Any]:
        run_dir = self.resolve(cwd) if cwd else self.workspace
        if not run_dir.is_dir():
            raise ToolError(f"Working directory does not exist: {run_dir}")
        shell = os.environ.get("SHELL") or "/bin/bash"
        try:
            process = subprocess.run(
                command,
                shell=True,
                executable=shell,
                cwd=run_dir,
                text=True,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(timeout) if int(timeout) > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            trimmed, was_trimmed = self._trim(output)
            return {
                "exit_code": 124,
                "output": trimmed,
                "truncated": was_trimmed,
                "timed_out": True,
            }
        output, was_trimmed = self._trim(process.stdout or "")
        return {
            "exit_code": process.returncode,
            "output": output,
            "truncated": was_trimmed,
            "timed_out": False,
            "cwd": str(run_dir),
        }

    def read_only_shell(
        self, command: str, cwd: str | None = None, timeout: int = 0
    ) -> dict[str, Any]:
        """Run a command with the host filesystem mounted read-only."""
        run_dir = self.resolve(cwd) if cwd else self.workspace
        if not run_dir.is_dir():
            raise ToolError(f"Working directory does not exist: {run_dir}")
        bubblewrap = shutil.which("bwrap")
        if bubblewrap is None:
            return self._failure(
                "Read-only shell is unavailable because bubblewrap (bwrap) is not installed"
            )
        shell = os.environ.get("SHELL") or "/bin/bash"
        environment = os.environ.copy()
        environment.update(
            {
                "TMPDIR": "/tmp",
                "XDG_CACHE_HOME": "/tmp/bob-der-cache",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            process = subprocess.run(
                [
                    bubblewrap,
                    "--die-with-parent",
                    "--ro-bind",
                    "/",
                    "/",
                    "--dev-bind",
                    "/dev",
                    "/dev",
                    "--proc",
                    "/proc",
                    "--tmpfs",
                    "/tmp",
                    "--ro-bind",
                    str(run_dir),
                    str(run_dir),
                    "--chdir",
                    str(run_dir),
                    shell,
                    "-c",
                    command,
                ],
                cwd=run_dir,
                env=environment,
                text=True,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(timeout) if int(timeout) > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            output, truncated = self._trim(output)
            return self._normalize_result(
                {
                    "ok": True,
                    "exit_code": 124,
                    "output": output,
                    "truncated": truncated,
                    "timed_out": True,
                    "read_only": True,
                }
            )
        output, truncated = self._trim(process.stdout or "")
        return self._normalize_result(
            {
                "ok": True,
                "exit_code": process.returncode,
                "output": output,
                "truncated": truncated,
                "timed_out": False,
                "read_only": True,
                "cwd": str(run_dir),
            }
        )

    def read_file(
        self, path: str, start_line: int = 1, end_line: int = 400
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"File does not exist: {target}")
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        selected_lines: list[str] = []
        total_lines = 0
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for total_lines, line in enumerate(handle, 1):
                if start <= total_lines <= end:
                    selected_lines.append(line)
        selected = "".join(selected_lines)
        selected, was_trimmed = self._trim(selected)
        return {
            "path": str(target),
            "content": selected,
            "start_line": start,
            "end_line": min(end, total_lines),
            "total_lines": total_lines,
            "truncated": was_trimmed or end < total_lines,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self.resolve(path)
        bytes_written = self._atomic_write_text(target, content)
        return {"path": str(target), "bytes_written": bytes_written}

    def replace_in_file(
        self, path: str, old: str, new: str, count: int = 1
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"File does not exist: {target}")
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old)
        requested = int(count)
        if not old:
            raise ToolError("old text cannot be empty")
        if occurrences == 0:
            raise ToolError("old text was not found")
        if requested == 1 and occurrences != 1:
            raise ToolError(f"old text appears {occurrences} times; provide a unique match")
        if requested <= 0:
            replaced = content.replace(old, new)
            replacements = occurrences
        else:
            replaced = content.replace(old, new, requested)
            replacements = min(requested, occurrences)
        self._atomic_write_text(target, replaced)
        return {"path": str(target), "replacements": replacements}

    def list_files(self, path: str = ".", pattern: str = "*") -> dict[str, Any]:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"Directory does not exist: {target}")
        entries: list[str] = []
        for root, dirs, files in os.walk(target):
            dirs[:] = sorted(d for d in dirs if d not in {".git", "node_modules", ".venv"})
            for filename in sorted(files):
                full = Path(root) / filename
                relative = str(full.relative_to(target))
                if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(filename, pattern):
                    entries.append(relative)
                if len(entries) >= 2000:
                    return {"path": str(target), "files": entries, "truncated": True}
        return {"path": str(target), "files": entries, "truncated": False}

    def grep(
        self,
        pattern: str,
        path: str = ".",
        case_insensitive: bool = False,
        file_pattern: str | None = None,
        max_results: int = 200,
    ) -> dict[str, Any]:
        """Search file contents, preferring ripgrep when it is installed."""
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        limit = max(1, min(int(max_results), 2000))
        rg = shutil.which("rg")
        if rg:
            command = [rg, "--line-number", "--no-heading", "--color=never"]
            if case_insensitive:
                command.append("--ignore-case")
            if file_pattern:
                command.extend(["--glob", file_pattern])
            command.extend([pattern, str(target)])
            result = subprocess.run(
                command,
                cwd=self.workspace,
                text=True,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode not in {0, 1}:
                raise ToolError(result.stderr.strip() or "Search failed")
            lines = result.stdout.splitlines()
        else:
            flags = re.IGNORECASE if case_insensitive else 0
            expression = re.compile(pattern, flags)
            candidates = [target] if target.is_file() else target.rglob("*")
            lines = []
            for candidate in candidates:
                if not candidate.is_file() or any(
                    part in {".git", "node_modules", ".venv", "__pycache__"}
                    for part in candidate.parts
                ):
                    continue
                if file_pattern and not fnmatch.fnmatch(candidate.name, file_pattern):
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line_number, line in enumerate(content.splitlines(), 1):
                    if expression.search(line):
                        lines.append(f"{candidate}:{line_number}:{line}")
        total = len(lines)
        return {
            "path": str(target),
            "matches": lines[:limit],
            "count": total,
            "truncated": total > limit,
        }

    def find_files(
        self, pattern: str, path: str = ".", max_results: int = 500
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        limit = max(1, min(int(max_results), 5000))
        candidates = [target] if target.is_file() else target.rglob("*")
        matches: list[str] = []
        for candidate in candidates:
            if not candidate.is_file() or any(
                part in {".git", "node_modules", ".venv", "__pycache__", "dist"}
                for part in candidate.parts
            ):
                continue
            try:
                relative = candidate.relative_to(target if target.is_dir() else target.parent)
            except ValueError:
                relative = candidate
            if fnmatch.fnmatch(candidate.name, pattern) or fnmatch.fnmatch(
                str(relative), pattern
            ):
                matches.append(str(relative))
        matches.sort()
        total = len(matches)
        return {
            "path": str(target),
            "matches": matches[:limit],
            "count": total,
            "truncated": total > limit,
        }

    def git_diff(
        self,
        ref_a: str = "",
        ref_b: str = "",
        path: str = "",
        staged: bool = False,
    ) -> dict[str, Any]:
        command = ["git", "diff"]
        if staged:
            command.append("--staged")
        if ref_a:
            command.append(ref_a)
        if ref_b:
            command.append(ref_b)
        if path:
            command.extend(["--", str(self.resolve(path))])
        result = subprocess.run(
            command,
            cwd=self.workspace,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise ToolError(result.stderr.strip() or "git diff failed")
        diff, truncated = self._trim(result.stdout)
        return {"diff": diff, "truncated": truncated}

    def test_run(
        self, path: str = "", runner: str = "", args: str = ""
    ) -> dict[str, Any]:
        target = self.resolve(path) if path else self.workspace
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        run_dir = target if target.is_dir() else target.parent
        extra = shlex.split(args)
        selected = runner.strip().lower()
        if not selected:
            if (run_dir / "package.json").exists():
                selected = "npm"
            elif (run_dir / "Cargo.toml").exists():
                selected = "cargo"
            elif (run_dir / "go.mod").exists():
                selected = "go"
            else:
                selected = "unittest"
        if selected == "pytest":
            command = [sys.executable, "-m", "pytest", "-q"]
            if target.is_file():
                command.append(str(target))
        elif selected == "unittest":
            command = [sys.executable, "-m", "unittest", "discover", "-v"]
            if (run_dir / "tests").is_dir():
                command.extend(["-s", "tests"])
        elif selected in {"npm", "jest"}:
            command = ["npm", "test", "--"]
        elif selected == "cargo":
            command = ["cargo", "test"]
        elif selected == "go":
            command = ["go", "test", "./..."]
        else:
            raise ToolError(f"Unknown test runner: {runner}")
        command.extend(extra)
        result = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output, truncated = self._trim(result.stdout)
        return {
            "runner": selected,
            "command": shlex.join(command),
            "exit_code": result.returncode,
            "output": output,
            "truncated": truncated,
        }

    def todo_scan(
        self, path: str = ".", tags: list[str] | None = None, max_results: int = 200
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        selected_tags = tags or ["TODO", "FIXME", "HACK", "XXX"]
        expression = re.compile(
            r"\b(" + "|".join(re.escape(tag) for tag in selected_tags) + r")\b[:\s]*(.{0,160})",
            re.IGNORECASE,
        )
        candidates = [target] if target.is_file() else target.rglob("*")
        matches: list[dict[str, object]] = []
        for candidate in candidates:
            if not candidate.is_file() or any(
                part in {".git", "node_modules", ".venv", "__pycache__"}
                for part in candidate.parts
            ):
                continue
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                match = expression.search(line)
                if match:
                    matches.append(
                        {
                            "file": str(candidate),
                            "line": line_number,
                            "tag": match.group(1).upper(),
                            "text": match.group(2).strip(),
                        }
                    )
        limit = max(1, min(int(max_results), 2000))
        return {
            "matches": matches[:limit],
            "count": len(matches),
            "truncated": len(matches) > limit,
        }

    def size_report(
        self, path: str = ".", extension: str = "", max_results: int = 50
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        candidates = [target] if target.is_file() else target.rglob("*")
        rows: list[dict[str, object]] = []
        total_lines = 0
        for candidate in candidates:
            if not candidate.is_file() or any(
                part in {".git", "node_modules", ".venv", "__pycache__"}
                for part in candidate.parts
            ):
                continue
            if extension and candidate.suffix != extension.lstrip("*"):
                continue
            try:
                line_count = len(
                    candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                continue
            total_lines += line_count
            rows.append({"file": str(candidate), "lines": line_count})
        rows.sort(key=lambda row: int(row["lines"]), reverse=True)
        limit = max(1, min(int(max_results), 500))
        return {
            "files": rows[:limit],
            "file_count": len(rows),
            "total_lines": total_lines,
            "truncated": len(rows) > limit,
        }
