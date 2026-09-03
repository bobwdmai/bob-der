from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ollama import OllamaError, OllamaRunner, OllamaSlowError, parse_action
from .prompts import (
    DEFAULT_SUBAGENT_SYSTEM_PROMPT,
    DEEP_REASONING_PROMPT,
    PLAN_MODE_PROMPT,
    build_prompt,
    system_prompt,
)
from .tools import ToolExecutor


EventHandler = Callable[[str, dict[str, object]], None]


@dataclass
class Agent:
    runner: OllamaRunner
    workspace: Path
    max_steps: int = 50
    on_event: EventHandler | None = None
    subagent_runner: OllamaRunner | None = None
    subagent_max_steps: int = 25
    is_subagent: bool = False
    subagent_system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT
    custom_system_prompt: str = ""
    context_char_limit: int = 60_000
    transcript: list[dict[str, object]] = field(default_factory=list)
    configured_model_timeout: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.configured_model_timeout = self.runner.timeout
        self.workspace = self.workspace.expanduser().resolve()
        self.tools = ToolExecutor(self.workspace)
        if self.on_event is not None:
            self.runner.on_chunk = lambda chunk: self.emit(
                "model_chunk", {"chunk": chunk}
            )
        self.refresh_system()

    def refresh_system(self) -> None:
        self.system = system_prompt(
            str(self.workspace),
            self.runner.model,
            subagent_model=self.subagent_runner.model if self.subagent_runner else None,
            is_subagent=self.is_subagent,
            custom_instructions=self.custom_system_prompt,
        )

    def emit(self, event: str, data: dict[str, object]) -> None:
        if self.on_event:
            self.on_event(event, data)

    def clear(self) -> None:
        self.transcript.clear()

    def set_workspace(self, workspace: Path) -> None:
        resolved = workspace.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Working directory does not exist: {resolved}")
        self.workspace = resolved
        self.tools = ToolExecutor(resolved)
        self.refresh_system()

    def set_subagent_system_prompt(self, prompt: str) -> None:
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("Subagent system prompt cannot be empty")
        self.subagent_system_prompt = cleaned

    def undo_last_turn(self) -> bool:
        for index in range(len(self.transcript) - 1, -1, -1):
            if self.transcript[index].get("role") == "user":
                del self.transcript[index:]
                return True
        return False

    def prompt_transcript(self) -> list[dict[str, object]]:
        """Keep recent context bounded so tool-heavy sessions stay responsive."""
        selected: list[dict[str, object]] = []
        used = 0
        omitted = 0
        for item in reversed(self.transcript):
            size = len(json.dumps(item, ensure_ascii=False, default=str))
            if selected and used + size > self.context_char_limit:
                omitted = len(self.transcript) - len(selected)
                break
            selected.append(item)
            used += size
        selected.reverse()
        if omitted:
            selected.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        f"{omitted} older transcript entries were omitted from this "
                        "model call for responsiveness. The full session remains saved."
                    ),
                },
            )
        return selected

    def run(
        self,
        request: str,
        *,
        plan_mode: bool = False,
        deep_mode: bool = False,
    ) -> str:
        if plan_mode and deep_mode:
            raise ValueError("Plan mode and deep reasoning mode are mutually exclusive")
        self.transcript.append({"role": "user", "content": request})
        step = 0
        malformed_attempts = 0
        active_system = self.system
        if plan_mode:
            active_system += PLAN_MODE_PROMPT
        elif deep_mode:
            active_system += DEEP_REASONING_PROMPT
        self.runner.thinking_level = "high" if deep_mode else "false"
        self.runner.timeout = 0 if deep_mode else self.configured_model_timeout
        active_runner = self.runner
        deep_stages = ("reason", "plan", "revise")
        deep_stage_index = 0

        while deep_mode or self.max_steps <= 0 or step < self.max_steps:
            step += 1
            self.emit("thinking", {"step": step})
            self.emit(
                "model_start",
                {
                    "step": step,
                    "model": active_runner.model,
                    "mode": "deep" if deep_mode else "plan" if plan_mode else "execute",
                },
            )
            prompt = build_prompt(active_system, self.prompt_transcript())
            try:
                raw = active_runner.generate(prompt)
            except OllamaSlowError as exc:
                if active_runner is not self.runner or self.subagent_runner is None:
                    raise
                active_runner = self.subagent_runner
                active_runner.thinking_level = "high" if deep_mode else "false"
                active_runner.timeout = 0 if deep_mode else self.configured_model_timeout
                active_runner.on_chunk = self.runner.on_chunk
                self.emit(
                    "fallback",
                    {
                        "from_model": self.runner.model,
                        "to_model": active_runner.model,
                        "reason": str(exc),
                    },
                )
                fallback_system = (
                    active_system
                    + "\n\nSUBAGENT TAKEOVER INSTRUCTIONS:\n"
                    + self.subagent_system_prompt
                )
                raw = active_runner.generate(
                    build_prompt(fallback_system, self.prompt_transcript())
                )
            try:
                action = parse_action(raw)
            except OllamaError as exc:
                malformed_attempts += 1
                self.transcript.append(
                    {
                        "role": "assistant_invalid",
                        "content": raw[:2000],
                    }
                )
                self.transcript.append(
                    {
                        "role": "tool",
                        "content": {
                            "ok": False,
                            "error": str(exc),
                            "instruction": "Return exactly one valid JSON tool action.",
                        },
                    }
                )
                if malformed_attempts >= 3:
                    raise
                continue

            malformed_attempts = 0
            tool = str(action["tool"])
            args = action["args"]
            assert isinstance(args, dict)
            thought = str(action.get("thought", ""))
            self.transcript.append({"role": "assistant", "content": action})

            if deep_mode and tool == "deep_phase":
                stage = str(args.get("stage", "")).lower()
                content = str(args.get("content", "")).strip()
                expected = (
                    deep_stages[deep_stage_index]
                    if deep_stage_index < len(deep_stages)
                    else None
                )
                if expected is None:
                    result = {
                        "ok": False,
                        "error": "Deep analysis phases are complete; execute the revised plan.",
                    }
                elif stage != expected or not content:
                    result = {
                        "ok": False,
                        "error": f"Expected non-empty deep phase '{expected}', got '{stage}'.",
                    }
                else:
                    deep_stage_index += 1
                    result = {"ok": True, "stage": stage, "content": content}
                    self.emit(
                        "deep_phase",
                        {"stage": stage, "content": content, "step": step},
                    )
                self.transcript.append(
                    {"role": "tool", "content": {"tool": tool, "result": result}}
                )
                continue

            if deep_mode and deep_stage_index < len(deep_stages):
                expected = deep_stages[deep_stage_index]
                result = {
                    "ok": False,
                    "error": (
                        f"Deep workflow is incomplete. Submit the '{expected}' "
                        "phase with deep_phase before using tools or final."
                    ),
                }
                self.transcript.append(
                    {"role": "tool", "content": {"tool": tool, "result": result}}
                )
                self.emit("tool_end", {"tool": tool, "result": result, "step": step})
                continue

            if tool == "final":
                message = str(args.get("message", "Task complete."))
                self.emit(
                    "final",
                    {"message": message, "thought": thought, "step": step},
                )
                return message

            if plan_mode and not self.tools.is_plan_safe(tool, args):
                result = {
                    "ok": False,
                    "error": (
                        f"Tool '{tool}' is disabled in plan mode because it may "
                        "modify files or state. Use read-only inspection tools, "
                        "then return the proposed plan with final."
                    ),
                }
                self.transcript.append(
                    {"role": "tool", "content": {"tool": tool, "result": result}}
                )
                self.emit("tool_end", {"tool": tool, "result": result, "step": step})
                continue

            self.emit("tool_start", {"tool": tool, "args": args, "thought": thought, "step": step})
            if tool == "subagent":
                result = self.run_subagent(args)
            elif plan_mode:
                result = self.tools.execute_read_only(tool, args)
            else:
                result = self.tools.execute(tool, args)
            self.transcript.append(
                {"role": "tool", "content": {"tool": tool, "result": result}}
            )
            self.emit("tool_end", {"tool": tool, "result": result, "step": step})

        message = f"Stopped after reaching the {self.max_steps}-step limit. Use --max-steps 0 for no limit."
        self.transcript.append({"role": "system", "content": message})
        raise RuntimeError(message)

    def run_subagent(self, args: dict[str, object]) -> dict[str, object]:
        if self.subagent_runner is None or self.is_subagent:
            return {"ok": False, "error": "No subagent is configured"}
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return {"ok": False, "error": "subagent requires a non-empty task"}
        child = Agent(
            runner=self.subagent_runner,
            workspace=self.workspace,
            max_steps=self.subagent_max_steps,
            on_event=None,
            subagent_runner=None,
            is_subagent=True,
            custom_system_prompt=self.subagent_system_prompt,
        )
        try:
            message = child.run(task)
            return {
                "ok": True,
                "model": self.subagent_runner.model,
                "message": message,
                "steps": sum(1 for item in child.transcript if item["role"] == "assistant"),
            }
        except (OllamaError, RuntimeError) as exc:
            return {"ok": False, "model": self.subagent_runner.model, "error": str(exc)}
