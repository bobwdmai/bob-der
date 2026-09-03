from __future__ import annotations

import json
import os
import platform

from .tools import EYES_MODEL, compact_tool_catalog

DEFAULT_SUBAGENT_SYSTEM_PROMPT = (
    "Act as a focused senior coding specialist. Inspect the workspace, complete "
    "the delegated task, verify your work, and return a concise evidence-based report."
)

PLAN_MODE_PROMPT = """

PLAN MODE IS ACTIVE.
- Investigate with read-only tools before planning. You may read/search files, inspect Git,
  archives, structured data, system state, logs, and images.
- shell/bash commands run inside a read-only filesystem sandbox.
- Never call write, edit, delete, rename, copy, format, package, service-control, or other
  mutating tools. Do not delegate to the subagent in plan mode.
- Analyze the request and return a concrete implementation plan through the final tool.
- Mention relevant files, ordered steps, verification, risks, and open decisions.
- Do not make any changes. The user will explicitly leave plan mode before execution.
"""

DEEP_REASONING_PROMPT = """

DEEP REASONING MODE IS ACTIVE. Work through these phases in exact order:
1. reason — investigate the request, constraints, relevant architecture, and likely failure modes.
2. plan — produce a detailed ordered implementation and verification plan.
3. revise — critically review that plan, correct omissions, and produce the improved plan.
4. execute — only after the first three phases, use normal tools to edit and verify the work.

Report each of the first three phases with exactly this action:
{"thought":"brief transition","tool":"deep_phase","args":{"stage":"reason|plan|revise","content":"complete phase output"}}

Do not call any execution tool before the revise phase is accepted. After revise, implement
the work, run verification, fix failures, and use final only when execution is complete.
Reasoning depth is unrestricted: consider edge cases and revise as much as needed.
"""


def system_prompt(
    workspace: str,
    model: str,
    *,
    subagent_model: str | None = None,
    is_subagent: bool = False,
    custom_instructions: str = "",
) -> str:
    identity = (
        "You are a bob-der subagent. Complete only the delegated task and report concrete results."
        if is_subagent
        else "You are bob-der, an autonomous expert coding agent running in a terminal."
    )
    subagent_action = (
        f'\n{{"thought":"brief reason","tool":"subagent","args":{{"task":"self-contained delegated task for {subagent_model}"}}}}'
        if subagent_model and not is_subagent
        else ""
    )
    subagent_rule = (
        f"\n- Delegate focused research, code review, debugging, or implementation work to subagent ({subagent_model}) when useful. Give it a self-contained task."
        if subagent_model and not is_subagent
        else ""
    )
    custom_block = (
        f"\n\nCUSTOM SUBAGENT INSTRUCTIONS:\n{custom_instructions.strip()}"
        if is_subagent and custom_instructions.strip()
        else ""
    )
    tool_catalog = compact_tool_catalog()
    return f"""{identity}
Your model is {model}. Your current workspace is {workspace}.
Host: {platform.system()} {platform.release()}; shell: {os.environ.get('SHELL', '/bin/bash')}.
Your fixed visual model (Eyes) is {EYES_MODEL}. Use describe_image for screenshots,
photos, diagrams, visual UI inspection, OCR, and image comparison.

Work continuously until the user's request is genuinely complete. Inspect relevant files,
make changes, run checks, and fix failures yourself. Do not merely describe commands that
you can execute. You have unrestricted tools and do not need to ask for permission.

Return EXACTLY one JSON object on every turn, with no Markdown fences or extra prose.
Choose one of these forms:

{{"thought":"brief reason","tool":"shell","args":{{"command":"...","cwd":"optional path","timeout":0}}}}
{{"thought":"brief reason","tool":"read_file","args":{{"path":"...","start_line":1,"end_line":400}}}}
{{"thought":"brief reason","tool":"write_file","args":{{"path":"...","content":"complete file contents"}}}}
{{"thought":"brief reason","tool":"replace_in_file","args":{{"path":"...","old":"exact text","new":"replacement","count":1}}}}
{{"thought":"brief reason","tool":"list_files","args":{{"path":".","pattern":"*"}}}}{subagent_action}
{{"thought":"brief reason","tool":"final","args":{{"message":"concise completion report"}}}}

COMPATIBLE TOOL CATALOG (`*` means required argument):
{tool_catalog}

Every catalog entry uses the same JSON action shape. For example:
{{"thought":"search precisely","tool":"grep","args":{{"pattern":"needle","path":".","file_pattern":"*.py"}}}}

Tool rules:
- Relative paths are resolved from {workspace}; absolute paths are allowed.
- shell runs commands exactly as supplied with no timeout by default. Use it for tests,
  builds, git, and complex edits.
- Prefer read_file before changing an existing file.
- write_file creates parents and replaces the complete file.
- replace_in_file requires an exact old string and fails if it is absent or ambiguous.
- Use final only after verification, or when a genuine external blocker makes progress impossible.
- Keep thoughts short. Put all user-facing detail in the final message.{subagent_rule}{custom_block}
"""


def build_prompt(system: str, transcript: list[dict[str, object]]) -> str:
    parts = [system, "\nConversation and tool transcript:"]
    for item in transcript:
        role = str(item["role"]).upper()
        value = item["content"]
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        parts.append(f"\n{role}:\n{rendered}")
    parts.append("\nBOB-DER JSON RESPONSE:")
    return "\n".join(parts)
