from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Markdown,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from .agent import Agent
from .cli import partial_final_message, partial_thinking
from .ollama import OllamaError
from .prompts import DEFAULT_SUBAGENT_SYSTEM_PROMPT
from .sessions import SessionStore
from .tools import EYES_MODEL

COMMANDS = (
    ("/help", "Show all commands"),
    ("/copy", "Copy the latest answer"),
    ("/update", "Fetch a verified bob-der update from GitHub"),
    ("/plan", "Toggle plan mode or plan a task"),
    ("/deep", "Reason, plan, revise, then execute without a step limit"),
    ("/subagent-prompt", "Show, set, or reset subagent instructions"),
    ("/model", "Show the fixed main and subagent models"),
    ("/cwd", "Show or change the workspace"),
    ("/undo", "Remove the latest turn from context"),
    ("/sessions", "List saved sessions"),
    ("/load", "Restore a saved session"),
    ("/new", "Start a new session"),
    ("/clear", "Clear conversation context"),
    ("/exit", "Quit bob-der"),
)

SPINNER = "⣾⣷⣯⣟⡿⢿⣻⣽"
THINKING_VERBS = ("reasoning", "mapping context", "checking edges", "connecting details")
PLAN_VERBS = ("planning", "sequencing steps", "checking risks", "designing verification")
DEEP_VERBS = ("reasoning deeply", "challenging assumptions", "revising strategy", "checking every edge")
CODE_FRAMES = ("</>", "{  }", "[ ]", "( )", "=>", "::")
CODE_VERBS = ("coding", "patching", "building", "verifying", "refactoring")
SUBAGENT_VERBS = ("delegating", "consulting specialist", "reviewing evidence")


class PromptTextArea(TextArea):
    """A multiline prompt where Enter edits and Ctrl+Enter submits."""

    BINDINGS = [
        Binding("ctrl+enter,shift+enter,ctrl+s", "submit", "Send", priority=True),
        Binding(
            "enter,alt+enter",
            "insert_prompt_newline",
            "New line",
            show=False,
            priority=True,
        ),
        *TextArea.BINDINGS,
    ]

    class Submitted(Message):
        def __init__(self, text_area: "PromptTextArea") -> None:
            self.text_area = text_area
            self.value = text_area.text
            super().__init__()

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    def action_insert_prompt_newline(self) -> None:
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)


class CopyBox(Vertical):
    """A response panel that supports selection and one-click whole-text copy."""

    class CopyRequested(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(
        self,
        text: str = "",
        *,
        title: str = "Answer",
        classes: str | None = None,
    ) -> None:
        super().__init__(classes=classes)
        self.copy_text = text
        self.box_title = title

    def compose(self) -> ComposeResult:
        with Horizontal(classes="copy-header"):
            yield Static(self.box_title, classes="copy-title")
            yield Button("Copy", classes="copy-button", compact=True)
        yield Static(self.copy_text, markup=False, classes="copy-area copy-content")

    def on_mount(self) -> None:
        self._fit_text_area()

    def set_text(self, text: str) -> None:
        self.copy_text = text
        areas = list(self.query(".copy-content"))
        if not areas:
            # Streaming may update immediately after the box is mounted, before
            # Textual has composed its children. compose() will use copy_text.
            return
        area = areas[0]
        if str(area.render()) != text:
            area.update(text)
            self._fit_text_area()

    def _fit_text_area(self) -> None:
        # Keep short replies compact while still making long output easy to scroll.
        line_count = max(1, self.copy_text.count("\n") + 1)
        areas = list(self.query(".copy-content"))
        if areas:
            areas[0].styles.height = min(18, max(3, line_count + 1))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CopyRequested(self.copy_text))


class BobDerApp(App[int]):
    TITLE = "bob-der"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_conversation", "Clear"),
    ]
    CSS = """
    Screen {
        background: #17151c;
        color: #e8e4ed;
    }

    #banner {
        height: auto;
        padding: 1 2;
        background: #241d2b;
        border-bottom: solid #a875d1;
        color: #d9b8f0;
    }

    #conversation {
        height: 1fr;
        padding: 1 2;
        scrollbar-color: #7c4d9e;
        scrollbar-background: #241d2b;
    }

    #prompt {
        height: 3;
        max-height: 10;
        margin: 0 2 1 2;
        border: tall #a875d1;
        background: #211c26;
    }

    #prompt:focus {
        border: tall #d9b8f0;
    }

    #prompt.plan-mode {
        border: tall #62a7e8;
    }

    #activity {
        display: none;
        height: 1;
        margin: 0 3;
        color: #b785db;
    }

    #slash-suggestions {
        display: none;
        height: auto;
        max-height: 9;
        margin: 0 2;
        border: round #6d557d;
        background: #211c26;
    }

    .user-message {
        margin: 1 0 0 0;
        color: #ffffff;
        text-style: bold;
    }

    .queued-message {
        margin: 1 0 0 0;
        color: #928a99;
    }

    .thinking-panel {
        margin: 0;
        padding: 0;
        background: #1d1922;
        color: #bcb2c5;
    }

    .thinking-body {
        padding: 0 1 1 2;
        color: #a9a0b0;
    }

    .tool-call {
        margin: 0 0 0 1;
        color: #d49cff;
    }

    .tool-result {
        margin: 0 0 0 3;
        color: #928a99;
    }

    .answer {
        margin: 1 0;
        color: #f3edf7;
    }

    CopyBox {
        height: auto;
        margin: 1 0;
        border: round #6d557d;
        background: #1d1922;
    }

    CopyBox:focus-within {
        border: round #a875d1;
    }

    .copy-header {
        height: 3;
        padding: 0 1;
        background: #241d2b;
        align-vertical: middle;
    }

    .copy-title {
        width: 1fr;
        color: #d9b8f0;
        text-style: bold;
    }

    .copy-button {
        width: 10;
        min-width: 10;
        height: 1;
        border: none;
        background: #6d557d;
        color: #ffffff;
    }

    .copy-button:hover, .copy-button:focus {
        background: #a875d1;
    }

    .copy-area {
        padding: 0 1;
        border: none;
        background: #1d1922;
        color: #f3edf7;
        scrollbar-color: #7c4d9e;
        scrollbar-background: #241d2b;
    }

    .error {
        margin: 1 0;
        color: #ff7b87;
    }

    Footer {
        background: #241d2b;
    }
    """

    def __init__(
        self, agent: Agent, session_store: SessionStore | None = None
    ) -> None:
        super().__init__()
        self.agent = agent
        self.agent.on_event = self.agent_event
        self.current_raw = ""
        self.current_thinking: Static | None = None
        self.current_received_thinking = False
        self.current_panel: Collapsible | None = None
        self.current_answer: CopyBox | None = None
        self.current_answer_text = ""
        self.last_copy_text = ""
        self.busy = False
        self.pending: deque[tuple[str, Static, bool, bool]] = deque()
        self.plan_mode = False
        self.deep_mode = False
        self.active_plan_mode = False
        self.active_deep_mode = False
        self.activity_kind = "idle"
        self.activity_label = ""
        self.activity_frame = 0
        self.sessions = session_store or SessionStore()
        self.session_id = self.sessions.new_id()

    def compose(self) -> ComposeResult:
        yield Static(self.banner_text(), id="banner")
        yield VerticalScroll(id="conversation")
        yield OptionList(id="slash-suggestions", compact=True)
        yield Static("", id="activity")
        yield PromptTextArea(
            placeholder="Ask bob-der something…  Ctrl+Enter to send",
            id="prompt",
            soft_wrap=True,
            show_line_numbers=False,
            compact=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", PromptTextArea).focus()
        self.set_interval(0.12, self.animate_activity)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt":
            return
        suggestions = self.query_one("#slash-suggestions", OptionList)
        value = event.text_area.text
        line_count = max(1, value.count("\n") + 1)
        event.text_area.styles.height = min(10, max(3, line_count + 2))
        if not value.startswith("/") or " " in value or "\n" in value:
            suggestions.styles.display = "none"
            return
        matches = [
            Option(f"{command:<20} {description}", id=command)
            for command, description in COMMANDS
            if command.startswith(value)
        ]
        suggestions.set_options(matches)
        suggestions.highlighted = 0 if matches else None
        suggestions.styles.display = "block" if matches else "none"

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_list.id != "slash-suggestions" or not event.option.id:
            return
        prompt = self.query_one("#prompt", PromptTextArea)
        value = f"{event.option.id} "
        prompt.load_text(value)
        prompt.move_cursor((0, len(value)))
        event.option_list.styles.display = "none"
        prompt.focus()

    def on_copy_box_copy_requested(self, event: CopyBox.CopyRequested) -> None:
        self.copy_to_clipboard(event.text)
        self.notify("Copied box to clipboard", timeout=1.5)

    async def on_prompt_text_area_submitted(
        self, event: PromptTextArea.Submitted
    ) -> None:
        if not event.value.strip():
            return
        # Preserve pasted indentation and internal newlines. Only remove line
        # breaks accidentally left at the outer edge of the prompt.
        request = event.value.strip("\r\n")
        event.text_area.load_text("")
        self.query_one("#slash-suggestions", OptionList).styles.display = "none"
        if request in {"/exit", "/quit"}:
            self.exit(0)
            return

        if request.startswith("/") and self.busy:
            await self.add(
                Static(
                    "A task is running. Type a follow-up to queue it, or wait before using commands.",
                    classes="error",
                )
            )
            return

        if request.startswith("/") and await self.handle_command(request):
            return

        if self.busy:
            queued = Static(f"↳ queued: {request}", classes="queued-message")
            await self.add(queued)
            self.pending.append((request, queued, self.plan_mode, self.deep_mode))
            self.update_prompt_hint()
            return

        await self.add(Static(f"❯ {request}", classes="user-message"))
        self.start_task(request, self.plan_mode, self.deep_mode)

    async def handle_command(self, request: str) -> bool:
        command, _, argument = request.partition(" ")
        argument = argument.strip()
        conversation = self.query_one("#conversation", VerticalScroll)

        if command == "/clear":
            self.agent.clear()
            self.pending.clear()
            await conversation.remove_children()
            self.session_id = self.sessions.new_id()
            return True

        if command == "/copy":
            if not self.last_copy_text:
                await self.add(Static("Nothing to copy yet.", classes="error"))
            else:
                self.copy_to_clipboard(self.last_copy_text)
                self.notify("Copied latest answer to clipboard", timeout=1.5)
            return True

        if command == "/update":
            await self.add(Static("Checking GitHub for updates…", classes="tool-result"))
            self.fetch_update_worker()
            return True

        if command == "/new":
            self.save_session()
            self.agent.clear()
            self.pending.clear()
            self.session_id = self.sessions.new_id()
            await conversation.remove_children()
            await self.add(
                Static(f"New session: {self.session_id}", classes="tool-result")
            )
            return True

        if command == "/model":
            if argument:
                await self.add(
                    Static(
                        "Models are fixed: Nemotron main, gpt-oss subagent, and Gemma Eyes.",
                        classes="error",
                    )
                )
            subagent = (
                self.agent.subagent_runner.model
                if self.agent.subagent_runner is not None
                else "disabled"
            )
            await self.add(
                Static(
                    f"Primary: {self.agent.runner.model}\n"
                    f"Subagent: {subagent}\nEyes: {EYES_MODEL}",
                    classes="tool-result",
                )
            )
            return True

        if command == "/plan":
            lowered = argument.lower()
            if lowered in {"off", "false", "0"}:
                self.plan_mode = False
            elif lowered in {"on", "true", "1", ""}:
                self.plan_mode = not self.plan_mode if not argument else True
                if self.plan_mode:
                    self.deep_mode = False
            else:
                self.plan_mode = True
                self.deep_mode = False
                self.update_mode_display()
                await self.add(
                    Static(f"❯ {argument}", classes="user-message")
                )
                self.start_task(argument, True, False)
                return True
            self.update_mode_display()
            state = "enabled" if self.plan_mode else "disabled"
            await self.add(
                Static(f"Plan mode {state}.", classes="tool-result")
            )
            return True

        if command == "/deep":
            lowered = argument.lower()
            if lowered in {"off", "false", "0"}:
                self.deep_mode = False
            elif lowered in {"on", "true", "1", ""}:
                self.deep_mode = not self.deep_mode if not argument else True
                if self.deep_mode:
                    self.plan_mode = False
            else:
                self.deep_mode = True
                self.plan_mode = False
                self.update_mode_display()
                await self.add(Static(f"❯ {argument}", classes="user-message"))
                self.start_task(argument, False, True)
                return True
            self.update_mode_display()
            state = "enabled" if self.deep_mode else "disabled"
            await self.add(
                Static(
                    f"Deep reasoning mode {state}.",
                    classes="tool-result",
                )
            )
            return True

        if command == "/subagent-prompt":
            if not argument:
                await self.add(
                    Static(
                        "Subagent system prompt:\n"
                        + self.agent.subagent_system_prompt,
                        classes="tool-result",
                    )
                )
            else:
                try:
                    if argument.lower() == "reset":
                        prompt = DEFAULT_SUBAGENT_SYSTEM_PROMPT
                    elif argument.startswith("@"):
                        prompt_path = Path(argument[1:]).expanduser()
                        if not prompt_path.is_absolute():
                            prompt_path = self.agent.workspace / prompt_path
                        prompt = prompt_path.read_text(encoding="utf-8")
                    else:
                        prompt = argument
                    self.agent.set_subagent_system_prompt(prompt)
                except (OSError, ValueError) as exc:
                    await self.add(Static(str(exc), classes="error"))
                else:
                    self.save_session()
                    await self.add(
                        Static("Subagent system prompt updated.", classes="tool-result")
                    )
            return True

        if command == "/cwd":
            if not argument:
                await self.add(
                    Static(str(self.agent.workspace), classes="tool-result")
                )
                return True
            candidate = Path(argument).expanduser()
            if not candidate.is_absolute():
                candidate = self.agent.workspace / candidate
            try:
                self.agent.set_workspace(candidate)
            except ValueError as exc:
                await self.add(Static(str(exc), classes="error"))
            else:
                self.save_session()
                self.query_one("#banner", Static).update(self.banner_text())
                await self.add(
                    Static(
                        f"Workspace: {self.agent.workspace}", classes="tool-result"
                    )
                )
            return True

        if command == "/undo":
            if self.agent.undo_last_turn():
                self.save_session()
                await self.add(
                    Static("Removed the latest user turn.", classes="tool-result")
                )
            else:
                await self.add(Static("Nothing to undo.", classes="tool-result"))
            return True

        if command == "/sessions":
            sessions = self.sessions.list()
            if not sessions:
                text = "No saved sessions."
            else:
                text = "\n".join(
                    f"{item['id']}  {item['preview']}\n  {item['workspace']}"
                    for item in sessions
                )
            await self.add(Static(text, classes="tool-result"))
            return True

        if command == "/load":
            if not argument:
                await self.add(
                    Static("Usage: /load <session-id>", classes="error")
                )
                return True
            try:
                data = self.sessions.load(argument)
                workspace = Path(str(data.get("workspace", self.agent.workspace)))
                self.agent.set_workspace(workspace)
                subagent_prompt = str(data.get("subagent_prompt", "")).strip()
                if subagent_prompt:
                    self.agent.set_subagent_system_prompt(subagent_prompt)
                self.agent.transcript = list(data["transcript"])
            except (OSError, ValueError, KeyError) as exc:
                await self.add(Static(str(exc), classes="error"))
            else:
                self.session_id = argument
                await conversation.remove_children()
                self.query_one("#banner", Static).update(self.banner_text())
                await self.add(
                    Static(
                        f"Loaded {argument} ({len(self.agent.transcript)} transcript entries)",
                        classes="tool-result",
                    )
                )
            return True

        if command == "/help":
            await self.add(
                Static(
                    "/clear  clear context  •  /new  new saved session\n"
                    "/copy  copy the latest answer to the clipboard\n"
                    "/sessions  list sessions  •  /load ID  restore one\n"
                    "/undo  remove latest turn  •  /cwd PATH  change workspace\n"
                    "/plan [on|off|TASK]  safe read-only planning mode\n"
                    "/deep [on|off|TASK]  reason → plan → revise → execute\n"
                    "/subagent-prompt [TEXT|@FILE|reset]  customize delegated-agent behavior\n"
                    "/model  show the two fixed models  •  /exit  quit\n"
                    "While bob-der works, keep typing: follow-ups are queued automatically.\n"
                    "Enter adds a line; Ctrl+Enter sends the whole prompt.\n"
                    "Click any Thinking row (or focus it and press Enter) to show/hide live reasoning.\n"
                    "Click Copy for a whole box, or select inside it and press Ctrl+C. Quit with Ctrl+Q.",
                    classes="tool-result",
                )
            )
            return True
        return False

    def banner_text(self) -> str:
        subagent = (
            self.agent.subagent_runner.model
            if self.agent.subagent_runner is not None
            else "disabled"
        )
        return (
            "[b]bob-der[/b]\n"
            f"Primary: {self.agent.runner.model}  •  Subagent: {subagent}\n"
            f"Eyes: {EYES_MODEL}\n"
            f"Mode: {'DEEP' if self.deep_mode else 'PLAN' if self.plan_mode else 'EXECUTE'}"
            f"  •  {self.agent.workspace}"
        )

    def start_task(
        self, request: str, plan_mode: bool, deep_mode: bool
    ) -> None:
        self.busy = True
        self.active_plan_mode = plan_mode
        self.active_deep_mode = deep_mode
        self.update_prompt_hint()
        self.set_activity("deep" if deep_mode else "plan" if plan_mode else "thinking")
        self.execute_task(request, plan_mode, deep_mode)

    def update_mode_display(self) -> None:
        self.query_one("#banner", Static).update(self.banner_text())
        prompt = self.query_one("#prompt", PromptTextArea)
        prompt.set_class(self.plan_mode or self.deep_mode, "plan-mode")
        self.update_prompt_hint()

    def update_prompt_hint(self) -> None:
        prompt = self.query_one("#prompt", PromptTextArea)
        if self.busy:
            count = len(self.pending)
            suffix = f" ({count} queued)" if count else ""
            prompt.placeholder = (
                f"Type a follow-up — it will be queued{suffix}  ·  Ctrl+Enter to send"
            )
        else:
            mode_hint = (
                "Describe the task for deep reason → plan → revise → execute"
                if self.deep_mode
                else "Describe what you want planned — no tools will run"
                if self.plan_mode
                else "Ask bob-der to build, fix, or inspect something…"
            )
            prompt.placeholder = f"{mode_hint}  ·  Ctrl+Enter to send"

    def save_session(self) -> None:
        try:
            self.sessions.save(
                self.session_id,
                workspace=self.agent.workspace,
                model=self.agent.runner.model,
                transcript=self.agent.transcript,
                subagent_prompt=self.agent.subagent_system_prompt,
            )
        except OSError:
            pass

    async def add(self, widget: Any) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(widget)
        conversation.scroll_end(animate=False)

    def set_activity(self, kind: str, label: str = "") -> None:
        self.activity_kind = kind
        self.activity_label = label
        self.activity_frame = 0
        activity = self.query_one("#activity", Static)
        activity.styles.display = "none" if kind == "idle" else "block"
        if kind != "idle":
            self.animate_activity()

    def animate_activity(self) -> None:
        if self.activity_kind == "idle":
            return
        frame = self.activity_frame
        spinner = SPINNER[frame % len(SPINNER)]
        if self.activity_kind == "plan":
            phrase = PLAN_VERBS[(frame // 8) % len(PLAN_VERBS)]
            text = f"[bold #62a7e8]{spinner}[/] [#8fc8f4]{phrase}…[/]  [dim]plan mode · read-only tools[/]"
        elif self.activity_kind == "deep":
            phrase = DEEP_VERBS[(frame // 10) % len(DEEP_VERBS)]
            text = f"[bold #e5a84b]{spinner} ◆[/] [#f0c979]{phrase}…[/]  [dim]deep · unlimited steps[/]"
        elif self.activity_kind == "coding":
            symbol = CODE_FRAMES[(frame // 5) % len(CODE_FRAMES)]
            phrase = CODE_VERBS[(frame // 9) % len(CODE_VERBS)]
            text = f"[bold #71d49b]{spinner} {symbol}[/] [#9be7b8]{phrase}…[/]"
        elif self.activity_kind == "subagent":
            phrase = SUBAGENT_VERBS[(frame // 9) % len(SUBAGENT_VERBS)]
            text = f"[bold #d49cff]{spinner} ◈[/] [#d9b8f0]{phrase}…[/]  [dim]gpt-oss:20b-cloud[/]"
        else:
            phrase = THINKING_VERBS[(frame // 9) % len(THINKING_VERBS)]
            text = f"[bold #b785db]{spinner}[/] [#d9b8f0]{phrase}…[/]"
        if self.activity_label:
            text += f"  [dim]{self.activity_label}[/]"
        self.query_one("#activity", Static).update(text)
        self.activity_frame += 1

    def agent_event(self, event: str, data: dict[str, object]) -> None:
        self.call_from_thread(self.apply_agent_event, event, data)

    def apply_agent_event(self, event: str, data: dict[str, object]) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        if event == "model_start":
            self.set_activity(
                "deep"
                if self.active_deep_mode
                else "plan"
                if self.active_plan_mode
                else "thinking"
            )
            self.current_raw = ""
            self.current_received_thinking = False
            self.current_answer = None
            self.current_answer_text = ""
            self.current_thinking = Static(
                "Waiting for reasoning…", classes="thinking-body"
            )
            self.current_panel = Collapsible(
                self.current_thinking,
                title="✻ Thinking… (click to expand)",
                collapsed=True,
                classes="thinking-panel",
            )
            conversation.mount(self.current_panel)
            conversation.scroll_end(animate=False)
            return

        if event == "model_chunk":
            self.current_raw += str(data.get("chunk", ""))
            thinking = partial_thinking(self.current_raw)
            if self.current_thinking is not None and thinking:
                self.current_received_thinking = True
                self.current_thinking.update(thinking)
            answer = partial_final_message(self.current_raw)
            if answer is not None:
                if self.current_answer is None:
                    self.current_answer = CopyBox("", title="Answer", classes="answer")
                    conversation.mount(self.current_answer)
                # Worker callbacks may reach the UI just after finish_task. Never
                # let an older partial chunk replace the complete final answer.
                if (
                    answer != self.current_answer_text
                    and len(answer) >= len(self.current_answer_text)
                ):
                    self.current_answer_text = answer
                    self.last_copy_text = answer
                    self.current_answer.set_text(answer)
                    conversation.scroll_end(animate=False)
            return

        if event == "tool_start":
            if self.current_panel is not None:
                self.current_panel.title = "✻ Thinking (click to show)"
            tool = str(data.get("tool", "tool"))
            args = data.get("args", {})
            thought = str(data.get("thought", "")).strip()
            if self.current_thinking is not None and not self.current_received_thinking:
                self.current_thinking.update(
                    thought
                    or "Fast mode skipped extended reasoning. Use /deep for the full stream."
                )
            label = self.tool_label(tool, args if isinstance(args, dict) else {})
            self.set_activity(
                "subagent"
                if tool == "subagent"
                else "plan"
                if self.active_plan_mode
                else "coding",
                label[:80],
            )
            conversation.mount(Static(f"● {label}", classes="tool-call"))
            conversation.scroll_end(animate=False)
            return

        if event == "fallback":
            self.set_activity("subagent", "main model was slow")
            conversation.mount(
                Static(
                    f"↻ {data.get('from_model')} produced no output in time; "
                    f"{data.get('to_model')} is taking over this task.",
                    classes="tool-result",
                )
            )
            conversation.scroll_end(animate=False)
            return

        if event == "tool_end":
            self.set_activity(
                "deep"
                if self.active_deep_mode
                else "plan"
                if self.active_plan_mode
                else "thinking"
            )
            result = data.get("result", {})
            if isinstance(result, dict):
                summary = self.result_summary(str(data.get("tool", "")), result)
                if summary:
                    conversation.mount(Static(f"⎿ {summary}", classes="tool-result"))
                    conversation.scroll_end(animate=False)
            return

        if event == "deep_phase":
            stage = str(data.get("stage", "")).capitalize()
            content = str(data.get("content", ""))
            phase_body = CopyBox(content, title=f"{stage} output")
            conversation.mount(
                Collapsible(
                    phase_body,
                    title=f"◆ Deep phase: {stage} (click to show)",
                    collapsed=False,
                    classes="thinking-panel",
                )
            )
            conversation.scroll_end(animate=False)
            return

        if event == "final" and self.current_panel is not None:
            if self.current_thinking is not None and not self.current_received_thinking:
                thought = str(data.get("thought", "")).strip()
                self.current_thinking.update(
                    thought
                    or "Fast mode skipped extended reasoning. Use /deep for the full stream."
                )
            self.current_panel.title = "✻ Decision summary (click to show)"

    @staticmethod
    def tool_label(tool: str, args: dict[str, object]) -> str:
        if tool == "shell":
            return f"Bash({args.get('command', '')})"
        if tool == "subagent":
            task = str(args.get("task", ""))
            return f"Task({task[:100]}{'…' if len(task) > 100 else ''})"
        if tool == "describe_image":
            target = args.get("path") or args.get("url") or args.get("paths", "images")
            return f"Eyes({target})"
        names = {
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
        }
        return f"{names.get(tool, tool)}({args.get('path', '')})"

    @staticmethod
    def result_summary(tool: str, result: dict[str, object]) -> str:
        if not result.get("ok"):
            return f"[red]Error: {result.get('error', 'tool failed')}[/red]"
        if tool == "shell":
            output = str(result.get("output", "")).strip()
            code = result.get("exit_code", 0)
            return output or f"Command exited with code {code}"
        if tool == "subagent":
            return (
                f"{result.get('model')} completed in {result.get('steps')} steps\n"
                f"{result.get('message', '')}"
            ).strip()
        if "path" in result:
            return str(result["path"])
        return "Done"

    @work(thread=True, exit_on_error=False)
    def fetch_update_worker(self) -> None:
        from .updater import UpdateError, fetch_update, format_outcome

        try:
            message = format_outcome(fetch_update(download=True))
        except UpdateError as exc:
            message = f"Update failed: {exc}"
            is_error = True
        else:
            is_error = False
        self.call_from_thread(self.show_update_result, message, is_error)

    def show_update_result(self, message: str, is_error: bool) -> None:
        self.query_one("#conversation", VerticalScroll).mount(
            CopyBox(message, title="Update")
            if not is_error
            else Static(message, classes="error")
        )

    @work(thread=True, exit_on_error=False)
    def execute_task(
        self, request: str, plan_mode: bool, deep_mode: bool
    ) -> None:
        try:
            answer = self.agent.run(
                request,
                plan_mode=plan_mode,
                deep_mode=deep_mode,
            )
        except (OllamaError, RuntimeError) as exc:
            self.call_from_thread(self.finish_task, None, str(exc))
        except BaseException as exc:
            self.call_from_thread(self.finish_task, None, f"{type(exc).__name__}: {exc}")
        else:
            self.call_from_thread(self.finish_task, answer, None)

    def finish_task(self, answer: str | None, error: str | None) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        if error:
            conversation.mount(Static(f"Error: {error}", classes="error"))
        elif answer:
            self.current_answer_text = answer
            self.last_copy_text = answer
            if self.current_answer is None:
                self.current_answer = CopyBox(answer, title="Answer", classes="answer")
                conversation.mount(self.current_answer)
            else:
                self.current_answer.set_text(answer)
        self.save_session()
        prompt = self.query_one("#prompt", PromptTextArea)
        prompt.focus()
        if self.pending:
            request, queued_widget, plan_mode, deep_mode = self.pending.popleft()
            queued_widget.update(f"❯ {request}")
            queued_widget.set_classes("user-message")
            self.update_prompt_hint()
            self.active_plan_mode = plan_mode
            self.active_deep_mode = deep_mode
            self.set_activity(
                "deep" if deep_mode else "plan" if plan_mode else "thinking"
            )
            self.execute_task(request, plan_mode, deep_mode)
        else:
            self.busy = False
            self.set_activity("idle")
            self.update_prompt_hint()
        conversation.scroll_end(animate=False)

    def action_clear_conversation(self) -> None:
        if self.busy:
            self.query_one("#conversation", VerticalScroll).mount(
                Static("Wait for the current task before clearing.", classes="error")
            )
            return
        self.agent.clear()
        self.pending.clear()
        self.session_id = self.sessions.new_id()
        self.query_one("#conversation", VerticalScroll).remove_children()


def run_tui(agent: Agent) -> int:
    result = BobDerApp(agent).run()
    return int(result or 0)
