# bob-der

`bob-der` is a Claude Code-style autonomous coding agent for the command line.
The primary agent uses `ollama run nemotron-3-super:cloud` and can delegate
focused work to a `gpt-oss:20b-cloud` subagent. Both can inspect and edit files,
execute shell commands, run tests, and keep working until the task is complete.
Those are the only two autonomous agents: model switching and extra agent
spawning are not part of the runtime. A fixed vision-only
`gemma4:31b-cloud` model acts as its
“Eyes” for screenshots, photos, diagrams, OCR, and image comparison; it is a
tool of the main agent, not another autonomous subagent.
Responses stream live in an interactive terminal as the model generates them.
Every model turn includes a collapsed `Thinking` panel: click its title (or
focus it and press Enter) to reveal the complete reasoning live.

The prompt remains active while the agent works. Submit additional requests at
any time and they will be queued and executed in order.
Multiline clipboard text is preserved when pasted into the growing prompt.
Press `Enter` to insert a newline and `Ctrl+Enter` to send. `Shift+Enter` and
`Ctrl+S` are alternate send shortcuts for terminals with different key support.

## Install

Requirements:

- Python 3.10+
- [Ollama](https://ollama.com/) installed and authenticated for cloud models
- `bubblewrap` (`bwrap`) for read-only shell commands in plan mode; dedicated
  read/search tools still work when it is unavailable

Run it immediately from this directory, with no install step:

```bash
./bob-der
```

To install `bob-der` as a command available everywhere, use `pipx`:

```bash
pipx install .
```

Alternatively, install it into a virtual environment with `pip install -e .`.

Build a self-contained Debian package:

```bash
./packaging/build-deb.sh
sudo apt install ./dist/bob-der_0.9.6_all.deb
bob-der-setup
bob-der
```

Copy the `.deb` to each Debian/Ubuntu computer before installing it. The
package contains bob-der and its Python libraries; `bob-der-setup` installs
Ollama from its official Linux installer, opens Ollama sign-in, and prepares
the main, subagent, and Eyes cloud models for that computer. Cloud credentials
are deliberately not copied from the computer that built the package.

## Updates

Releases are published at <https://github.com/bobwdmai/bob-der/releases>.
Check for a newer version or download its verified Ubuntu package with:

```bash
bob-der --check-update
bob-der --update
```

Inside the interactive interface, `/update` performs the same verified
download. bob-der validates the release package against its published SHA-256
before saving it to `~/Downloads`; it then shows the `apt install` command.

## Interactive commands

```text
/help               Show commands and interaction tips
/plan [on|off|TASK] Inspect with read-only tools and produce a plan
/deep [on|off|TASK] Reason, plan, revise, then execute with no reasoning limit
/update             Fetch and verify a newer Ubuntu package from GitHub
/copy                Copy the latest answer to the system clipboard
/subagent-prompt X  Show/set instructions; X may be text, @FILE, or reset
/model              Show the main, subagent, and fixed Eyes model
/cwd [PATH]         Show or change the agent workspace
/undo               Remove the latest user turn from model context
/sessions           List automatically saved sessions
/load SESSION_ID    Restore a saved session
/new                Save the current session and start a new one
/clear              Clear the current conversation
/exit               Quit
```

Every final answer and deep-mode phase appears in a bordered copy box. Click
**Copy** to copy the entire box, or click inside it, select text, and press
`Ctrl+C`. Paste normally with `Ctrl+V`. bob-der now quits with `Ctrl+Q` so the
standard copy shortcut remains available.

bob-der also exposes 79 compatible tools recovered from the beta: project
search and inspection, Git, tests/lint/formatting, archives, HTTP, Docker,
SQLite/JSON/YAML/CSV, process and network diagnostics, clipboard, screenshots,
PDF creation, and more. The beta's extra-model tools are intentionally excluded,
so the runtime still has exactly one main model and one subagent.

Tool execution is isolated from the agent loop: malformed calls and optional
dependency failures return structured errors instead of crashing a run. File
writes are atomic, undecodable command output is replaced safely, and oversized
or non-JSON tool results are normalized before entering conversation context.

Sessions are saved atomically under `~/.bob-der/sessions/` after every completed
task. Follow-up requests entered during a task stay visible in the interface and
run sequentially when the current task finishes.

Typing `/` opens a clickable filtered command list. Activity animations distinguish
normal reasoning, plan mode, deep reasoning, coding tools, and subagent work.

Plan mode can read/search files and run shell commands in a filesystem sandbox,
but rejects tools that can edit files or mutate state. Deep mode requests Ollama's highest
reasoning level, removes the model timeout and agent-step ceiling, and enforces
the sequence `reason → plan → revise → execute`. Each phase and the underlying
thinking stream remain visible in expandable panels.

Or give it a one-shot task:

```bash
./bob-der "inspect this project, fix the failing tests, and verify the fix"
```

Use another working directory from anywhere:

```bash
./bob-der -C /path/to/project "add a health-check endpoint"
```

One-shot plan and deep runs are also available:

```bash
bob-der --plan "design the authentication migration"
bob-der --deep "implement and thoroughly verify the migration"
```

Useful options:

```text
--subagent-steps N        Step ceiling for each delegated task (default: 25)
--subagent-prompt TEXT    System instructions for the fixed gpt-oss subagent
--plan                    Read-only one-shot planning mode
--deep                    Unbounded deep workflow: reason, plan, revise, execute
-C, --directory PATH      Agent working directory (default: current directory)
--max-steps N             Autonomous step ceiling; 0 disables it
--model-timeout SECONDS   Optional timeout; 0 means unlimited (default)
--fallback-after SECONDS  Optional takeover delay; 0 disables it (default)
-v, --verbose             Show reasoning summaries and command output
-q, --quiet               Print only the final response
```

## Unrestricted execution

As requested, `bob-der` has no confirmation or permission gate. Shell commands
run directly with your user account, and both relative and absolute file paths
are allowed. Run it only in environments where you trust the model with the
permissions of the current OS user. Model calls have no timeout by default.
For no autonomous step ceiling either, use `--max-steps 0`.
