"""
Tool implementations — file I/O, bash, search, git, npm dev, browser, keyboard,
self-compaction signal, and meta add_tool.
"""

import json
import os
import re
import subprocess
import shlex
import shutil
import tempfile
import threading
import fnmatch
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# Avoid circular import — CONFIG_DIR is just a path
_CONFIG_DIR = Path.home() / ".bob-der"
DYNAMIC_TOOLS_FILE = _CONFIG_DIR / "dynamic_tools.py"

# ── Background process store ──────────────────────────────────────────────────
_bg_procs: dict[str, dict] = {}  # f"{cwd}:{script}" → {proc, logs, script, cwd}

# ── Dynamic tool registry ─────────────────────────────────────────────────────
_dynamic_fns: dict[str, Any] = {}
_dynamic_schemas: list[dict] = []

# ── Ollama tool schemas ───────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns file content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "Start line (1-based)"},
                    "limit": {"type": "integer", "description": "Max lines"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact unique string in a file (surgical edit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command. Returns stdout, stderr, returncode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Default: ."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files (ripgrep if available, else grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "case_insensitive": {"type": "boolean"},
                    "file_pattern": {"type": "string", "description": "e.g. '*.py'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run a git command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string"},
                },
                "required": ["args"],
            },
        },
    },
    # ── New tools ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "npm_dev",
            "description": (
                "Manage a background npm dev/test/build server. "
                "action: start | stop | logs | status. "
                "Use script to override the npm script name (default: dev)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "logs", "status"],
                    },
                    "script": {"type": "string", "description": "npm script name (default: dev)"},
                    "port": {"type": "integer", "description": "Expected port to check"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": (
                "CLI web browser — fetch a URL and return readable text content. "
                "Uses w3m/lynx if installed, otherwise requests + HTML stripping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "raw": {"type": "boolean", "description": "Return raw HTML instead of text"},
                    "links_only": {"type": "boolean", "description": "Return only hyperlinks"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyboard",
            "description": (
                "Simulate keyboard input via xdotool. "
                "action: type | key | focus | screenshot. "
                "Requires xdotool (apt install xdotool)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["type", "key", "focus", "screenshot"],
                    },
                    "text": {"type": "string", "description": "Text to type (action=type)"},
                    "keys": {"type": "string", "description": "Key combo e.g. 'ctrl+c' (action=key)"},
                    "window": {"type": "string", "description": "Window name/id for focus or targeting"},
                    "delay_ms": {"type": "integer", "description": "Delay between keystrokes ms (default 0)"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_tool",
            "description": (
                "Add a new tool to abby-der at runtime. "
                "The tool is immediately available and persisted to ~/.abby-der/dynamic_tools.py. "
                "python_body is the function body only (not the def line). "
                "The function receives cwd:str and **kwargs matching the parameters schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool name (snake_case, no spaces)"},
                    "description": {"type": "string"},
                    "parameters_schema": {
                        "type": "string",
                        "description": "JSON string of the Ollama parameters schema object",
                    },
                    "python_body": {
                        "type": "string",
                        "description": "Python function body (indented 4 spaces). Must return a dict with ok:bool.",
                    },
                    "auto_approve": {
                        "type": "string",
                        "enum": ["reads", "writes", "bash"],
                        "description": "Auto-approve category (default: bash = manual)",
                    },
                },
                "required": ["name", "description", "parameters_schema", "python_body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "package",
            "description": (
                "Install, remove, update, check, or list system/language packages. "
                "Supports: apt (system), pip (Python), npm (Node), cargo (Rust), "
                "snap, gem (Ruby), go. "
                "Set manager='auto' to pick the best one automatically. "
                "apt/snap use sudo automatically. "
                "WARNING: this runs real install commands — always confirm with the user first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["install", "remove", "update", "check", "search", "list"],
                        "description": "Operation to perform",
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Package name(s). Empty list is valid for 'list' and apt 'update'.",
                    },
                    "manager": {
                        "type": "string",
                        "enum": ["auto", "apt", "pip", "pip3", "npm", "cargo", "snap", "gem", "go", "flatpak"],
                        "description": "Package manager (default: auto-detect from environment)",
                    },
                    "flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra flags passed verbatim, e.g. ['--user'] for pip or ['--classic'] for snap",
                    },
                },
                "required": ["action", "packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a question with selectable options. "
                "The user navigates with ↑/↓ arrow keys and confirms with Enter. "
                "For multi_select=true, Space toggles options and Enter confirms. "
                "Returns the user's selection. Use this to clarify ambiguous requests, "
                "ask preferences, or confirm before taking irreversible actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of choices for the user",
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": "Allow selecting multiple options with Space (default: false)",
                    },
                    "allow_freetext": {
                        "type": "boolean",
                        "description": "Append a 'type custom answer' option at the end",
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact_conversation",
            "description": (
                "Request the CLI to compact the conversation history into a summary. "
                "Call this when the context is growing very long and you want to free up space. "
                "The CLI will generate a summary and replace the history."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ── New tools ─────────────────────────────────────────────────────────────
    {"type":"function","function":{"name":"http_request","description":"Make an HTTP request (GET/POST/PUT/PATCH/DELETE). Returns status, headers, body.","parameters":{"type":"object","properties":{"method":{"type":"string","enum":["GET","POST","PUT","PATCH","DELETE"],"description":"HTTP method"},"url":{"type":"string"},"headers":{"type":"object","description":"Extra request headers"},"body":{"type":"string","description":"Request body (JSON string or plain text)"},"timeout":{"type":"number","description":"Seconds (default 15)"}},"required":["method","url"]}}},
    {"type":"function","function":{"name":"lint","description":"Run a linter on a file or directory. Auto-selects ruff (Python), eslint (JS/TS), shellcheck (shell). Returns issues list.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory to lint"},"linter":{"type":"string","description":"Force a specific linter: ruff, eslint, shellcheck (default: auto)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"test_run","description":"Run the project test suite. Auto-detects pytest, jest, cargo test, go test. Returns pass/fail summary and output.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory or test file to run (default: cwd)"},"runner":{"type":"string","description":"Force runner: pytest, jest, cargo, go (default: auto)"},"args":{"type":"string","description":"Extra args to pass to the test runner"}},"required":[]}}},
    {"type":"function","function":{"name":"sqlite_query","description":"Run a SQL query on a local SQLite database file. Returns rows as list of dicts.","parameters":{"type":"object","properties":{"db":{"type":"string","description":"Path to the .sqlite / .db file"},"query":{"type":"string","description":"SQL query to execute"},"params":{"type":"array","items":{"type":"string"},"description":"Positional parameters for the query"}},"required":["db","query"]}}},
    {"type":"function","function":{"name":"secret_scan","description":"Scan files for hardcoded secrets (API keys, tokens, passwords). Returns matches with file/line.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory to scan (default: cwd)"},"include":{"type":"string","description":"Glob pattern for files to include, e.g. '*.py'"}},"required":[]}}},
    {"type":"function","function":{"name":"diff_files","description":"Show a unified diff between two files.","parameters":{"type":"object","properties":{"a":{"type":"string","description":"First file path"},"b":{"type":"string","description":"Second file path"},"context":{"type":"integer","description":"Lines of context (default 3)"}},"required":["a","b"]}}},
    {"type":"function","function":{"name":"rename_file","description":"Rename or move a file or directory.","parameters":{"type":"object","properties":{"src":{"type":"string"},"dst":{"type":"string"}},"required":["src","dst"]}}},
    {"type":"function","function":{"name":"copy_file","description":"Copy a file or directory tree to a destination.","parameters":{"type":"object","properties":{"src":{"type":"string"},"dst":{"type":"string"}},"required":["src","dst"]}}},
    {"type":"function","function":{"name":"delete_file","description":"Delete a file or directory (recursive for dirs). Use with care.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"hash_file","description":"Compute MD5 and SHA-256 hash of a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"todo_scan","description":"Find all TODO, FIXME, HACK, XXX comments in a file tree. Returns list with file/line/text.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory or file to scan (default: cwd)"},"tags":{"type":"array","items":{"type":"string"},"description":"Comment tags to find (default: TODO FIXME HACK XXX)"}},"required":[]}}},
    {"type":"function","function":{"name":"format_code","description":"Auto-format source files in-place. Auto-selects black (Python), prettier (JS/TS/CSS), gofmt (Go), rustfmt (Rust).","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory to format"},"formatter":{"type":"string","description":"Force formatter: black, prettier, gofmt, rustfmt"}},"required":["path"]}}},
    {"type":"function","function":{"name":"port_kill","description":"Kill the process listening on a given TCP port.","parameters":{"type":"object","properties":{"port":{"type":"integer"}},"required":["port"]}}},
    {"type":"function","function":{"name":"notify","description":"Send a desktop notification via notify-send.","parameters":{"type":"object","properties":{"title":{"type":"string"},"body":{"type":"string"},"icon":{"type":"string","description":"Icon name or path (optional)"}},"required":["title"]}}},
    {"type":"function","function":{"name":"git_log","description":"Structured git commit log with author, date, hash, message. Supports limit and file filter.","parameters":{"type":"object","properties":{"limit":{"type":"integer","description":"Max commits (default 20)"},"file":{"type":"string","description":"Limit to commits touching this file"},"branch":{"type":"string","description":"Branch or ref (default: HEAD)"}},"required":[]}}},
    {"type":"function","function":{"name":"git_blame","description":"Show who last changed each line of a file (git blame).","parameters":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer","description":"Start line"},"end":{"type":"integer","description":"End line"}},"required":["path"]}}},
    {"type":"function","function":{"name":"git_stash","description":"Manage git stashes: push, pop, list, drop.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["push","pop","list","drop"],"description":"Stash action"},"message":{"type":"string","description":"Stash message for push"},"index":{"type":"integer","description":"Stash index for drop (default 0)"}},"required":["action"]}}},
    {"type":"function","function":{"name":"type_check","description":"Run mypy (Python) or tsc (TypeScript) type checker on a file or directory.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory to check"},"checker":{"type":"string","description":"Force checker: mypy or tsc (default: auto)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"ps_list","description":"List running processes with PID, CPU%, memory%, and command.","parameters":{"type":"object","properties":{"filter":{"type":"string","description":"Filter by process name (optional)"}},"required":[]}}},
    {"type":"function","function":{"name":"kill_proc","description":"Kill a process by PID or name (sends SIGTERM, then SIGKILL if needed).","parameters":{"type":"object","properties":{"target":{"type":"string","description":"PID (number) or process name"},"force":{"type":"boolean","description":"Use SIGKILL immediately (default false)"}},"required":["target"]}}},
    {"type":"function","function":{"name":"disk_usage","description":"Show disk space (df) and directory size (du) for a path.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to inspect (default: cwd)"}},"required":[]}}},
    {"type":"function","function":{"name":"service_ctl","description":"Control a systemd service: start, stop, restart, status, enable, disable.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["start","stop","restart","status","enable","disable"]},"service":{"type":"string"}},"required":["action","service"]}}},
    {"type":"function","function":{"name":"journalctl","description":"Get recent systemd journal logs for a service unit.","parameters":{"type":"object","properties":{"unit":{"type":"string","description":"Service name, e.g. nginx"},"lines":{"type":"integer","description":"Number of lines (default 50)"},"follow":{"type":"boolean","description":"Follow for 3 seconds (default false)"}},"required":["unit"]}}},
    {"type":"function","function":{"name":"download_file","description":"Download a URL to a local file.","parameters":{"type":"object","properties":{"url":{"type":"string"},"dest":{"type":"string","description":"Destination file path (default: filename from URL in cwd)"}},"required":["url"]}}},
    {"type":"function","function":{"name":"dns_lookup","description":"Resolve a hostname to IP addresses (A, AAAA records).","parameters":{"type":"object","properties":{"host":{"type":"string"}},"required":["host"]}}},
    {"type":"function","function":{"name":"ping","description":"Ping a host and return latency stats.","parameters":{"type":"object","properties":{"host":{"type":"string"}},"required":["host"]}}},
    {"type":"function","function":{"name":"port_scan","description":"Scan localhost for open TCP ports in a range.","parameters":{"type":"object","properties":{"start":{"type":"integer","description":"Start port (default 1)"},"end":{"type":"integer","description":"End port (default 1024)"}},"required":[]}}},
    {"type":"function","function":{"name":"docker_ps","description":"List Docker containers (running or all).","parameters":{"type":"object","properties":{"all":{"type":"boolean","description":"Include stopped containers (default false)"}},"required":[]}}},
    {"type":"function","function":{"name":"docker_exec","description":"Run a command inside a running Docker container.","parameters":{"type":"object","properties":{"container":{"type":"string","description":"Container name or ID"},"command":{"type":"string","description":"Shell command to run inside"}},"required":["container","command"]}}},
    {"type":"function","function":{"name":"docker_logs","description":"Tail logs from a Docker container.","parameters":{"type":"object","properties":{"container":{"type":"string"},"lines":{"type":"integer","description":"Number of lines (default 50)"}},"required":["container"]}}},
    {"type":"function","function":{"name":"docker_stop","description":"Stop and optionally remove a Docker container.","parameters":{"type":"object","properties":{"container":{"type":"string"},"remove":{"type":"boolean","description":"Also remove the container (default false)"}},"required":["container"]}}},
    {"type":"function","function":{"name":"clipboard_get","description":"Read the current system clipboard contents.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"clipboard_set","description":"Write text to the system clipboard.","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},
    {"type":"function","function":{"name":"screenshot","description":"Take a screenshot and save to a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Output file path (default: /tmp/screenshot.png)"},"delay":{"type":"integer","description":"Delay in seconds before capture (default 0)"}},"required":[]}}},
    {"type":"function","function":{"name":"head_file","description":"Return the first N lines of a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"lines":{"type":"integer","description":"Number of lines (default 10)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"tail_file","description":"Return the last N lines of a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"lines":{"type":"integer","description":"Number of lines (default 10)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"json_query","description":"Run a jq-style query on a JSON file or string. Returns the matching data.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"JSON file path (or omit to pass raw json)"},"query":{"type":"string","description":"jq expression, e.g. .users[0].name"},"json":{"type":"string","description":"Inline JSON string (if no file)"}},"required":["query"]}}},
    {"type":"function","function":{"name":"base64_encode","description":"Base64-encode a string or file.","parameters":{"type":"object","properties":{"text":{"type":"string","description":"String to encode (or omit to encode a file)"},"path":{"type":"string","description":"File to encode"}},"required":[]}}},
    {"type":"function","function":{"name":"base64_decode","description":"Base64-decode a string.","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},
    {"type":"function","function":{"name":"uuid_gen","description":"Generate one or more UUIDs.","parameters":{"type":"object","properties":{"count":{"type":"integer","description":"How many to generate (default 1)"}},"required":[]}}},
    {"type":"function","function":{"name":"math_eval","description":"Evaluate a mathematical expression using Python (supports sympy for symbolic math).","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression, e.g. 'integrate(x**2, x)' or '2**32'"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"size_report","description":"Report lines of code per file and total for a directory.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to analyse (default: cwd)"},"ext":{"type":"string","description":"File extension filter, e.g. .py"}},"required":[]}}},
    {"type":"function","function":{"name":"watch_file","description":"Return the last N new lines appended to a log file within a timeout.","parameters":{"type":"object","properties":{"path":{"type":"string"},"lines":{"type":"integer","description":"Lines to tail (default 20)"},"timeout":{"type":"number","description":"Seconds to wait for new content (default 3)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"git_diff","description":"Show a git diff. Can compare working tree, staged, or two refs.","parameters":{"type":"object","properties":{"ref_a":{"type":"string","description":"First ref/branch/commit (default: working tree)"},"ref_b":{"type":"string","description":"Second ref (optional)"},"path":{"type":"string","description":"Limit diff to this file/dir"},"staged":{"type":"boolean","description":"Show staged changes (default false)"}},"required":[]}}},
    {"type":"function","function":{"name":"git_branch","description":"List, create, rename, or delete git branches.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["list","create","delete","rename","checkout"],"description":"What to do"},"name":{"type":"string","description":"Branch name"},"new_name":{"type":"string","description":"New branch name (for rename)"},"force":{"type":"boolean","description":"Force delete/rename (default false)"}},"required":["action"]}}},
    {"type":"function","function":{"name":"archive_create","description":"Create a zip or tar.gz archive from files or a directory.","parameters":{"type":"object","properties":{"output":{"type":"string","description":"Output archive path (.zip or .tar.gz)"},"sources":{"type":"array","items":{"type":"string"},"description":"Files/dirs to include"},"format":{"type":"string","enum":["zip","tar.gz","tar.bz2"],"description":"Archive format (auto-detected from extension if omitted)"}},"required":["output","sources"]}}},
    {"type":"function","function":{"name":"archive_extract","description":"Extract a zip, tar.gz, or tar.bz2 archive.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Archive file path"},"dest":{"type":"string","description":"Destination directory (default: cwd)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"archive_list","description":"List contents of a zip or tar archive without extracting.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Archive file path"}},"required":["path"]}}},
    {"type":"function","function":{"name":"yaml_query","description":"Read and query a YAML file. Returns parsed data or a specific key path.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"YAML file path"},"key":{"type":"string","description":"Dot-separated key path, e.g. 'server.port' (optional, returns full file if omitted)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"env_get","description":"Read environment variables. Returns all env vars or a specific one.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Specific variable name (optional)"}},"required":[]}}},
    {"type":"function","function":{"name":"open_file","description":"Open a file or URL with the system default application (xdg-open on Linux, open on Mac).","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path or URL to open"}},"required":["path"]}}},
    {"type":"function","function":{"name":"regex_test","description":"Test a regular expression against a string. Returns matches and groups.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Regex pattern"},"text":{"type":"string","description":"Text to match against"},"flags":{"type":"array","items":{"type":"string"},"description":"Flags: ignorecase, multiline, dotall"}},"required":["pattern","text"]}}},
    {"type":"function","function":{"name":"csv_query","description":"Query a CSV file: read all rows, filter by column value, or get column stats.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"CSV file path"},"filter_col":{"type":"string","description":"Column name to filter on"},"filter_val":{"type":"string","description":"Value to match in filter_col"},"columns":{"type":"array","items":{"type":"string"},"description":"Columns to return (default: all)"},"limit":{"type":"integer","description":"Max rows (default 100)"}},"required":["path"]}}},
    {"type":"function","function":{"name":"image_info","description":"Get image metadata: dimensions, format, color mode, file size.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Image file path"}},"required":["path"]}}},
    {"type":"function","function":{"name":"network_info","description":"List network interfaces with IP addresses, MAC, and status.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"uptime_info","description":"Show system uptime, load averages, and memory usage.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"ssl_check","description":"Check SSL/TLS certificate for a hostname: expiry, issuer, SANs.","parameters":{"type":"object","properties":{"host":{"type":"string","description":"Hostname (e.g. example.com)"},"port":{"type":"integer","description":"Port (default 443)"}},"required":["host"]}}},
    {"type":"function","function":{"name":"whois_lookup","description":"WHOIS lookup for a domain name or IP address.","parameters":{"type":"object","properties":{"target":{"type":"string","description":"Domain name or IP"}},"required":["target"]}}},
    {"type":"function","function":{"name":"top_procs","description":"Show top N processes by CPU or memory usage.","parameters":{"type":"object","properties":{"sort_by":{"type":"string","enum":["cpu","memory"],"description":"Sort key (default: cpu)"},"limit":{"type":"integer","description":"Max processes (default 10)"}},"required":[]}}},
    {"type":"function","function":{"name":"cron_list","description":"List cron jobs for the current user (and optionally root).","parameters":{"type":"object","properties":{"all_users":{"type":"boolean","description":"Also show /etc/cron.d entries (default false)"}},"required":[]}}},
    {"type":"function","function":{"name":"coverage_run","description":"Run test coverage with coverage.py (Python) or jest --coverage (JS). Returns a summary.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to run in (default: cwd)"},"runner":{"type":"string","description":"Force runner: pytest or jest (default: auto)"}},"required":[]}}},
    {"type":"function","function":{"name":"ast_analyze","description":"Analyze a Python source file: list classes, functions, imports, and top-level variables.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Python file path"}},"required":["path"]}}},
    {"type":"function","function":{"name":"template_render","description":"Render a Jinja2 template string or file with provided variables.","parameters":{"type":"object","properties":{"template":{"type":"string","description":"Template string (or omit to use template_file)"},"template_file":{"type":"string","description":"Path to a .j2 / .jinja2 template file"},"variables":{"type":"object","description":"Key-value pairs passed to the template"}},"required":[]}}},
    {"type":"function","function":{"name":"sub_ai","description":"Spin up a lightweight sub-AI (Ollama model, default qwen2:0.5b) to answer a sub-question or do a focused task. Returns the model's response. Useful for delegation, summarisation, or quick lookups without using the main context.","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"The question or task to send to the sub-model"},"model":{"type":"string","description":"Ollama model to use (default: qwen2:0.5b)"},"system":{"type":"string","description":"Optional system prompt for the sub-model"},"timeout":{"type":"integer","description":"Max seconds to wait (default 60)"}},"required":["prompt"]}}},
    {"type":"function","function":{"name":"describe_image","description":"Describe or analyse one or more images with a local Ollama vision model. Auto-detects installed vision models; auto-pulls moondream if nothing found. Modes: describe (detailed), ocr (extract text), caption (one-liner), objects (structured list), code (transcribe code/terminal), emotions, compare (diff two images), qa (custom prompt). Batch via 'paths' list (parallel). Resizes before send to save VRAM. Caches by content hash. Can save markdown report.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Local image file path (.jpg, .png, .webp, etc.)"},"url":{"type":"string","description":"Image URL (alternative to path)"},"paths":{"type":"array","items":{"type":"string"},"description":"List of local paths for batch processing (parallel, max 2 at once)"},"compare_path":{"type":"string","description":"Second image path for compare mode"},"compare_url":{"type":"string","description":"Second image URL for compare mode"},"prompt":{"type":"string","description":"Custom question — overrides mode default (required for mode=qa)"},"mode":{"type":"string","enum":["describe","ocr","caption","objects","code","emotions","compare","qa"],"description":"Analysis mode (default: describe)"},"model":{"type":"string","description":"Ollama vision model (default: auto-detect best available, auto-pulls moondream)"},"auto_pull":{"type":"boolean","description":"Pull moondream if no vision model installed (default: true)"},"max_px":{"type":"integer","description":"Resize longest side to this before sending — saves VRAM (default: 1024)"},"use_cache":{"type":"boolean","description":"Skip re-processing identical images, keyed by content+prompt hash (default: true)"},"save_to":{"type":"string","description":"Write all descriptions to a markdown file at this path"},"timeout":{"type":"integer","description":"Seconds per image (default: 120)"}},"required":[]}}},
    {
        "type": "function",
        "function": {
            "name": "gamepad",
            "description": (
                "Interact with a connected gamepad/controller (e.g. UCOM Twin USB Gamepad). "
                "Actions: list — show connected gamepads; "
                "state — read current button and axis values; "
                "listen — wait for a button press (with timeout)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "state", "listen"],
                        "description": "What to do",
                    },
                    "device": {
                        "type": "string",
                        "description": "Device path, e.g. /dev/input/js0 (default: first found)",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Seconds to wait for input in 'listen' mode (default 5)",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_pdf",
            "description": (
                "Create a PDF file from markdown or plain text content. "
                "Renders headings (#/##/###), code blocks (```), and paragraphs. "
                "Use this whenever the user asks to generate, export, or save a PDF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Output file path (e.g. report.pdf)"},
                    "content": {"type": "string", "description": "Markdown or plain text content"},
                    "title":   {"type": "string", "description": "PDF document title metadata"},
                    "author":  {"type": "string", "description": "PDF author metadata"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve(path: str, cwd: str) -> str:
    p = Path(path)
    return str(p.resolve() if p.is_absolute() else (Path(cwd) / p).resolve())


def _number_lines(text: str, offset: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(offset + len(lines)))
    return "\n".join(f"{str(i + offset).rjust(width)}\t{line}" for i, line in enumerate(lines))


def _atomic_write_text(path: str, content: str) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            f.write(content)
        if target.exists():
            shutil.copymode(target, tmp_path)
        os.replace(tmp_path, target)
        return len(content.encode())
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

# ── Core tool implementations ─────────────────────────────────────────────────

_READ_MAX_CHARS = 8000

def tool_read_file(path: str, cwd: str, offset: int = None, limit: int = None) -> dict:
    full = _resolve(path, cwd)
    try:
        with open(full, "r", errors="replace") as f:
            lines = f.readlines()
        start = (offset - 1) if offset and offset > 0 else 0
        end = (start + limit) if limit else len(lines)
        content = "".join(lines[start:end])
        truncated = False
        if len(content) > _READ_MAX_CHARS:
            content = content[:_READ_MAX_CHARS]
            truncated = True
        result = {"ok": True, "path": full, "content": _number_lines(content, start + 1), "total_lines": len(lines)}
        if truncated:
            result["truncated"] = True
            result["note"] = f"Output capped at {_READ_MAX_CHARS} chars. Use offset/limit to read more."
        return result
    except FileNotFoundError:
        return {"ok": False, "error": f"File not found: {full}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_write_file(path: str, content: str, cwd: str) -> dict:
    full = _resolve(path, cwd)
    try:
        bytes_written = _atomic_write_text(full, content)
        return {"ok": True, "path": full, "bytes_written": bytes_written}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_edit_file(path: str, old_string: str, new_string: str, cwd: str) -> dict:
    full = _resolve(path, cwd)
    try:
        with open(full, "r", errors="replace") as f:
            original = f.read()
        count = original.count(old_string)
        if count == 0:
            return {"ok": False, "error": "old_string not found in file"}
        if count > 1:
            return {"ok": False, "error": f"old_string appears {count} times — be more specific"}
        _atomic_write_text(full, original.replace(old_string, new_string, 1))
        return {"ok": True, "path": full}
    except FileNotFoundError:
        return {"ok": False, "error": f"File not found: {full}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_bash(command: str, cwd: str, timeout: int = 0) -> dict:
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
        )
        result = {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
        if r.returncode != 0:
            result["error"] = f"Command exited with status {r.returncode}"
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_list_dir(path: str, cwd: str) -> dict:
    full = _resolve(path or ".", cwd)
    try:
        entries = sorted(os.listdir(full))
        result = []
        for e in entries:
            ep = os.path.join(full, e)
            result.append(f"{e}/" if os.path.isdir(ep) else f"{e}  ({os.path.getsize(ep)} bytes)")
        return {"ok": True, "path": full, "entries": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_grep(pattern: str, cwd: str, path: str = ".", case_insensitive: bool = False, file_pattern: str = None) -> dict:
    full_path = _resolve(path or ".", cwd)
    if not os.path.exists(full_path):
        return {"ok": False, "error": f"Path not found: {full_path}"}
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color=never"]
        if case_insensitive:
            cmd.append("-i")
        if file_pattern:
            cmd += ["-g", file_pattern]
        cmd += [pattern, full_path]
    else:
        cmd = ["grep", "-rn"] + (["-i"] if case_insensitive else [])
        if file_pattern:
            cmd += ["--include", file_pattern]
        cmd += [pattern, full_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode not in (0, 1):
            error = (r.stderr or r.stdout).strip()
            return {"ok": False, "error": error or f"Search failed with exit {r.returncode}", "returncode": r.returncode}
        all_lines = r.stdout.strip().splitlines()
        total = len(all_lines)
        lines = all_lines
        truncated = total > 200
        if truncated:
            lines = all_lines[:200] + [f"... ({total} total)"]
        return {"ok": True, "matches": lines, "count": total, "truncated": truncated}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Search timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_find_files(pattern: str, cwd: str, path: str = ".") -> dict:
    full_path = _resolve(path or ".", cwd)
    matches = []
    _skip = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build", ".next"}
    try:
        if not os.path.exists(full_path):
            return {"ok": False, "error": f"Path not found: {full_path}"}
        if os.path.isfile(full_path):
            fname = os.path.basename(full_path)
            if fnmatch.fnmatch(fname, pattern):
                matches.append(os.path.relpath(full_path, cwd))
            return {"ok": True, "matches": matches, "count": len(matches), "truncated": False}
        for root, dirs, files in os.walk(full_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _skip]
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    matches.append(os.path.relpath(os.path.join(root, fname), cwd))
        matches.sort()
        total = len(matches)
        truncated = total > 200
        if truncated:
            matches = matches[:200] + [f"... ({total} total)"]
        return {"ok": True, "matches": matches, "count": total, "truncated": truncated}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_git(args: str, cwd: str) -> dict:
    try:
        r = subprocess.run(["git"] + shlex.split(args), cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── npm dev ───────────────────────────────────────────────────────────────────

def tool_npm_dev(action: str, cwd: str, script: str = "dev", port: int = None) -> dict:
    key = f"{cwd}:{script}"

    if action == "start":
        if key in _bg_procs and _bg_procs[key]["proc"].poll() is None:
            return {"ok": False, "error": f"Already running (PID {_bg_procs[key]['proc'].pid}). Stop it first."}
        if not shutil.which("npm"):
            return {"ok": False, "error": "npm not found — is Node.js installed?"}

        logs: list[str] = []

        try:
            proc = subprocess.Popen(
                ["npm", "run", script],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            return {"ok": False, "error": "npm not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

        def _reader():
            for line in proc.stdout:
                logs.append(line.rstrip())
                if len(logs) > 500:
                    logs.pop(0)

        threading.Thread(target=_reader, daemon=True).start()
        _bg_procs[key] = {"proc": proc, "logs": logs, "script": script, "cwd": cwd}

        # Wait up to 5s for a "ready" signal
        import time
        _ready_patterns = re.compile(r"localhost|127\.0\.0\.1|ready|started|listening|compiled|dev server", re.I)
        for _ in range(25):
            time.sleep(0.2)
            if proc.poll() is not None:
                return {"ok": False, "error": f"Process exited early (rc={proc.returncode})", "logs": logs[-20:]}
            if any(_ready_patterns.search(ln) for ln in logs[-10:]):
                break

        if port:
            url = f"http://localhost:{port}"
        else:
            # Extract port from logs
            url_match = re.search(r"https?://localhost:?(\d+)", "\n".join(logs))
            url = url_match.group(0) if url_match else None

        return {
            "ok": True,
            "action": "started",
            "pid": proc.pid,
            "script": script,
            "url": url,
            "initial_logs": logs[-30:],
        }

    elif action == "stop":
        if key not in _bg_procs:
            return {"ok": False, "error": "No running process found for this cwd/script"}
        entry = _bg_procs.pop(key)
        proc = entry["proc"]
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return {"ok": True, "action": "stopped", "pid": proc.pid}

    elif action == "logs":
        if key not in _bg_procs:
            return {"ok": False, "error": "No running process found"}
        entry = _bg_procs[key]
        return {
            "ok": True,
            "running": entry["proc"].poll() is None,
            "pid": entry["proc"].pid,
            "logs": entry["logs"][-80:],
        }

    elif action == "status":
        if key not in _bg_procs:
            return {"ok": True, "running": False}
        entry = _bg_procs[key]
        running = entry["proc"].poll() is None
        return {"ok": True, "running": running, "pid": entry["proc"].pid, "script": entry["script"]}

    return {"ok": False, "error": f"Unknown action '{action}'. Use start/stop/logs/status"}

# ── CLI browser ───────────────────────────────────────────────────────────────

class _HtmlToText(HTMLParser):
    """Minimal but effective HTML → text converter."""

    _BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "tr", "article", "section", "header", "footer", "pre"}
    _SKIP = {"script", "style", "noscript", "svg", "iframe", "head"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title: str = ""
        self._skip_depth = 0
        self._in_title = False
        self._cur_href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "a":
            d = dict(attrs)
            self._cur_href = d.get("href", "")
        if tag == "img":
            d = dict(attrs)
            alt = d.get("alt", "")
            if alt:
                self.parts.append(f"[img: {alt}] ")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._cur_href:
            self.links.append(self._cur_href)
            self._cur_href = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = text
        else:
            self.parts.append(text + " ")

    def result(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r" {2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def tool_browse(url: str, cwd: str, raw: bool = False, links_only: bool = False) -> dict:
    import html as _html

    # Try external text browsers first
    for browser in ("w3m", "lynx", "elinks"):
        if shutil.which(browser):
            flags = {
                "w3m": ["-dump", "-T", "text/html"],
                "lynx": ["-dump", "-nolist"],
                "elinks": ["-dump"],
            }[browser]
            try:
                r = subprocess.run([browser] + flags + [url], capture_output=True, text=True, timeout=20)
                if r.returncode == 0 and r.stdout.strip():
                    return {
                        "ok": True,
                        "url": url,
                        "renderer": browser,
                        "content": r.stdout[:8000],
                        "truncated": len(r.stdout) > 8000,
                    }
            except Exception:
                pass

    # Fallback: requests + HTML parser
    try:
        import requests as _req
        headers = {"User-Agent": "abby-der/2.1 (CLI browser; +https://github.com/abby-der)"}
        resp = _req.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()

        ctype = resp.headers.get("content-type", "")

        if raw:
            return {"ok": True, "url": resp.url, "status": resp.status_code, "content": resp.text[:6000]}

        if "text/html" not in ctype and "text/plain" not in ctype:
            return {"ok": True, "url": resp.url, "status": resp.status_code, "content_type": ctype, "size": len(resp.content)}

        if "text/plain" in ctype:
            return {"ok": True, "url": resp.url, "status": resp.status_code, "content": resp.text[:8000]}

        parser = _HtmlToText()
        parser.feed(_html.unescape(resp.text))
        text = parser.result()

        if links_only:
            return {"ok": True, "url": resp.url, "links": parser.links[:100]}

        return {
            "ok": True,
            "url": resp.url,
            "status": resp.status_code,
            "title": parser.title,
            "content": text[:8000],
            "links": parser.links[:30],
            "truncated": len(text) > 8000,
            "renderer": "requests+htmlparser",
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── Keyboard (xdotool) ────────────────────────────────────────────────────────

def tool_keyboard(action: str, cwd: str, text: str = None, keys: str = None,
                  window: str = None, delay_ms: int = 0) -> dict:
    # Wayland: try ydotool; X11: xdotool
    for tool in ("xdotool", "ydotool"):
        if shutil.which(tool):
            _kbtool = tool
            break
    else:
        return {
            "ok": False,
            "error": "Neither xdotool nor ydotool found.\n"
                     "Install: sudo apt install xdotool  (X11)\n"
                     "      or: sudo apt install ydotool (Wayland)",
        }

    def _run(*cmd_args):
        try:
            r = subprocess.run(list(cmd_args), capture_output=True, text=True, timeout=10)
            return {"ok": r.returncode == 0, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Keyboard command timed out"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if action == "type":
        if not text:
            return {"ok": False, "error": "text is required for action='type'"}
        if _kbtool == "xdotool":
            cmd = [_kbtool, "type", "--delay", str(delay_ms)]
            if window:
                cmd += ["--window", window]
            cmd += ["--", text]
        else:
            cmd = [_kbtool, "type", text]
        return _run(*cmd)

    elif action == "key":
        if not keys:
            return {"ok": False, "error": "keys is required for action='key'"}
        if _kbtool == "xdotool":
            cmd = [_kbtool, "key"]
            if window:
                cmd += ["--window", window]
            cmd += keys.split()
        else:
            cmd = [_kbtool, "key", keys]
        return _run(*cmd)

    elif action == "focus":
        if not window:
            return {"ok": False, "error": "window is required for action='focus'"}
        if _kbtool == "xdotool":
            r = subprocess.run(
                [_kbtool, "search", "--name", window],
                capture_output=True, text=True, timeout=5,
            )
            wid = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
            if not wid:
                return {"ok": False, "error": f"Window '{window}' not found"}
            return _run(_kbtool, "windowfocus", wid)
        return {"ok": False, "error": "focus not supported with ydotool"}

    elif action == "screenshot":
        out = f"/tmp/abby-der-shot-{os.getpid()}.png"
        for scrot in ("scrot", "gnome-screenshot", "spectacle"):
            if not shutil.which(scrot):
                continue
            flags = {"scrot": [out], "gnome-screenshot": ["-f", out], "spectacle": ["-o", out, "-b"]}[scrot]
            r = _run(scrot, *flags)
            if r.get("ok"):
                return {"ok": True, "path": out, "tool": scrot}
        return {"ok": False, "error": "No screenshot tool found (scrot, gnome-screenshot, spectacle)"}

    return {"ok": False, "error": f"Unknown action '{action}'. Use type/key/focus/screenshot"}

# ── add_tool (meta) ───────────────────────────────────────────────────────────

def tool_add_tool(name: str, description: str, parameters_schema: str, python_body: str,
                  cwd: str, auto_approve: str = "bash") -> dict:
    # Validate name
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return {"ok": False, "error": "name must be lowercase snake_case, start with a letter"}

    existing_names = {s["function"]["name"] for s in TOOL_SCHEMAS} | set(_dynamic_fns)
    if name in existing_names:
        return {"ok": False, "error": f"Tool '{name}' already exists"}

    # Parse schema
    if isinstance(parameters_schema, str):
        try:
            schema_obj = json.loads(parameters_schema)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"Invalid parameters_schema JSON: {e}"}
    else:
        schema_obj = parameters_schema

    # Build and compile function
    fn_lines = [f"def tool_{name}(cwd='.', **kwargs):"]
    for line in python_body.splitlines():
        fn_lines.append(f"    {line}")
    fn_src = "\n".join(fn_lines)

    try:
        code_obj = compile(fn_src, f"<dynamic:{name}>", "exec")
    except SyntaxError as e:
        return {"ok": False, "error": f"Syntax error: {e}"}

    ns: dict[str, Any] = {
        "os": os, "subprocess": subprocess, "shlex": shlex, "shutil": shutil,
        "json": json, "Path": Path, "re": re,
    }
    try:
        exec(code_obj, ns)
        fn = ns[f"tool_{name}"]
    except Exception as e:
        return {"ok": False, "error": f"Error executing tool code: {e}"}

    # Build schema entry
    schema_entry = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema_obj,
        },
    }
    auto_flag = f"auto_approve_{auto_approve}"

    # Register in memory
    _dynamic_fns[name] = fn
    _dynamic_schemas.append(schema_entry)
    TOOL_SCHEMAS.append(schema_entry)
    AUTO_APPROVE_MAP[name] = auto_flag
    TOOL_DESCRIPTIONS[name] = (name, "magenta")

    # Persist
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    header = "# abby-der dynamic tools — auto-generated\nimport os, subprocess, shlex, json, re\nfrom pathlib import Path\n\n"
    existing = ""
    if DYNAMIC_TOOLS_FILE.exists():
        with open(DYNAMIC_TOOLS_FILE) as f:
            existing = f.read()
        if not existing.startswith("#"):
            existing = header
    else:
        existing = header

    entry_block = (
        f"\n# ── {name} ──\n"
        f"{fn_src}\n"
        f"_DYNAMIC_REGISTRY.append({{\n"
        f"    'name': {json.dumps(name)},\n"
        f"    'description': {json.dumps(description)},\n"
        f"    'schema': {json.dumps(schema_entry)},\n"
        f"    'auto_approve': {json.dumps(auto_flag)},\n"
        f"}})\n"
    )

    # Ensure _DYNAMIC_REGISTRY exists in file
    if "_DYNAMIC_REGISTRY" not in existing:
        existing += "\n_DYNAMIC_REGISTRY = []\n"

    with open(DYNAMIC_TOOLS_FILE, "w") as f:
        f.write(existing + entry_block)

    return {
        "ok": True,
        "name": name,
        "message": f"Tool '{name}' added and persisted to {DYNAMIC_TOOLS_FILE}",
    }


# ── Package manager ───────────────────────────────────────────────────────────

_MGR_NEEDS_SUDO = {"apt", "apt-get", "snap", "flatpak"}

def _detect_manager() -> str:
    for m in ("pip3", "pip", "npm", "cargo", "apt", "snap", "gem"):
        if shutil.which(m):
            return m
    return ""

def _can_sudo_nopass() -> bool:
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    return r.returncode == 0


def tool_package(action: str, packages: list, cwd: str,
                 manager: str = "auto", flags: list = None) -> dict:
    flags = flags or []

    # Normalise package list
    if isinstance(packages, str):
        packages = [p.strip() for p in packages.replace(",", " ").split() if p.strip()]

    # Resolve manager
    if manager == "auto":
        manager = _detect_manager()
        if not manager:
            return {"ok": False, "error": "No supported package manager found (apt, pip, npm, cargo, snap)"}
    manager = manager.lower()
    if manager == "pip" and shutil.which("pip3"):
        manager = "pip3"

    needs_sudo = manager in _MGR_NEEDS_SUDO
    sudo = ["sudo"] if needs_sudo else []

    # ── Build command ─────────────────────────────────────────────────────────
    cmd: list[str] = []

    if action == "install":
        if manager in ("apt", "apt-get"):
            cmd = sudo + ["apt-get", "install", "-y"] + flags + packages
        elif manager in ("pip", "pip3"):
            cmd = [manager, "install"] + flags + packages
        elif manager == "npm":
            cmd = ["npm", "install", "-g"] + flags + packages
        elif manager == "cargo":
            cmd = ["cargo", "install"] + flags + packages
        elif manager == "snap":
            cmd = sudo + ["snap", "install"] + flags + packages
        elif manager == "gem":
            cmd = ["gem", "install"] + flags + packages
        elif manager == "go":
            cmd = ["go", "install"] + flags + [f"{p}@latest" for p in packages]
        elif manager == "flatpak":
            cmd = sudo + ["flatpak", "install", "-y"] + flags + packages
        else:
            return {"ok": False, "error": f"install not supported for {manager}"}

    elif action == "remove":
        if manager in ("apt", "apt-get"):
            cmd = sudo + ["apt-get", "remove", "-y"] + flags + packages
        elif manager in ("pip", "pip3"):
            cmd = [manager, "uninstall", "-y"] + flags + packages
        elif manager == "npm":
            cmd = ["npm", "uninstall", "-g"] + flags + packages
        elif manager == "cargo":
            cmd = ["cargo", "uninstall"] + flags + packages
        elif manager == "snap":
            cmd = sudo + ["snap", "remove"] + flags + packages
        else:
            return {"ok": False, "error": f"remove not supported for {manager}"}

    elif action == "update":
        if manager in ("apt", "apt-get"):
            # update = refresh package lists; if packages given also upgrade them
            update_cmd = sudo + ["apt-get", "update"]
            r = subprocess.run(update_cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return {"ok": False, "error": r.stdout + r.stderr, "command": update_cmd}
            if packages:
                cmd = sudo + ["apt-get", "install", "--only-upgrade", "-y"] + packages
            else:
                return {"ok": True, "action": "update", "manager": manager,
                        "output": r.stdout[-2000:], "returncode": 0}
        elif manager in ("pip", "pip3"):
            cmd = [manager, "install", "--upgrade"] + flags + packages
        elif manager == "npm":
            cmd = ["npm", "update", "-g"] + flags + (packages or [])
        else:
            return {"ok": False, "error": f"update not supported for {manager}"}

    elif action == "check":
        results: dict[str, Any] = {}
        for pkg in packages:
            if manager in ("pip", "pip3"):
                r = subprocess.run([manager, "show", pkg], capture_output=True, text=True)
                info = {}
                for line in r.stdout.splitlines():
                    if ": " in line:
                        k, _, v = line.partition(": ")
                        info[k.lower()] = v
                results[pkg] = {"installed": r.returncode == 0, "version": info.get("version", "")}
            elif manager in ("apt", "apt-get"):
                r = subprocess.run(["dpkg", "-s", pkg], capture_output=True, text=True)
                ver = ""
                for line in r.stdout.splitlines():
                    if line.startswith("Version:"):
                        ver = line.split(": ", 1)[-1]
                results[pkg] = {"installed": r.returncode == 0, "version": ver}
            elif manager == "npm":
                r = subprocess.run(["npm", "list", "-g", "--depth=0", pkg],
                                    capture_output=True, text=True)
                results[pkg] = {"installed": pkg in r.stdout}
            elif manager == "snap":
                r = subprocess.run(["snap", "list", pkg], capture_output=True, text=True)
                results[pkg] = {"installed": r.returncode == 0}
            else:
                results[pkg] = {"installed": bool(shutil.which(pkg))}
        return {"ok": True, "action": "check", "manager": manager, "results": results}

    elif action == "search":
        if manager in ("apt", "apt-get"):
            cmd = ["apt-cache", "search"] + packages
        elif manager in ("pip", "pip3"):
            # pip index versions <pkg> (replaces deprecated pip search)
            if packages:
                cmd = [manager, "index", "versions", packages[0]]
            else:
                return {"ok": False, "error": "search requires at least one package name"}
        elif manager == "npm":
            cmd = ["npm", "search", "--no-description"] + packages
        else:
            return {"ok": False, "error": f"search not supported for {manager}"}

    elif action == "list":
        if manager in ("apt", "apt-get"):
            cmd = ["dpkg", "--get-selections"]
        elif manager in ("pip", "pip3"):
            cmd = [manager, "list", "--format=columns"]
        elif manager == "npm":
            cmd = ["npm", "list", "-g", "--depth=0"]
        elif manager == "snap":
            cmd = ["snap", "list"]
        elif manager == "cargo":
            cmd = ["cargo", "install", "--list"]
        else:
            return {"ok": False, "error": f"list not supported for {manager}"}

    else:
        return {"ok": False, "error": f"Unknown action '{action}'. Use install/remove/update/check/search/list"}

    if not cmd:
        return {"ok": False, "error": "Could not build command"}

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        # stdout captured; stdin=None so sudo can open /dev/tty for password if needed
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=None,
            text=True,
            timeout=300,
        )
        return {
            "ok": result.returncode == 0,
            "action": action,
            "manager": manager,
            "packages": packages,
            "command": " ".join(cmd),
            "output": result.stdout[-3000:] if result.stdout else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timed out after 300 s"}
    except FileNotFoundError as e:
        return {"ok": False, "error": f"{cmd[0]!r} not found — is {manager} installed?"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _gamepad_list() -> list[dict]:
    """Return all /dev/input/js* devices with their names."""
    import glob, struct, fcntl, array
    JSIOCGNAME = 0x80006a13  # ioctl to get joystick name (up to 128 bytes)
    devices = []
    for path in sorted(glob.glob("/dev/input/js*")):
        name = "Unknown"
        try:
            with open(path, "rb") as f:
                buf = array.array("B", [0] * 128)
                fcntl.ioctl(f, JSIOCGNAME + (128 << 16), buf)
                name = bytes(buf).rstrip(b"\x00").decode(errors="replace").strip()
        except Exception:
            pass
        devices.append({"path": path, "name": name})
    return devices


def _gamepad_state(path: str) -> dict:
    """Read current button + axis state via the Linux joystick API."""
    import struct, select
    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS   = 0x02
    JS_EVENT_INIT   = 0x80
    FMT = "IhBB"   # time(u32), value(s16), type(u8), number(u8)
    SZ  = struct.calcsize(FMT)

    buttons: dict[int, int] = {}
    axes:    dict[int, int] = {}
    try:
        with open(path, "rb") as f:
            # Drain the init events (type | JS_EVENT_INIT) to get current state
            while True:
                ready, _, _ = select.select([f], [], [], 0.05)
                if not ready:
                    break
                raw = f.read(SZ)
                if len(raw) < SZ:
                    break
                _t, value, etype, number = struct.unpack(FMT, raw)
                if etype & JS_EVENT_BUTTON:
                    buttons[number] = value
                elif etype & JS_EVENT_AXIS:
                    axes[number] = value
    except PermissionError:
        return {"ok": False, "error": f"Permission denied: {path}  (try: sudo chmod a+r {path})"}
    except FileNotFoundError:
        return {"ok": False, "error": f"Device not found: {path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok": True, "device": path,
        "buttons": {f"btn{k}": v for k, v in sorted(buttons.items())},
        "axes":    {f"axis{k}": v for k, v in sorted(axes.items())},
    }


def _gamepad_listen(path: str, timeout: float) -> dict:
    """Block until a button-press event arrives or timeout expires."""
    import struct, select, time
    JS_EVENT_BUTTON = 0x01
    JS_EVENT_INIT   = 0x80
    FMT = "IhBB"
    SZ  = struct.calcsize(FMT)
    deadline = time.monotonic() + timeout
    try:
        with open(path, "rb") as f:
            # skip init events first
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                ready, _, _ = select.select([f], [], [], min(0.05, remaining))
                if not ready:
                    break
                raw = f.read(SZ)
                if len(raw) < SZ:
                    break
                _t, value, etype, number = struct.unpack(FMT, raw)
                if not (etype & JS_EVENT_INIT):
                    break  # done draining

            # now wait for a real button press
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                ready, _, _ = select.select([f], [], [], remaining)
                if not ready:
                    return {"ok": True, "device": path, "event": None, "timed_out": True}
                raw = f.read(SZ)
                if len(raw) < SZ:
                    continue
                _t, value, etype, number = struct.unpack(FMT, raw)
                if (etype & ~JS_EVENT_INIT) == JS_EVENT_BUTTON and value == 1:
                    return {"ok": True, "device": path,
                            "event": {"type": "button", "number": number, "value": value},
                            "timed_out": False}
    except PermissionError:
        return {"ok": False, "error": f"Permission denied: {path}  (try: sudo chmod a+r {path})"}
    except FileNotFoundError:
        return {"ok": False, "error": f"Device not found: {path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "device": path, "event": None, "timed_out": True}


def tool_gamepad(action: str, cwd: str, device: str = "", timeout: float = 5.0) -> dict:
    if action == "list":
        devices = _gamepad_list()
        if not devices:
            return {"ok": True, "devices": [], "note": "No gamepads found in /dev/input/js*"}
        return {"ok": True, "devices": devices}

    # resolve device path
    if not device:
        devices = _gamepad_list()
        if not devices:
            return {"ok": False, "error": "No gamepads found. Is the controller plugged in?"}
        device = devices[0]["path"]

    if action == "state":
        return _gamepad_state(device)
    if action == "listen":
        return _gamepad_listen(device, timeout)
    return {"ok": False, "error": f"Unknown action: {action}"}


def tool_http_request(method: str, url: str, cwd: str,
                      headers: dict = None, body: str = None, timeout: float = 15) -> dict:
    import requests as _req
    try:
        h = headers or {}
        resp = _req.request(method.upper(), url, headers=h,
                            data=body.encode() if body else None, timeout=timeout)
        ct = resp.headers.get("content-type", "")
        try:
            body_out = resp.json() if "json" in ct else resp.text[:4000]
        except Exception:
            body_out = resp.text[:4000]
        result = {"ok": resp.ok, "status": resp.status_code, "headers": dict(resp.headers),
                  "body": body_out, "url": resp.url}
        if not resp.ok:
            result["error"] = f"HTTP {resp.status_code}"
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_lint(path: str, cwd: str, linter: str = "") -> dict:
    full = _resolve(path, cwd)
    ext = Path(full).suffix.lower() if Path(full).is_file() else ""
    if not linter:
        if ext in (".py",) or Path(full).is_dir():
            linter = "ruff"
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            linter = "eslint"
        elif ext in (".sh", ".bash"):
            linter = "shellcheck"
        else:
            linter = "ruff"
    if linter == "ruff":
        cmd = ["ruff", "check", "--output-format=text", full]
    elif linter == "eslint":
        cmd = ["eslint", "--format=compact", full]
    elif linter == "shellcheck":
        cmd = ["shellcheck", full]
    else:
        return {"ok": False, "error": f"Unknown linter: {linter}"}
    if not shutil.which(cmd[0]):
        return {"ok": False, "error": f"{cmd[0]} not found — install it first"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    output = (r.stdout + r.stderr).strip()
    issues = [l for l in output.splitlines() if l.strip()]
    return {"ok": True, "linter": linter, "issues": issues,
            "count": len(issues), "clean": r.returncode == 0}


def tool_test_run(cwd: str, path: str = "", runner: str = "", args: str = "") -> dict:
    base_cwd = Path(cwd).resolve()
    target_path = Path(_resolve(path, cwd)) if path else Path(cwd).resolve()
    if not target_path.exists():
        return {"ok": False, "error": f"Path not found: {target_path}"}

    if target_path.is_file():
        try:
            path_args = [str(target_path.relative_to(base_cwd))]
            run_cwd = base_cwd
        except ValueError:
            path_args = [target_path.name]
            run_cwd = target_path.parent
    else:
        run_cwd = target_path
        path_args = []

    try:
        extra_args = shlex.split(args) if args else []
    except ValueError as e:
        return {"ok": False, "error": f"Invalid args: {e}"}

    if not runner:
        tests_dir = run_cwd / "tests"
        has_pytest_tests = (
            target_path.is_file() and target_path.suffix == ".py"
            or (run_cwd / "pytest.ini").exists()
            or any(run_cwd.glob("test_*.py"))
            or any(run_cwd.glob("*_test.py"))
            or (tests_dir.is_dir() and any(tests_dir.rglob("test_*.py")))
            or (tests_dir.is_dir() and any(tests_dir.rglob("*_test.py")))
        )
        if has_pytest_tests:
            runner = "pytest"
        elif (run_cwd / "package.json").exists():
            runner = "jest"
        elif (run_cwd / "Cargo.toml").exists():
            runner = "cargo"
        elif (run_cwd / "go.mod").exists():
            runner = "go"
        else:
            runner = "pytest"
    if runner == "pytest":
        cmd = ["python3", "-m", "pytest", "--tb=short", "-q"] + path_args + extra_args
    elif runner == "jest":
        cmd = ["npx", "jest", "--ci"] + path_args + extra_args
    elif runner == "cargo":
        cmd = ["cargo", "test"] + extra_args
    elif runner == "go":
        cmd = ["go", "test", "./..."] + extra_args
    else:
        return {"ok": False, "error": f"Unknown runner: {runner}"}
    if not shutil.which(cmd[0]):
        return {"ok": False, "runner": runner, "error": f"{cmd[0]} not found"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_cwd), timeout=120)
        out = (r.stdout + r.stderr)[-3000:]
        return {"ok": r.returncode == 0, "runner": runner, "output": out, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "runner": runner, "error": "Test run timed out after 120s"}
    except Exception as e:
        return {"ok": False, "runner": runner, "error": str(e)}


def tool_sqlite_query(db: str, query: str, cwd: str, params: list = None) -> dict:
    import sqlite3
    full = _resolve(db, cwd)
    try:
        con = sqlite3.connect(full)
        con.row_factory = sqlite3.Row
        cur = con.execute(query, params or [])
        if cur.description:
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchmany(500)]
            con.close()
            return {"ok": True, "columns": cols, "rows": rows, "count": len(rows)}
        else:
            con.commit()
            affected = cur.rowcount
            con.close()
            return {"ok": True, "affected": affected}
    except Exception as e:
        return {"ok": False, "error": str(e)}


_SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})', "API key"),
    (r'(?i)(secret|token)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})', "secret/token"),
    (r'sk-[A-Za-z0-9]{32,}', "OpenAI key"),
    (r'sk-or-v1-[A-Za-z0-9]{32,}', "OpenRouter key"),
    (r'ghp_[A-Za-z0-9]{36}', "GitHub PAT"),
    (r'AKIA[0-9A-Z]{16}', "AWS access key"),
    (r'(?i)password\s*[=:]\s*["\']?([^\s"\']{8,})', "password"),
    (r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----', "private key"),
]


def _redact_secret_line(line: str, pattern: str) -> str:
    match = re.search(pattern, line)
    if not match:
        return line.strip()[:120]
    group_index = match.lastindex or 0
    start, end = match.span(group_index)
    redacted = line[:start] + "[redacted]" + line[end:]
    return redacted.strip()[:120]


def tool_secret_scan(cwd: str, path: str = "", include: str = "") -> dict:
    import fnmatch as _fn
    target = Path(_resolve(path, cwd)) if path else Path(cwd).resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    matches = []
    files = list(target.rglob("*")) if target.is_dir() else [target]
    for f in files:
        if not f.is_file():
            continue
        if include and not _fn.fnmatch(f.name, include):
            continue
        if any(part.startswith(".") for part in f.parts):
            continue
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pattern, label in _SECRET_PATTERNS:
                if re.search(pattern, line):
                    snippet = _redact_secret_line(line, pattern)
                    matches.append({"file": str(f), "line": i, "type": label, "snippet": snippet})
                    break
    return {"ok": True, "matches": matches, "count": len(matches)}


def tool_diff_files(a: str, b: str, cwd: str, context: int = 3) -> dict:
    import difflib
    fa, fb = _resolve(a, cwd), _resolve(b, cwd)
    try:
        ta = Path(fa).read_text(errors="replace").splitlines(keepends=True)
        tb = Path(fb).read_text(errors="replace").splitlines(keepends=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    diff = list(difflib.unified_diff(ta, tb, fromfile=fa, tofile=fb, n=context))
    return {"ok": True, "diff": "".join(diff), "changed": len(diff) > 0}


def tool_rename_file(src: str, dst: str, cwd: str) -> dict:
    fs, fd = _resolve(src, cwd), _resolve(dst, cwd)
    try:
        shutil.move(fs, fd)
        return {"ok": True, "src": fs, "dst": fd}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_copy_file(src: str, dst: str, cwd: str) -> dict:
    fs, fd = _resolve(src, cwd), _resolve(dst, cwd)
    try:
        if Path(fs).is_dir():
            shutil.copytree(fs, fd)
        else:
            shutil.copy2(fs, fd)
        return {"ok": True, "src": fs, "dst": fd}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_delete_file(path: str, cwd: str) -> dict:
    full = _resolve(path, cwd)
    try:
        if Path(full).is_dir():
            shutil.rmtree(full)
        else:
            os.unlink(full)
        return {"ok": True, "path": full}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_hash_file(path: str, cwd: str) -> dict:
    import hashlib
    full = _resolve(path, cwd)
    try:
        data = Path(full).read_bytes()
        return {"ok": True, "path": full,
                "md5": hashlib.md5(data).hexdigest(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_todo_scan(cwd: str, path: str = "", tags: list = None) -> dict:
    target = Path(_resolve(path, cwd)) if path else Path(cwd).resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    tag_list = tags or ["TODO", "FIXME", "HACK", "XXX"]
    pattern = re.compile(r"(?i)(" + "|".join(re.escape(t) for t in tag_list) + r")[:\s](.{0,120})")
    matches = []
    files = list(target.rglob("*")) if target.is_dir() else [target]
    for f in files:
        if not f.is_file() or any(p.startswith(".") for p in f.parts):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                m = pattern.search(line)
                if m:
                    matches.append({"file": str(f), "line": i, "tag": m.group(1).upper(),
                                    "text": m.group(2).strip()})
        except Exception:
            continue
    return {"ok": True, "matches": matches, "count": len(matches)}


def tool_format_code(path: str, cwd: str, formatter: str = "") -> dict:
    full = _resolve(path, cwd)
    ext = Path(full).suffix.lower() if Path(full).is_file() else ""
    if not formatter:
        if ext == ".py" or Path(full).is_dir():
            formatter = "black"
        elif ext in (".js", ".ts", ".jsx", ".tsx", ".css", ".html", ".json"):
            formatter = "prettier"
        elif ext == ".go":
            formatter = "gofmt"
        elif ext == ".rs":
            formatter = "rustfmt"
        else:
            formatter = "black"
    if formatter == "black":
        cmd = ["black", full]
    elif formatter == "prettier":
        cmd = ["npx", "prettier", "--write", full]
    elif formatter == "gofmt":
        cmd = ["gofmt", "-w", full]
    elif formatter == "rustfmt":
        cmd = ["rustfmt", full]
    else:
        return {"ok": False, "error": f"Unknown formatter: {formatter}"}
    if not shutil.which(cmd[0]):
        return {"ok": False, "error": f"{cmd[0]} not found — install it first"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return {"ok": r.returncode == 0, "formatter": formatter, "path": full,
            "output": (r.stdout + r.stderr).strip()}


def tool_port_kill(port: int, cwd: str) -> dict:
    if not shutil.which("lsof"):
        return {"ok": False, "error": "lsof not found — install it first"}
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if r.returncode not in (0, 1):
        error = (r.stderr or r.stdout).strip() or f"lsof exited with status {r.returncode}"
        return {"ok": False, "error": error, "returncode": r.returncode}
    pids = r.stdout.strip().splitlines()
    if not pids:
        return {"ok": True, "note": f"No process found on port {port}"}
    killed = []
    ok = True
    for pid in pids:
        try:
            k = subprocess.run(["kill", "-9", pid], capture_output=True, text=True)
            entry = {"pid": pid, "killed": k.returncode == 0}
            if k.returncode != 0:
                ok = False
                entry["error"] = (k.stderr or k.stdout).strip() or f"kill exited with status {k.returncode}"
            killed.append(entry)
        except Exception as e:
            ok = False
            killed.append({"pid": pid, "killed": False, "error": str(e)})
    return {"ok": ok, "port": port, "killed": killed}


def tool_notify(title: str, cwd: str, body: str = "", icon: str = "") -> dict:
    cmd = ["notify-send"]
    if icon:
        cmd += ["-i", icon]
    cmd.append(title)
    if body:
        cmd.append(body)
    if not shutil.which("notify-send"):
        return {"ok": False, "error": "notify-send not found — install libnotify-bin"}
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": r.returncode == 0, "title": title, "body": body}


def tool_git_log(cwd: str, limit: int = 20, file: str = "", branch: str = "HEAD") -> dict:
    cmd = ["git", "log", branch, f"-{limit}", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=short"]
    if file:
        cmd += ["--", _resolve(file, cwd)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()}
    commits = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            commits.append({"hash": parts[0][:8], "author": parts[1], "email": parts[2],
                            "date": parts[3], "message": parts[4]})
    return {"ok": True, "commits": commits, "count": len(commits)}


def tool_git_blame(path: str, cwd: str, start: int = None, end: int = None) -> dict:
    full = _resolve(path, cwd)
    cmd = ["git", "blame", "--porcelain"]
    if start and end:
        cmd += [f"-L{start},{end}"]
    cmd.append(full)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()}
    lines = []
    cur = {}
    for line in r.stdout.splitlines():
        if line.startswith("\t"):
            cur["text"] = line[1:]
            lines.append(cur)
            cur = {}
        elif " " in line and not cur:
            parts = line.split()
            cur = {"hash": parts[0][:8]}
        elif line.startswith("author "):
            cur["author"] = line[7:]
        elif line.startswith("author-time "):
            import datetime
            cur["date"] = datetime.datetime.fromtimestamp(int(line[12:])).strftime("%Y-%m-%d")
        elif line.startswith("summary "):
            cur["summary"] = line[8:]
    return {"ok": True, "lines": lines}


def tool_git_stash(action: str, cwd: str, message: str = "", index: int = 0) -> dict:
    if action == "push":
        cmd = ["git", "stash", "push"] + (["-m", message] if message else [])
    elif action == "pop":
        cmd = ["git", "stash", "pop"]
    elif action == "list":
        cmd = ["git", "stash", "list"]
    elif action == "drop":
        cmd = ["git", "stash", "drop", f"stash@{{{index}}}"]
    else:
        return {"ok": False, "error": f"Unknown action: {action}"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip()}


def tool_type_check(path: str, cwd: str, checker: str = "") -> dict:
    full = _resolve(path, cwd)
    ext = Path(full).suffix.lower() if Path(full).is_file() else ""
    if not checker:
        checker = "tsc" if ext in (".ts", ".tsx") else "mypy"
    if checker == "mypy":
        cmd = ["python3", "-m", "mypy", "--ignore-missing-imports", full]
    elif checker == "tsc":
        cmd = ["npx", "tsc", "--noEmit", "--strict"]
    else:
        return {"ok": False, "error": f"Unknown checker: {checker}"}
    if checker == "mypy" and not shutil.which("mypy"):
        r = subprocess.run(["python3", "-m", "mypy", "--version"], capture_output=True)
        if r.returncode != 0:
            return {"ok": False, "error": "mypy not installed — run: pip install mypy"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    issues = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return {"ok": r.returncode == 0, "checker": checker, "issues": issues, "count": len(issues)}


def tool_ps_list(cwd: str, filter: str = "") -> dict:
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    lines = r.stdout.splitlines()
    header = lines[0] if lines else ""
    procs = []
    for line in lines[1:]:
        if filter and filter.lower() not in line.lower():
            continue
        parts = line.split(None, 10)
        if len(parts) >= 11:
            procs.append({"pid": parts[1], "cpu": parts[2], "mem": parts[3], "cmd": parts[10]})
    return {"ok": True, "processes": procs[:100], "count": len(procs)}


def tool_kill_proc(target: str, cwd: str, force: bool = False) -> dict:
    target = (target or "").strip()
    if not target:
        return {"ok": False, "error": "target required"}
    sig = "-9" if force else "-15"
    if target.isdigit():
        if int(target) <= 1:
            return {"ok": False, "pid": target, "error": f"Refusing to kill protected PID: {target}"}
        if not shutil.which("kill"):
            return {"ok": False, "pid": target, "error": "kill not found"}
        try:
            r = subprocess.run(["kill", sig, target], capture_output=True, text=True)
        except Exception as e:
            return {"ok": False, "pid": target, "error": str(e)}
        output = (r.stderr or r.stdout).strip()
        result = {"ok": r.returncode == 0, "pid": target, "returncode": r.returncode}
        if r.returncode != 0:
            result["error"] = output or f"kill exited with status {r.returncode}"
        return result
    if not shutil.which("pkill"):
        return {"ok": False, "name": target, "error": "pkill not found"}
    try:
        r = subprocess.run(["pkill", sig, "-f", target], capture_output=True, text=True)
    except Exception as e:
        return {"ok": False, "name": target, "error": str(e)}
    output = (r.stderr or r.stdout).strip()
    result = {"ok": r.returncode == 0, "name": target, "returncode": r.returncode}
    if r.returncode != 0:
        if r.returncode == 1 and not output:
            result["error"] = f"No process matched: {target}"
        else:
            result["error"] = output or f"pkill exited with status {r.returncode}"
    return result


def tool_disk_usage(cwd: str, path: str = "") -> dict:
    target = Path(_resolve(path, cwd)) if path else Path(cwd).resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    missing = [cmd for cmd in ("df", "du") if not shutil.which(cmd)]
    if missing:
        return {"ok": False, "error": f"Missing command(s): {', '.join(missing)}"}
    try:
        df = subprocess.run(["df", "-h", str(target)], capture_output=True, text=True)
        du = subprocess.run(["du", "-sh", str(target)], capture_output=True, text=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    result = {
        "ok": df.returncode == 0 and du.returncode == 0,
        "df": (df.stdout + df.stderr).strip(),
        "du": (du.stdout + du.stderr).strip(),
        "returncodes": {"df": df.returncode, "du": du.returncode},
    }
    if not result["ok"]:
        if df.returncode != 0:
            result["error"] = result["df"] or f"df exited with status {df.returncode}"
        elif du.returncode != 0:
            result["error"] = result["du"] or f"du exited with status {du.returncode}"
        else:
            result["error"] = "disk usage command failed"
    return result


def tool_service_ctl(action: str, service: str, cwd: str) -> dict:
    cmd = ["systemctl", action, service]
    if action in ("start", "stop", "restart", "enable", "disable"):
        cmd = ["sudo", "-n"] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": r.returncode == 0, "service": service, "action": action,
            "output": (r.stdout + r.stderr).strip()}


def tool_journalctl(unit: str, cwd: str, lines: int = 50, follow: bool = False) -> dict:
    cmd = ["journalctl", "-u", unit, f"-n{lines}", "--no-pager"]
    if follow:
        cmd = ["timeout", "3"] + cmd + ["-f"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    output = (r.stdout + r.stderr).strip()[-4000:]
    result = {"ok": r.returncode == 0, "unit": unit, "output": output, "returncode": r.returncode}
    if r.returncode != 0:
        result["error"] = output or f"journalctl exited with status {r.returncode}"
    return result


def tool_download_file(url: str, cwd: str, dest: str = "") -> dict:
    if not dest:
        dest = os.path.join(cwd, url.split("/")[-1].split("?")[0] or "download")
    else:
        dest = _resolve(dest, cwd)
    if shutil.which("wget"):
        cmd = ["wget", "-q", "-O", dest, url]
    elif shutil.which("curl"):
        cmd = ["curl", "-sL", "-o", dest, url]
    else:
        return {"ok": False, "error": "Neither wget nor curl found"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip() or r.stdout.strip()}
        size = os.path.getsize(dest)
        return {"ok": True, "path": dest, "size_bytes": size, "url": url}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Download timed out after 120s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_dns_lookup(host: str, cwd: str) -> dict:
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
        ips = list({i[4][0] for i in infos})
        return {"ok": True, "host": host, "addresses": ips}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_ping(host: str, cwd: str) -> dict:
    if not shutil.which("ping"):
        return {"ok": False, "host": host, "error": "ping not found"}
    try:
        r = subprocess.run(["ping", "-c", "4", "-W", "2", host], capture_output=True, text=True)
    except Exception as e:
        return {"ok": False, "host": host, "error": str(e)}
    output = (r.stdout + r.stderr).strip()
    lines = output.splitlines()
    summary = lines[-1] if lines else ""
    result = {"ok": r.returncode == 0, "host": host, "output": output[-1000:],
              "summary": summary, "returncode": r.returncode}
    if r.returncode != 0:
        result["error"] = summary or f"ping exited with status {r.returncode}"
    return result


def tool_port_scan(cwd: str, start: int = 1, end: int = 1024) -> dict:
    import socket
    open_ports = []
    for port in range(start, end + 1):
        try:
            s = socket.socket()
            s.settimeout(0.05)
            s.connect(("127.0.0.1", port))
            s.close()
            open_ports.append(port)
        except Exception:
            pass
    return {"ok": True, "open_ports": open_ports, "scanned": end - start + 1}


def tool_docker_ps(cwd: str, all: bool = False) -> dict:
    cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"]
    if all:
        cmd.append("-a")
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker not found"}
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()}
    containers = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            containers.append({"id": parts[0], "name": parts[1], "image": parts[2],
                               "status": parts[3], "ports": parts[4] if len(parts) > 4 else ""})
    return {"ok": True, "containers": containers, "count": len(containers)}


def tool_docker_exec(container: str, command: str, cwd: str) -> dict:
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker not found"}
    if not container:
        return {"ok": False, "error": "container required"}
    if not command:
        return {"ok": False, "error": "command required"}
    try:
        r = subprocess.run(["docker", "exec", container, "sh", "-c", command],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "docker exec timed out after 30s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    result = {"ok": r.returncode == 0, "stdout": r.stdout[-2000:], "stderr": r.stderr[-500:],
              "returncode": r.returncode}
    if r.returncode != 0:
        result["error"] = (r.stderr or r.stdout).strip() or f"docker exec exited with status {r.returncode}"
    return result


def tool_docker_logs(container: str, cwd: str, lines: int = 50) -> dict:
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker not found"}
    r = subprocess.run(["docker", "logs", "--tail", str(lines), container],
                       capture_output=True, text=True)
    output = (r.stdout + r.stderr)[-3000:]
    result = {"ok": r.returncode == 0, "output": output, "returncode": r.returncode}
    if r.returncode != 0:
        result["error"] = output.strip() or f"docker logs exited with status {r.returncode}"
    return result


def tool_docker_stop(container: str, cwd: str, remove: bool = False) -> dict:
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker not found"}
    if not container:
        return {"ok": False, "error": "container required"}
    try:
        r = subprocess.run(["docker", "stop", container], capture_output=True, text=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if r.returncode != 0:
        return {"ok": False, "container": container,
                "error": (r.stderr or r.stdout).strip() or f"docker stop exited with status {r.returncode}",
                "returncode": r.returncode}
    if remove:
        try:
            rm = subprocess.run(["docker", "rm", container], capture_output=True, text=True)
        except Exception as e:
            return {"ok": False, "container": container, "stopped": True, "removed": False, "error": str(e)}
        if rm.returncode != 0:
            return {"ok": False, "container": container, "stopped": True, "removed": False,
                    "error": (rm.stderr or rm.stdout).strip() or f"docker rm exited with status {rm.returncode}",
                    "returncode": rm.returncode}
    return {"ok": True, "container": container, "stopped": True, "removed": remove}


def tool_clipboard_get(cwd: str) -> dict:
    for cmd in (["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"],
                ["pbpaste"]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return {"ok": False, "error": (r.stderr or r.stdout).strip() or f"{cmd[0]} exited with status {r.returncode}"}
            return {"ok": True, "text": r.stdout}
    return {"ok": False, "error": "No clipboard tool found (install xclip or xsel)"}


def tool_clipboard_set(text: str, cwd: str) -> dict:
    for cmd in (["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
                ["pbcopy"]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, input=text, capture_output=True, text=True)
            result = {"ok": r.returncode == 0}
            if r.returncode != 0:
                result["error"] = (r.stderr or r.stdout).strip() or f"{cmd[0]} exited with status {r.returncode}"
            return result
    return {"ok": False, "error": "No clipboard tool found (install xclip or xsel)"}


def tool_screenshot(cwd: str, path: str = "", delay: int = 0) -> dict:
    dest = _resolve(path, cwd) if path else "/tmp/screenshot.png"
    if shutil.which("scrot"):
        cmd = ["scrot", f"-d{delay}", dest]
    elif shutil.which("gnome-screenshot"):
        cmd = ["gnome-screenshot", f"--delay={delay}", "-f", dest]
    elif shutil.which("import"):
        cmd = ["import", "-window", "root", dest]
    else:
        return {"ok": False, "error": "No screenshot tool found (install scrot or gnome-screenshot)"}
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()}
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return {"ok": True, "path": dest, "size_bytes": size}


def tool_head_file(path: str, cwd: str, lines: int = 10) -> dict:
    full = _resolve(path, cwd)
    try:
        with open(full, errors="replace") as f:
            content = "".join(f.readline() for _ in range(lines))
        return {"ok": True, "path": full, "content": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_tail_file(path: str, cwd: str, lines: int = 10) -> dict:
    full = _resolve(path, cwd)
    try:
        with open(full, errors="replace") as f:
            all_lines = f.readlines()
        content = "".join(all_lines[-lines:])
        return {"ok": True, "path": full, "content": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_json_query(query: str, cwd: str, path: str = "", json_str: str = "") -> dict:
    if not shutil.which("jq"):
        return {"ok": False, "error": "jq not installed — run: sudo apt install jq"}
    if path:
        full = _resolve(path, cwd)
        r = subprocess.run(["jq", query, full], capture_output=True, text=True)
    else:
        r = subprocess.run(["jq", query], input=json_str, capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()}
    try:
        import json as _json
        return {"ok": True, "result": _json.loads(r.stdout)}
    except Exception:
        return {"ok": True, "result": r.stdout.strip()}


def tool_base64_encode(cwd: str, text: str = "", path: str = "") -> dict:
    import base64 as _b64
    try:
        if path:
            full = Path(_resolve(path, cwd))
            if not full.exists():
                return {"ok": False, "error": f"File not found: {full}"}
            if not full.is_file():
                return {"ok": False, "error": f"Not a file: {full}"}
            data = full.read_bytes()
        else:
            data = text.encode()
        return {"ok": True, "encoded": _b64.b64encode(data).decode()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_base64_decode(text: str, cwd: str) -> dict:
    import base64 as _b64
    try:
        compact = re.sub(r"\s+", "", text)
        decoded = _b64.b64decode(compact, validate=True)
        try:
            return {"ok": True, "text": decoded.decode(), "bytes": len(decoded)}
        except Exception:
            return {"ok": True, "text": f"<binary, {len(decoded)} bytes>", "bytes": len(decoded)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_uuid_gen(cwd: str, count: int = 1) -> dict:
    import uuid
    ids = [str(uuid.uuid4()) for _ in range(min(count, 100))]
    return {"ok": True, "uuids": ids}


def tool_math_eval(expression: str, cwd: str) -> dict:
    try:
        import sympy
        result = sympy.sympify(expression)
        evaluated = sympy.simplify(result)
        return {"ok": True, "result": str(evaluated), "numeric": float(evaluated.evalf()) if evaluated.is_number else None}
    except ImportError:
        try:
            import ast, operator

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            def _eval(node):
                if isinstance(node, ast.Expression):
                    return _eval(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.BinOp) and type(node.op) in ops:
                    return ops[type(node.op)](_eval(node.left), _eval(node.right))
                if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                    return ops[type(node.op)](_eval(node.operand))
                raise ValueError("Only numeric arithmetic expressions are supported without sympy")

            result = _eval(ast.parse(expression, mode="eval"))
            return {"ok": True, "result": str(result)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_size_report(cwd: str, path: str = "", ext: str = "") -> dict:
    target = Path(_resolve(path, cwd)) if path else Path(cwd).resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path not found: {target}"}
    pattern = f"*{ext}" if ext else "*"
    if target.is_file():
        files = [target] if fnmatch.fnmatch(target.name, pattern) else []
        base = target.parent
    else:
        files = list(target.rglob(pattern))
        base = target
    report = []
    total = 0
    for f in sorted(files):
        if not f.is_file() or any(p.startswith(".") for p in f.parts):
            continue
        try:
            lines = len(f.read_text(errors="replace").splitlines())
            total += lines
            report.append({"file": str(f.relative_to(base)), "lines": lines})
        except Exception:
            continue
    report.sort(key=lambda x: x["lines"], reverse=True)
    return {"ok": True, "files": report[:50], "total_lines": total, "file_count": len(report)}


def tool_watch_file(path: str, cwd: str, lines: int = 20, timeout: float = 3.0) -> dict:
    import time as _time
    full = _resolve(path, cwd)
    try:
        size = os.path.getsize(full)
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            new_size = os.path.getsize(full)
            if new_size != size:
                break
            _time.sleep(0.2)
        with open(full, errors="replace") as f:
            all_lines = f.readlines()
        return {"ok": True, "path": full, "content": "".join(all_lines[-lines:])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


_VISION_MODELS = ["moondream", "llava", "llava-phi3", "llava-llama3", "minicpm-v", "bakllava"]

_MODE_PROMPTS = {
    "describe": "Describe this image in detail. Include objects, colors, layout, text, and any notable features.",
    "ocr":      "Extract and return ALL text visible in this image exactly as written. Format as plain text.",
    "caption":  "Write a single short caption for this image (one sentence, under 20 words).",
    "objects":  "List every distinct object visible in this image as a bullet list. Be specific.",
    "code":     "Transcribe all code or terminal text in this image exactly, preserving indentation and formatting.",
    "emotions": "Describe the mood, emotions, and atmosphere conveyed by this image.",
    "compare":  "Compare these two images. Describe similarities and differences in detail.",
    "qa":       None,
}

_IMAGE_CACHE: dict[str, Any] = {}  # sha256 hex → structured vision result


def _vision_installed(ollama_host: str) -> str:
    """Return name of best installed vision model, or ''."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as r:
            installed = {m["name"].split(":")[0] for m in json.loads(r.read()).get("models", [])}
        for c in _VISION_MODELS:
            if c in installed:
                return c
    except Exception:
        pass
    return ""


def _pull_vision_model(model: str, ollama_host: str) -> dict:
    import urllib.request
    payload = json.dumps({"name": model, "stream": False}).encode()
    try:
        req = urllib.request.Request(f"{ollama_host}/api/pull", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            return {"ok": True, "status": json.loads(r.read()).get("status", "done")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _resize_image(img_bytes: bytes, max_px: int = 1024) -> bytes:
    """Resize image so longest side ≤ max_px. Returns original bytes if PIL unavailable."""
    try:
        from PIL import Image as _Img
        import io
        img = _Img.open(io.BytesIO(img_bytes))
        fmt = img.format or "JPEG"
        w, h = img.size
        if max(w, h) <= max_px:
            return img_bytes
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _Img.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return img_bytes


def _image_meta(img_bytes: bytes, label: str) -> dict:
    """Extract dimensions, format, EXIF GPS/date if PIL available."""
    meta: dict = {"size_bytes": len(img_bytes)}
    try:
        from PIL import Image as _Img
        import io
        img = _Img.open(io.BytesIO(img_bytes))
        meta["width"], meta["height"] = img.size
        meta["format"] = img.format
        meta["mode"] = img.mode
        # EXIF
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            exif_raw = img._getexif() or {}
            exif: dict = {}
            for tag_id, val in exif_raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime", "Make", "Model", "Software"):
                    exif[tag] = str(val)
                elif tag == "GPSInfo":
                    gps: dict = {}
                    for k, v in val.items():
                        gps[GPSTAGS.get(k, k)] = str(v)
                    exif["GPS"] = gps
            if exif:
                meta["exif"] = exif
        except Exception:
            pass
    except Exception:
        pass
    return meta


def _fetch_bytes(path: str, url: str, cwd: str) -> tuple[bytes, str]:
    """Load image bytes from local path or URL. Returns (bytes, label)."""
    import urllib.request
    if path:
        full = _resolve(path, cwd)
        return Path(full).read_bytes(), full
    if url:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read(), url
    raise ValueError("Provide 'path' or 'url'")


def _ollama_vision(model: str, images_b64: list[str], prompt: str,
                   ollama_host: str, timeout: int | None) -> tuple[str, dict]:
    """Call Ollama vision endpoint. Returns (text, stats)."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": True,
        "options": {"temperature": 0.15, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        f"{ollama_host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    stats: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                stats = {
                    "prompt_tokens": chunk.get("prompt_eval_count"),
                    "completion_tokens": chunk.get("eval_count"),
                    "duration_s": round(chunk.get("total_duration", 0) / 1e9, 2),
                }
    return "".join(parts).strip(), stats


def _postprocess(text: str, mode: str) -> tuple[str, dict]:
    """Clean up text and, for structured modes, also return parsed data."""
    extra: dict = {}
    if mode == "ocr":
        # strip model preamble like "The text in the image reads:"
        for prefix in ("the text", "this image contains", "i can see", "text visible"):
            if text.lower().startswith(prefix):
                text = re.sub(r"^[^:]+:\s*", "", text, flags=re.IGNORECASE)
                break
    elif mode == "objects":
        items = [re.sub(r"^[-*•\d.)\s]+", "", l).strip() for l in text.splitlines() if l.strip()]
        extra["object_list"] = [i for i in items if i]
    elif mode == "caption":
        text = text.split("\n")[0].strip().rstrip(".")
    return text, extra


def _describe_one(label: str, img_bytes: bytes, prompt: str, mode: str,
                  model: str, ollama_host: str, timeout: int,
                  max_px: int, use_cache: bool) -> dict:
    import hashlib, base64 as _b64
    resized = _resize_image(img_bytes, max_px)
    meta = _image_meta(resized, label)
    cache_key = hashlib.sha256(resized + prompt.encode() + model.encode()).hexdigest()
    if use_cache and cache_key in _IMAGE_CACHE:
        cached = _IMAGE_CACHE[cache_key]
        if isinstance(cached, dict):
            text = cached.get("text", "")
            stats = cached.get("stats", {})
            cached_meta = cached.get("meta", meta)
        else:
            text = str(cached)
            stats = {}
            cached_meta = meta
        text, extra = _postprocess(text, mode)
        return {"image": label, "ok": True, "description": text,
                "meta": cached_meta, "stats": stats, "cached": True, **extra}
    img_b64 = _b64.b64encode(resized).decode()
    try:
        text, stats = _ollama_vision(model, [img_b64], prompt, ollama_host, timeout)
        if len(_IMAGE_CACHE) >= 128:
            _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
        _IMAGE_CACHE[cache_key] = {"text": text, "meta": meta, "stats": stats}
        text, extra = _postprocess(text, mode)
        return {"image": label, "ok": True, "description": text,
                "meta": meta, "stats": stats, "cached": False, **extra}
    except Exception as e:
        return {"image": label, "ok": False, "error": str(e), "meta": meta}


def tool_describe_image(cwd: str, path: str = "", url: str = "", paths: list = None,
                        compare_path: str = "", compare_url: str = "",
                        prompt: str = "", mode: str = "describe",
                        model: str = "", auto_pull: bool = True,
                        max_px: int = 1024, use_cache: bool = True,
                        save_to: str = "", timeout: int | None = None,
                        ollama_host: str = "http://localhost:11434") -> dict:
    """
    Describe / analyse images with a local Ollama vision model.

    Modes: describe | ocr | caption | objects | code | emotions | compare | qa
    Batch: pass paths=[...] for multiple files (processed in parallel, max 2 at once).
    Compare: set compare_path/compare_url to diff two images side-by-side.
    auto_pull: pull moondream automatically if no vision model is installed.
    max_px: resize longest side to this before sending (saves VRAM, default 1024).
    use_cache: skip re-processing identical images (keyed by content hash).
    save_to: write all descriptions to a markdown file.
    """
    import urllib.error
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ollama_host = (ollama_host or "http://localhost:11434").rstrip("/")

    # ── Model resolution ─────────────────────────────────────────────────────
    if not model:
        model = _vision_installed(ollama_host)
        if not model:
            if not auto_pull:
                return {"ok": False, "error": "No vision model found. Run: ollama pull moondream"}
            pull = _pull_vision_model("moondream", ollama_host)
            if not pull["ok"]:
                return {"ok": False, "error": f"Auto-pull failed: {pull['error']}. Run: ollama pull moondream"}
            model = "moondream"

    # ── Prompt ───────────────────────────────────────────────────────────────
    effective_prompt = prompt or _MODE_PROMPTS.get(mode, _MODE_PROMPTS["describe"]) or "Describe this image."

    # ── Compare mode: two images in one message ──────────────────────────────
    if mode == "compare" or compare_path or compare_url:
        mode = "compare"
        effective_prompt = prompt or _MODE_PROMPTS["compare"]
        try:
            bytes_a, label_a = _fetch_bytes(path, url, cwd)
            bytes_b, label_b = _fetch_bytes(compare_path, compare_url, cwd)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        import base64 as _b64
        b64_a = _b64.b64encode(_resize_image(bytes_a, max_px)).decode()
        b64_b = _b64.b64encode(_resize_image(bytes_b, max_px)).decode()
        try:
            text, stats = _ollama_vision(model, [b64_a, b64_b], effective_prompt, ollama_host, timeout)
            text, extra = _postprocess(text, mode)
            return {"ok": True, "model": model, "mode": "compare",
                    "images": [label_a, label_b], "description": text,
                    "stats": stats, **extra}
        except urllib.error.URLError as e:
            return {"ok": False, "error": f"Ollama unreachable: {e}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Collect sources ──────────────────────────────────────────────────────
    raw_sources: list[tuple[str, str, bytes]] = []  # (path, url, bytes)
    if paths:
        for p in paths:
            try:
                b, lbl = _fetch_bytes(p, "", cwd)
                raw_sources.append((p, "", b))
            except Exception as e:
                raw_sources.append((p, "", b""))
    elif path or url:
        try:
            b, lbl = _fetch_bytes(path, url, cwd)
            raw_sources.append((path, url, b))
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        return {"ok": False, "error": "Provide 'path', 'url', or 'paths'"}

    # ── Process (parallel for batch, serial for single) ──────────────────────
    def _job(src_path, src_url, img_bytes):
        label = _resolve(src_path, cwd) if src_path else src_url
        if not img_bytes:
            return {"image": label, "ok": False, "error": "Could not load image"}
        return _describe_one(label, img_bytes, effective_prompt, mode,
                             model, ollama_host, timeout, max_px, use_cache)

    results = []
    if len(raw_sources) == 1:
        sp, su, sb = raw_sources[0]
        try:
            results.append(_job(sp, su, sb))
        except urllib.error.URLError as e:
            return {"ok": False, "error": f"Ollama unreachable: {e}"}
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(_job, sp, su, sb): (sp, su) for sp, su, sb in raw_sources}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    sp, su = futs[fut]
                    results.append({"image": sp or su, "ok": False, "error": str(e)})

    # ── Save output ──────────────────────────────────────────────────────────
    if save_to:
        out_path = _resolve(save_to, cwd)
        sections = []
        for r in results:
            hdr = f"## {r['image']}"
            body = r.get("description", r.get("error", ""))
            if r.get("meta"):
                m = r["meta"]
                dims = f"{m.get('width','?')}×{m.get('height','?')}" if "width" in m else ""
                hdr += f"  \n*{dims} · {m.get('format','')} · {m.get('size_bytes',0):,} bytes*"
            sections.append(f"{hdr}\n\n{body}")
        Path(out_path).write_text("\n\n---\n\n".join(sections))

    # ── Return ───────────────────────────────────────────────────────────────
    if len(results) == 1:
        r = results[0]
        out = {"ok": r["ok"], "model": model, "mode": mode}
        out.update(r)
        if save_to:
            out["saved_to"] = _resolve(save_to, cwd)
        return out

    return {"ok": True, "model": model, "mode": mode, "count": len(results),
            "results": results,
            **({"saved_to": _resolve(save_to, cwd)} if save_to else {})}


def tool_sub_ai(prompt: str, cwd: str, model: str = "qwen2:0.5b",
                system: str = "", timeout: int = 60) -> dict:
    """Run a sub-query against a lightweight local Ollama model and return the response."""
    import urllib.request, urllib.error
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        content = data.get("message", {}).get("content", "")
        return {"ok": True, "model": model, "response": content,
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count")}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Ollama unreachable: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_git_diff(cwd: str, ref_a: str = "", ref_b: str = "", path: str = "", staged: bool = False) -> dict:
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if ref_a:
        cmd.append(ref_a)
    if ref_b:
        cmd.append(ref_b)
    if path:
        cmd += ["--", _resolve(path, cwd)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    diff = r.stdout[:6000]
    return {"ok": r.returncode == 0, "diff": diff, "truncated": len(r.stdout) > 6000,
            "stderr": r.stderr.strip() or None}


def tool_git_branch(action: str, cwd: str, name: str = "", new_name: str = "", force: bool = False) -> dict:
    if action == "list":
        r = subprocess.run(["git", "branch", "-a", "--format=%(refname:short) %(objectname:short) %(upstream:short)"],
                           capture_output=True, text=True, cwd=cwd)
        branches = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return {"ok": True, "branches": branches}
    elif action == "create":
        if not name:
            return {"ok": False, "error": "name required"}
        r = subprocess.run(["git", "checkout", "-b", name], capture_output=True, text=True, cwd=cwd)
    elif action == "delete":
        if not name:
            return {"ok": False, "error": "name required"}
        flag = "-D" if force else "-d"
        r = subprocess.run(["git", "branch", flag, name], capture_output=True, text=True, cwd=cwd)
    elif action == "rename":
        if not name or not new_name:
            return {"ok": False, "error": "name and new_name required"}
        r = subprocess.run(["git", "branch", "-m", name, new_name], capture_output=True, text=True, cwd=cwd)
    elif action == "checkout":
        if not name:
            return {"ok": False, "error": "name required"}
        r = subprocess.run(["git", "checkout", name], capture_output=True, text=True, cwd=cwd)
    else:
        return {"ok": False, "error": f"Unknown action: {action}"}
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr).strip()}


def tool_archive_create(output: str, sources: list, cwd: str, format: str = "") -> dict:
    import zipfile, tarfile
    out = _resolve(output, cwd)
    if not format:
        if out.endswith(".zip"):
            format = "zip"
        elif out.endswith(".tar.gz") or out.endswith(".tgz"):
            format = "tar.gz"
        elif out.endswith(".tar.bz2"):
            format = "tar.bz2"
        else:
            format = "zip"
    try:
        if format == "zip":
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for src in sources:
                    full = _resolve(src, cwd)
                    p = Path(full)
                    if p.is_dir():
                        for f in p.rglob("*"):
                            if f.is_file():
                                zf.write(f, f.relative_to(p.parent))
                    else:
                        zf.write(full, p.name)
        else:
            mode = "w:gz" if format == "tar.gz" else "w:bz2"
            with tarfile.open(out, mode) as tf:
                for src in sources:
                    full = _resolve(src, cwd)
                    tf.add(full, arcname=Path(full).name)
        return {"ok": True, "path": out, "format": format, "size_bytes": os.path.getsize(out)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_archive_extract(path: str, cwd: str, dest: str = "") -> dict:
    import zipfile, tarfile
    full = _resolve(path, cwd)
    out = _resolve(dest, cwd) if dest else cwd

    def _safe_target(member_name: str) -> str:
        base = Path(out).resolve()
        target = (base / member_name).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Archive member escapes destination: {member_name}")
        return str(target)

    try:
        if zipfile.is_zipfile(full):
            with zipfile.ZipFile(full) as zf:
                for info in zf.infolist():
                    _safe_target(info.filename)
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        raise ValueError(f"Refusing to extract symlink: {info.filename}")
                zf.extractall(out)
                names = zf.namelist()
        elif tarfile.is_tarfile(full):
            with tarfile.open(full) as tf:
                for member in tf.getmembers():
                    target = Path(_safe_target(member.name))
                    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                        raise ValueError(f"Refusing to extract special file: {member.name}")
                    if member.issym() or member.islnk():
                        link_target = (target.parent / member.linkname).resolve()
                        try:
                            link_target.relative_to(Path(out).resolve())
                        except ValueError:
                            raise ValueError(f"Archive link escapes destination: {member.name}")
                tf.extractall(out)
                names = tf.getnames()
        else:
            return {"ok": False, "error": "Unsupported archive format (need .zip or .tar.*)"}
        return {"ok": True, "dest": out, "files_extracted": len(names)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_archive_list(path: str, cwd: str) -> dict:
    import zipfile, tarfile
    full = _resolve(path, cwd)
    try:
        if zipfile.is_zipfile(full):
            with zipfile.ZipFile(full) as zf:
                entries = [{"name": i.filename, "size": i.file_size, "compressed": i.compress_size}
                           for i in zf.infolist()]
        elif tarfile.is_tarfile(full):
            with tarfile.open(full) as tf:
                entries = [{"name": m.name, "size": m.size, "type": "dir" if m.isdir() else "file"}
                           for m in tf.getmembers()]
        else:
            return {"ok": False, "error": "Unsupported archive format"}
        return {"ok": True, "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_yaml_query(path: str, cwd: str, key: str = "") -> dict:
    full = _resolve(path, cwd)
    try:
        import yaml as _yaml
    except ImportError:
        r = subprocess.run(["python3", "-m", "pip", "install", "--quiet", "pyyaml"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"ok": False, "error": "pyyaml not installed and auto-install failed"}
        import yaml as _yaml
    try:
        with open(full) as f:
            data = _yaml.safe_load(f)
        if key:
            for part in key.split("."):
                if isinstance(data, dict):
                    data = data.get(part)
                else:
                    return {"ok": False, "error": f"Key '{part}' not found at this level"}
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_env_get(cwd: str, name: str = "") -> dict:
    if name:
        val = os.environ.get(name)
        return {"ok": True, "name": name, "value": val, "set": val is not None}
    safe = {k: v for k, v in os.environ.items()
            if not any(x in k.lower() for x in ("secret", "password", "token", "key", "pass"))}
    return {"ok": True, "variables": safe, "count": len(safe), "redacted_count": len(os.environ) - len(safe)}


def tool_open_file(path: str, cwd: str) -> dict:
    full = _resolve(path, cwd) if not path.startswith("http") else path
    opener = "xdg-open" if shutil.which("xdg-open") else "open"
    if not shutil.which(opener):
        return {"ok": False, "error": f"{opener} not found"}
    r = subprocess.Popen([opener, full], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True, "opened": full, "pid": r.pid}


def tool_regex_test(pattern: str, text: str, cwd: str, flags: list = None) -> dict:
    import re as _re
    flag_val = 0
    for f in (flags or []):
        if f.lower() == "ignorecase": flag_val |= _re.IGNORECASE
        elif f.lower() == "multiline": flag_val |= _re.MULTILINE
        elif f.lower() == "dotall": flag_val |= _re.DOTALL
    try:
        compiled = _re.compile(pattern, flag_val)
        matches = []
        for m in compiled.finditer(text):
            matches.append({"match": m.group(0), "start": m.start(), "end": m.end(),
                            "groups": list(m.groups()), "groupdict": m.groupdict()})
        return {"ok": True, "matches": matches, "count": len(matches), "matched": len(matches) > 0}
    except _re.error as e:
        return {"ok": False, "error": f"Invalid regex: {e}"}


def tool_csv_query(path: str, cwd: str, filter_col: str = "", filter_val: str = "",
                   columns: list = None, limit: int = 100) -> dict:
    import csv
    full = _resolve(path, cwd)
    try:
        with open(full, newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if filter_col and row.get(filter_col) != filter_val:
                    continue
                if columns:
                    row = {k: row[k] for k in columns if k in row}
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
        return {"ok": True, "rows": rows, "count": len(rows), "columns": list(rows[0].keys()) if rows else []}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_image_info(path: str, cwd: str) -> dict:
    full = _resolve(path, cwd)
    size = os.path.getsize(full)
    try:
        from PIL import Image as _Image
        with _Image.open(full) as img:
            return {"ok": True, "path": full, "format": img.format, "mode": img.mode,
                    "width": img.width, "height": img.height, "size_bytes": size}
    except ImportError:
        pass
    # Fallback: use file command
    r = subprocess.run(["file", full], capture_output=True, text=True)
    return {"ok": True, "path": full, "size_bytes": size, "file_info": r.stdout.strip()}


def tool_network_info(cwd: str) -> dict:
    r = subprocess.run(["ip", "-j", "addr"], capture_output=True, text=True)
    if r.returncode == 0:
        try:
            import json as _j
            ifaces = _j.loads(r.stdout)
            result = []
            for iface in ifaces:
                addrs = [{"family": a.get("family"), "addr": a.get("local"), "prefix": a.get("prefixlen")}
                         for a in iface.get("addr_info", [])]
                result.append({"name": iface.get("ifname"), "state": iface.get("operstate"),
                                "mac": iface.get("address"), "addresses": addrs})
            return {"ok": True, "interfaces": result}
        except Exception:
            pass
    # Fallback
    r2 = subprocess.run(["ifconfig"], capture_output=True, text=True)
    if r2.returncode == 0:
        return {"ok": True, "output": r2.stdout[:3000]}
    return {"ok": False, "error": "Neither 'ip' nor 'ifconfig' available"}


def tool_uptime_info(cwd: str) -> dict:
    r = subprocess.run(["uptime", "-p"], capture_output=True, text=True)
    uptime_str = r.stdout.strip()
    load = os.getloadavg()
    mem_r = subprocess.run(["free", "-h"], capture_output=True, text=True)
    return {"ok": True, "uptime": uptime_str, "load_avg": {"1m": load[0], "5m": load[1], "15m": load[2]},
            "memory": mem_r.stdout.strip()}


def tool_ssl_check(host: str, cwd: str, port: int = 443) -> dict:
    import ssl, socket, datetime
    ctx = ssl.create_default_context()
    try:
        with ctx.wrap_socket(socket.create_connection((host, port), timeout=10), server_hostname=host) as s:
            cert = s.getpeercert()
        not_after = datetime.datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
        days_left = (not_after - datetime.datetime.utcnow()).days
        sans = [v for t, v in cert.get("subjectAltName", [])]
        subject = dict(x[0] for x in cert.get("subject", []))
        issuer = dict(x[0] for x in cert.get("issuer", []))
        return {"ok": True, "host": host, "port": port, "subject": subject, "issuer": issuer,
                "not_after": str(not_after), "days_until_expiry": days_left, "sans": sans[:10],
                "expired": days_left < 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_whois_lookup(target: str, cwd: str) -> dict:
    if shutil.which("whois"):
        r = subprocess.run(["whois", target], capture_output=True, text=True, timeout=15)
        return {"ok": r.returncode == 0, "output": r.stdout[:3000]}
    try:
        import socket
        s = socket.socket()
        s.settimeout(10)
        s.connect(("whois.iana.org", 43))
        s.sendall((target + "\r\n").encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        return {"ok": True, "output": data.decode(errors="replace")[:3000]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_top_procs(cwd: str, sort_by: str = "cpu", limit: int = 10) -> dict:
    sort_flag = "--sort=-%cpu" if sort_by == "cpu" else "--sort=-%mem"
    r = subprocess.run(
        ["ps", "aux", sort_flag],
        capture_output=True, text=True,
    )
    lines = r.stdout.splitlines()
    header = lines[0] if lines else ""
    procs = []
    for line in lines[1:limit + 1]:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            procs.append({"user": parts[0], "pid": parts[1], "cpu": parts[2],
                          "mem": parts[3], "cmd": parts[10]})
    return {"ok": True, "sort_by": sort_by, "processes": procs}


def tool_cron_list(cwd: str, all_users: bool = False) -> dict:
    results = {}
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    results["user"] = r.stdout.strip() if r.returncode == 0 else "(no crontab)"
    if all_users:
        for p in Path("/etc/cron.d").iterdir() if Path("/etc/cron.d").exists() else []:
            try:
                results[f"/etc/cron.d/{p.name}"] = p.read_text()[:500]
            except Exception:
                pass
    return {"ok": True, "crontabs": results}


def tool_coverage_run(cwd: str, path: str = "", runner: str = "") -> dict:
    target = _resolve(path, cwd) if path else cwd
    if not runner:
        if (Path(target) / "package.json").exists():
            runner = "jest"
        else:
            runner = "pytest"
    if runner == "pytest":
        if not shutil.which("coverage"):
            r0 = subprocess.run(["python3", "-m", "pip", "install", "--quiet", "coverage"],
                                 capture_output=True, text=True)
        cmd = ["python3", "-m", "coverage", "run", "-m", "pytest", "--tb=no", "-q"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=target, timeout=120)
        r2 = subprocess.run(["python3", "-m", "coverage", "report", "--skip-empty"],
                             capture_output=True, text=True, cwd=target)
        return {"ok": r.returncode == 0, "runner": "coverage+pytest",
                "test_output": r.stdout[-1000:], "coverage_report": r2.stdout[-2000:]}
    elif runner == "jest":
        cmd = ["npx", "jest", "--coverage", "--ci", "--coverageReporters=text"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=target, timeout=120)
        return {"ok": r.returncode == 0, "runner": "jest", "output": (r.stdout + r.stderr)[-3000:]}
    return {"ok": False, "error": f"Unknown runner: {runner}"}


def tool_ast_analyze(path: str, cwd: str) -> dict:
    import ast as _ast
    full = _resolve(path, cwd)
    try:
        src = Path(full).read_text(errors="replace")
        tree = _ast.parse(src, filename=full)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    classes, functions, imports, variables = [], [], [], []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            methods = [n.name for n in _ast.walk(node) if isinstance(n, _ast.FunctionDef)]
            classes.append({"name": node.name, "line": node.lineno, "methods": methods})
        elif isinstance(node, _ast.FunctionDef) and isinstance(getattr(node, "parent", None), _ast.Module):
            functions.append({"name": node.name, "line": node.lineno,
                              "args": [a.arg for a in node.args.args]})
        elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
            if isinstance(node, _ast.Import):
                imports += [a.name for a in node.names]
            else:
                imports.append(f"from {node.module or '.'} import ...")
    # top-level functions (walk doesn't track parents, use direct children)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef):
            functions.append({"name": node.name, "line": node.lineno,
                              "args": [a.arg for a in node.args.args]})
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    variables.append({"name": t.id, "line": node.lineno})
    seen = set()
    functions = [f for f in functions if f["name"] not in seen and not seen.add(f["name"])]
    return {"ok": True, "path": full, "classes": classes, "functions": functions,
            "imports": list(dict.fromkeys(imports))[:30], "top_level_vars": variables}


def tool_template_render(cwd: str, template: str = "", template_file: str = "",
                          variables: dict = None) -> dict:
    variables = variables or {}
    try:
        from jinja2 import Environment, FileSystemLoader, BaseLoader
    except ImportError:
        r = subprocess.run(["python3", "-m", "pip", "install", "--quiet", "jinja2"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"ok": False, "error": "jinja2 not installed and auto-install failed"}
        from jinja2 import Environment, FileSystemLoader, BaseLoader
    try:
        if template_file:
            full = _resolve(template_file, cwd)
            env = Environment(loader=FileSystemLoader(str(Path(full).parent)))
            tmpl = env.get_template(Path(full).name)
        else:
            env = Environment(loader=BaseLoader())
            tmpl = env.from_string(template)
        rendered = tmpl.render(**variables)
        return {"ok": True, "rendered": rendered}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_create_pdf(path: str, content: str, cwd: str, title: str = "", author: str = "") -> dict:
    try:
        from fpdf import FPDF
    except ImportError:
        # Try to install fpdf2 automatically
        result = subprocess.run(
            ["python3", "-m", "pip", "install", "--quiet", "fpdf2"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {"ok": False, "error": "fpdf2 not installed and auto-install failed. Run: pip install fpdf2"}
        try:
            from fpdf import FPDF
        except ImportError:
            return {"ok": False, "error": "fpdf2 installed but import still failed — try restarting."}

    abs_path = path if os.path.isabs(path) else os.path.join(cwd, path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    if title:
        pdf.set_title(title)
    if author:
        pdf.set_author(author)
    pdf.add_page()

    in_code = False
    code_buf: list[str] = []

    def flush_code():
        if not code_buf:
            return
        pdf.set_font("Courier", size=9)
        pdf.set_fill_color(240, 240, 240)
        block = "\n".join(code_buf)
        pdf.multi_cell(0, 5, block, fill=True)
        pdf.ln(2)
        code_buf.clear()

    for line in content.splitlines():
        if line.startswith("```"):
            if in_code:
                flush_code()
            in_code = not in_code
            continue

        if in_code:
            code_buf.append(line)
            continue

        stripped = line.rstrip()

        if stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, stripped[4:])
            pdf.ln(1)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, stripped[3:])
            pdf.ln(2)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 20)
            pdf.multi_cell(0, 10, stripped[2:])
            pdf.ln(3)
        elif stripped == "" or stripped == "---":
            pdf.ln(4)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, f"  • {stripped[2:]}")
        elif stripped[0].isdigit() and ". " in stripped[:4]:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, f"  {stripped}")
        else:
            # strip inline bold/italic markers for plain rendering
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            text = re.sub(r"\*(.+?)\*", r"\1", text)
            text = re.sub(r"`(.+?)`", r"\1", text)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, text)

    flush_code()  # trailing code block without closing ```

    pdf.output(abs_path)
    size = os.path.getsize(abs_path)
    return {"ok": True, "path": abs_path, "size_bytes": size}


def _load_dynamic_tools():
    """Load persisted dynamic tools from ~/.abby-der/dynamic_tools.py."""
    if not DYNAMIC_TOOLS_FILE.exists():
        return

    ns: dict[str, Any] = {
        "_DYNAMIC_REGISTRY": [],
        "os": os, "subprocess": subprocess, "shlex": shlex, "shutil": shutil,
        "json": json, "Path": Path, "re": re,
    }
    try:
        with open(DYNAMIC_TOOLS_FILE) as f:
            src = f.read()
        exec(compile(src, str(DYNAMIC_TOOLS_FILE), "exec"), ns)

        for entry in ns.get("_DYNAMIC_REGISTRY", []):
            n = entry["name"]
            fn = ns.get(f"tool_{n}")
            if not fn:
                continue
            if n in _dynamic_fns or any(s["function"]["name"] == n for s in TOOL_SCHEMAS):
                continue
            _dynamic_fns[n] = fn
            _dynamic_schemas.append(entry["schema"])
            TOOL_SCHEMAS.append(entry["schema"])
            AUTO_APPROVE_MAP[n] = entry.get("auto_approve", "auto_approve_bash")
            TOOL_DESCRIPTIONS[n] = (n, "magenta")
    except Exception as e:
        import sys
        print(f"[abby-der] Warning: failed to load dynamic tools: {e}", file=sys.stderr)

# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(name: str, args: dict, cwd: str, cfg: Any = None) -> dict:
    if name == "read_file":
        return tool_read_file(args["path"], cwd, args.get("offset"), args.get("limit"))
    elif name == "write_file":
        return tool_write_file(args["path"], args["content"], cwd)
    elif name == "edit_file":
        return tool_edit_file(args["path"], args["old_string"], args["new_string"], cwd)
    elif name == "bash":
        return tool_bash(args["command"], cwd, args.get("timeout", 0))
    elif name == "list_dir":
        return tool_list_dir(args.get("path", "."), cwd)
    elif name == "grep":
        return tool_grep(args["pattern"], cwd, args.get("path", "."),
                         args.get("case_insensitive", False), args.get("file_pattern"))
    elif name == "find_files":
        return tool_find_files(args["pattern"], cwd, args.get("path", "."))
    elif name == "git":
        return tool_git(args["args"], cwd)
    elif name == "npm_dev":
        return tool_npm_dev(args["action"], cwd, args.get("script", "dev"), args.get("port"))
    elif name == "browse":
        return tool_browse(args["url"], cwd, args.get("raw", False), args.get("links_only", False))
    elif name == "keyboard":
        return tool_keyboard(args["action"], cwd, args.get("text"), args.get("keys"),
                             args.get("window"), args.get("delay_ms", 0))
    elif name == "add_tool":
        return tool_add_tool(
            args["name"], args["description"], args["parameters_schema"],
            args["python_body"], cwd, args.get("auto_approve", "bash"),
        )
    elif name == "package":
        return tool_package(
            args["action"], args.get("packages", []), cwd,
            args.get("manager", "auto"), args.get("flags"),
        )
    elif name == "http_request":
        return tool_http_request(args["method"], args["url"], cwd,
                                 args.get("headers"), args.get("body"), args.get("timeout", 15))
    elif name == "lint":
        return tool_lint(args["path"], cwd, args.get("linter", ""))
    elif name == "test_run":
        return tool_test_run(cwd, args.get("path", ""), args.get("runner", ""), args.get("args", ""))
    elif name == "sqlite_query":
        return tool_sqlite_query(args["db"], args["query"], cwd, args.get("params"))
    elif name == "secret_scan":
        return tool_secret_scan(cwd, args.get("path", ""), args.get("include", ""))
    elif name == "diff_files":
        return tool_diff_files(args["a"], args["b"], cwd, args.get("context", 3))
    elif name == "rename_file":
        return tool_rename_file(args["src"], args["dst"], cwd)
    elif name == "copy_file":
        return tool_copy_file(args["src"], args["dst"], cwd)
    elif name == "delete_file":
        return tool_delete_file(args["path"], cwd)
    elif name == "hash_file":
        return tool_hash_file(args["path"], cwd)
    elif name == "todo_scan":
        return tool_todo_scan(cwd, args.get("path", ""), args.get("tags"))
    elif name == "format_code":
        return tool_format_code(args["path"], cwd, args.get("formatter", ""))
    elif name == "port_kill":
        return tool_port_kill(args["port"], cwd)
    elif name == "notify":
        return tool_notify(args["title"], cwd, args.get("body", ""), args.get("icon", ""))
    elif name == "git_log":
        return tool_git_log(cwd, args.get("limit", 20), args.get("file", ""), args.get("branch", "HEAD"))
    elif name == "git_blame":
        return tool_git_blame(args["path"], cwd, args.get("start"), args.get("end"))
    elif name == "git_stash":
        return tool_git_stash(args["action"], cwd, args.get("message", ""), args.get("index", 0))
    elif name == "type_check":
        return tool_type_check(args["path"], cwd, args.get("checker", ""))
    elif name == "ps_list":
        return tool_ps_list(cwd, args.get("filter", ""))
    elif name == "kill_proc":
        return tool_kill_proc(args["target"], cwd, args.get("force", False))
    elif name == "disk_usage":
        return tool_disk_usage(cwd, args.get("path", ""))
    elif name == "service_ctl":
        return tool_service_ctl(args["action"], args["service"], cwd)
    elif name == "journalctl":
        return tool_journalctl(args["unit"], cwd, args.get("lines", 50), args.get("follow", False))
    elif name == "download_file":
        return tool_download_file(args["url"], cwd, args.get("dest", ""))
    elif name == "dns_lookup":
        return tool_dns_lookup(args["host"], cwd)
    elif name == "ping":
        return tool_ping(args["host"], cwd)
    elif name == "port_scan":
        return tool_port_scan(cwd, args.get("start", 1), args.get("end", 1024))
    elif name == "docker_ps":
        return tool_docker_ps(cwd, args.get("all", False))
    elif name == "docker_exec":
        return tool_docker_exec(args["container"], args["command"], cwd)
    elif name == "docker_logs":
        return tool_docker_logs(args["container"], cwd, args.get("lines", 50))
    elif name == "docker_stop":
        return tool_docker_stop(args["container"], cwd, args.get("remove", False))
    elif name == "clipboard_get":
        return tool_clipboard_get(cwd)
    elif name == "clipboard_set":
        return tool_clipboard_set(args["text"], cwd)
    elif name == "screenshot":
        return tool_screenshot(cwd, args.get("path", ""), args.get("delay", 0))
    elif name == "head_file":
        return tool_head_file(args["path"], cwd, args.get("lines", 10))
    elif name == "tail_file":
        return tool_tail_file(args["path"], cwd, args.get("lines", 10))
    elif name == "json_query":
        return tool_json_query(args["query"], cwd, args.get("path", ""), args.get("json", ""))
    elif name == "base64_encode":
        return tool_base64_encode(cwd, args.get("text", ""), args.get("path", ""))
    elif name == "base64_decode":
        return tool_base64_decode(args["text"], cwd)
    elif name == "uuid_gen":
        return tool_uuid_gen(cwd, args.get("count", 1))
    elif name == "math_eval":
        return tool_math_eval(args["expression"], cwd)
    elif name == "size_report":
        return tool_size_report(cwd, args.get("path", ""), args.get("ext", ""))
    elif name == "watch_file":
        return tool_watch_file(args["path"], cwd, args.get("lines", 20), args.get("timeout", 3.0))
    elif name == "git_diff":
        return tool_git_diff(cwd, args.get("ref_a", ""), args.get("ref_b", ""),
                             args.get("path", ""), args.get("staged", False))
    elif name == "git_branch":
        return tool_git_branch(args["action"], cwd, args.get("name", ""),
                               args.get("new_name", ""), args.get("force", False))
    elif name == "archive_create":
        return tool_archive_create(args["output"], args["sources"], cwd, args.get("format", ""))
    elif name == "archive_extract":
        return tool_archive_extract(args["path"], cwd, args.get("dest", ""))
    elif name == "archive_list":
        return tool_archive_list(args["path"], cwd)
    elif name == "yaml_query":
        return tool_yaml_query(args["path"], cwd, args.get("key", ""))
    elif name == "env_get":
        return tool_env_get(cwd, args.get("name", ""))
    elif name == "open_file":
        return tool_open_file(args["path"], cwd)
    elif name == "regex_test":
        return tool_regex_test(args["pattern"], args["text"], cwd, args.get("flags"))
    elif name == "csv_query":
        return tool_csv_query(args["path"], cwd, args.get("filter_col", ""),
                              args.get("filter_val", ""), args.get("columns"), args.get("limit", 100))
    elif name == "image_info":
        return tool_image_info(args["path"], cwd)
    elif name == "network_info":
        return tool_network_info(cwd)
    elif name == "uptime_info":
        return tool_uptime_info(cwd)
    elif name == "ssl_check":
        return tool_ssl_check(args["host"], cwd, args.get("port", 443))
    elif name == "whois_lookup":
        return tool_whois_lookup(args["target"], cwd)
    elif name == "top_procs":
        return tool_top_procs(cwd, args.get("sort_by", "cpu"), args.get("limit", 10))
    elif name == "cron_list":
        return tool_cron_list(cwd, args.get("all_users", False))
    elif name == "coverage_run":
        return tool_coverage_run(cwd, args.get("path", ""), args.get("runner", ""))
    elif name == "ast_analyze":
        return tool_ast_analyze(args["path"], cwd)
    elif name == "template_render":
        return tool_template_render(cwd, args.get("template", ""), args.get("template_file", ""),
                                    args.get("variables"))
    elif name == "describe_image":
        model = args.get("model", "") or getattr(cfg, "vision_model", "")
        max_px = args.get("max_px", None)
        if max_px in (None, ""):
            max_px = getattr(cfg, "vision_max_px", 1024)
        try:
            max_px = int(max_px)
            if max_px <= 0:
                raise ValueError("must be positive")
        except (TypeError, ValueError):
            return {"ok": False, "error": "max_px must be a positive integer"}
        use_cache = args.get("use_cache", None)
        if use_cache is None:
            use_cache = getattr(cfg, "vision_cache", True)
        elif isinstance(use_cache, str):
            use_cache = use_cache.lower() in ("true", "1", "yes", "on")
        ollama_host = getattr(cfg, "ollama_host", "http://localhost:11434")
        return tool_describe_image(
            cwd,
            path=args.get("path", ""), url=args.get("url", ""),
            paths=args.get("paths"), compare_path=args.get("compare_path", ""),
            compare_url=args.get("compare_url", ""), prompt=args.get("prompt", ""),
            mode=args.get("mode", "describe"), model=model,
            auto_pull=args.get("auto_pull", True), max_px=max_px,
            use_cache=bool(use_cache), save_to=args.get("save_to", ""),
            timeout=args.get("timeout"), ollama_host=ollama_host,
        )
    elif name == "sub_ai":
        return tool_sub_ai(args["prompt"], cwd, args.get("model", "qwen2:0.5b"),
                           args.get("system", ""), args.get("timeout", 60))
    elif name == "gamepad":
        return tool_gamepad(args["action"], cwd, args.get("device", ""), args.get("timeout", 5.0))
    elif name == "create_pdf":
        return tool_create_pdf(
            args["path"], args["content"], cwd,
            args.get("title", ""), args.get("author", ""),
        )
    elif name == "compact_conversation":
        # Handled specially in cli.py — should not reach here
        return {"ok": True, "signal": "compact"}
    elif name == "ask_user":
        # Handled specially in cli.py — should not reach here
        return {"ok": False, "error": "ask_user must be handled by the CLI (no terminal access here)"}
    elif name in _dynamic_fns:
        try:
            return _dynamic_fns[name](cwd=cwd, **args)
        except Exception as e:
            return {"ok": False, "error": f"Dynamic tool error: {e}"}
    else:
        return {"ok": False, "error": f"Unknown tool: {name}"}


# ── UI metadata ───────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "read_file":             ("Read",     "cyan"),
    "write_file":            ("Write",    "yellow"),
    "edit_file":             ("Edit",     "yellow"),
    "bash":                  ("Bash",     "red"),
    "list_dir":              ("List",     "cyan"),
    "grep":                  ("Search",   "green"),
    "find_files":            ("Find",     "green"),
    "git":                   ("Git",      "magenta"),
    "npm_dev":               ("npm dev",  "bright_green"),
    "browse":                ("Browse",   "blue"),
    "keyboard":              ("Keyboard", "bright_yellow"),
    "add_tool":              ("AddTool",  "bright_magenta"),
    "compact_conversation":  ("Compact",  "dim"),
    "ask_user":              ("AskUser",  "bright_cyan"),
    "package":               ("Package",  "bright_red"),
    "create_pdf":            ("PDF",        "bright_magenta"),
    "gamepad":               ("Gamepad",    "bright_yellow"),
    "http_request":          ("HTTP",       "blue"),
    "lint":                  ("Lint",       "yellow"),
    "test_run":              ("Tests",      "bright_green"),
    "sqlite_query":          ("SQLite",     "cyan"),
    "secret_scan":           ("SecretScan", "red"),
    "diff_files":            ("Diff",       "cyan"),
    "rename_file":           ("Rename",     "yellow"),
    "copy_file":             ("Copy",       "yellow"),
    "delete_file":           ("Delete",     "red"),
    "hash_file":             ("Hash",       "dim"),
    "todo_scan":             ("TODOs",      "bright_yellow"),
    "format_code":           ("Format",     "bright_green"),
    "port_kill":             ("PortKill",   "red"),
    "notify":                ("Notify",     "bright_cyan"),
    "gamepad":               ("Gamepad",    "bright_yellow"),
    "git_log":               ("GitLog",     "magenta"),
    "git_blame":             ("GitBlame",   "magenta"),
    "git_stash":             ("GitStash",   "magenta"),
    "type_check":            ("TypeCheck",  "yellow"),
    "ps_list":               ("Processes",  "cyan"),
    "kill_proc":             ("KillProc",   "red"),
    "disk_usage":            ("DiskUsage",  "cyan"),
    "service_ctl":           ("Service",    "red"),
    "journalctl":            ("Logs",       "cyan"),
    "download_file":         ("Download",   "blue"),
    "dns_lookup":            ("DNS",        "cyan"),
    "ping":                  ("Ping",       "cyan"),
    "port_scan":             ("PortScan",   "yellow"),
    "docker_ps":             ("DockerPS",   "bright_blue"),
    "docker_exec":           ("DockerExec", "bright_red"),
    "docker_logs":           ("DockerLogs", "bright_blue"),
    "docker_stop":           ("DockerStop", "red"),
    "clipboard_get":         ("Clipboard",  "cyan"),
    "clipboard_set":         ("Clipboard",  "yellow"),
    "screenshot":            ("Screenshot", "bright_cyan"),
    "head_file":             ("Head",       "cyan"),
    "tail_file":             ("Tail",       "cyan"),
    "json_query":            ("JQ",         "cyan"),
    "base64_encode":         ("B64Enc",     "dim"),
    "base64_decode":         ("B64Dec",     "dim"),
    "uuid_gen":              ("UUID",       "dim"),
    "math_eval":             ("Math",       "dim"),
    "size_report":           ("SizeReport", "cyan"),
    "watch_file":            ("WatchFile",  "yellow"),
    "git_diff":              ("GitDiff",    "magenta"),
    "git_branch":            ("GitBranch",  "magenta"),
    "archive_create":        ("Zip",        "yellow"),
    "archive_extract":       ("Unzip",      "yellow"),
    "archive_list":          ("ZipList",    "cyan"),
    "yaml_query":            ("YAML",       "cyan"),
    "env_get":               ("Env",        "cyan"),
    "open_file":             ("Open",       "bright_blue"),
    "regex_test":            ("Regex",      "bright_cyan"),
    "csv_query":             ("CSV",        "cyan"),
    "image_info":            ("ImageInfo",  "bright_cyan"),
    "network_info":          ("NetInfo",    "cyan"),
    "uptime_info":           ("Uptime",     "cyan"),
    "ssl_check":             ("SSL",        "green"),
    "whois_lookup":          ("WHOIS",      "cyan"),
    "top_procs":             ("Top",        "yellow"),
    "cron_list":             ("Cron",       "cyan"),
    "coverage_run":          ("Coverage",   "bright_green"),
    "ast_analyze":           ("AST",        "bright_cyan"),
    "template_render":       ("Template",   "bright_magenta"),
    "sub_ai":                ("SubAI",      "bright_magenta"),
    "describe_image":        ("Vision",     "bright_magenta"),
}

AUTO_APPROVE_MAP: dict[str, str] = {
    "read_file":            "auto_approve_reads",
    "list_dir":             "auto_approve_reads",
    "grep":                 "auto_approve_reads",
    "find_files":           "auto_approve_reads",
    "write_file":           "auto_approve_writes",
    "edit_file":            "auto_approve_writes",
    "bash":                 "auto_approve_bash",
    "git":                  "auto_approve_bash",
    "npm_dev":              "auto_approve_bash",
    "browse":               "auto_approve_reads",
    "keyboard":             "auto_approve_bash",
    "add_tool":             "auto_approve_bash",
    "compact_conversation": "auto_approve_bash",
    "ask_user":             "auto_approve_reads",
    "package":              "auto_approve_bash",
    "create_pdf":           "auto_approve_writes",
    "http_request":         "auto_approve_reads",
    "lint":                 "auto_approve_reads",
    "test_run":             "auto_approve_bash",
    "sqlite_query":         "auto_approve_reads",
    "secret_scan":          "auto_approve_reads",
    "diff_files":           "auto_approve_reads",
    "rename_file":          "auto_approve_writes",
    "copy_file":            "auto_approve_writes",
    "delete_file":          "auto_approve_bash",
    "hash_file":            "auto_approve_reads",
    "todo_scan":            "auto_approve_reads",
    "format_code":          "auto_approve_writes",
    "port_kill":            "auto_approve_bash",
    "notify":               "auto_approve_bash",
    "gamepad":              "auto_approve_reads",
    "git_log":              "auto_approve_reads",
    "git_blame":            "auto_approve_reads",
    "git_stash":            "auto_approve_bash",
    "type_check":           "auto_approve_reads",
    "ps_list":              "auto_approve_reads",
    "kill_proc":            "auto_approve_bash",
    "disk_usage":           "auto_approve_reads",
    "service_ctl":          "auto_approve_bash",
    "journalctl":           "auto_approve_reads",
    "download_file":        "auto_approve_bash",
    "dns_lookup":           "auto_approve_reads",
    "ping":                 "auto_approve_reads",
    "port_scan":            "auto_approve_reads",
    "docker_ps":            "auto_approve_reads",
    "docker_exec":          "auto_approve_bash",
    "docker_logs":          "auto_approve_reads",
    "docker_stop":          "auto_approve_bash",
    "clipboard_get":        "auto_approve_reads",
    "clipboard_set":        "auto_approve_bash",
    "screenshot":           "auto_approve_bash",
    "head_file":            "auto_approve_reads",
    "tail_file":            "auto_approve_reads",
    "json_query":           "auto_approve_reads",
    "base64_encode":        "auto_approve_reads",
    "base64_decode":        "auto_approve_reads",
    "uuid_gen":             "auto_approve_reads",
    "math_eval":            "auto_approve_reads",
    "size_report":          "auto_approve_reads",
    "watch_file":           "auto_approve_reads",
    "git_diff":             "auto_approve_reads",
    "git_branch":           "auto_approve_bash",
    "archive_create":       "auto_approve_writes",
    "archive_extract":      "auto_approve_writes",
    "archive_list":         "auto_approve_reads",
    "yaml_query":           "auto_approve_reads",
    "env_get":              "auto_approve_reads",
    "open_file":            "auto_approve_bash",
    "regex_test":           "auto_approve_reads",
    "csv_query":            "auto_approve_reads",
    "image_info":           "auto_approve_reads",
    "network_info":         "auto_approve_reads",
    "uptime_info":          "auto_approve_reads",
    "ssl_check":            "auto_approve_reads",
    "whois_lookup":         "auto_approve_reads",
    "top_procs":            "auto_approve_reads",
    "cron_list":            "auto_approve_reads",
    "coverage_run":         "auto_approve_bash",
    "ast_analyze":          "auto_approve_reads",
    "template_render":      "auto_approve_reads",
    "sub_ai":               "auto_approve_bash",
    "describe_image":       "auto_approve_bash",
}
