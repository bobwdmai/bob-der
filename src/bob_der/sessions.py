from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class SessionStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = (
            directory
            if directory is not None
            else Path.home() / ".bob-der" / "sessions"
        )

    def new_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    def path(self, session_id: str) -> Path:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError("Invalid session ID")
        base = self.directory.expanduser().resolve()
        target = (base / f"{session_id}.json").resolve()
        target.relative_to(base)
        return target

    def save(
        self,
        session_id: str,
        *,
        workspace: Path,
        model: str,
        transcript: list[dict[str, object]],
        subagent_prompt: str = "",
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(session_id)
        payload = {
            "id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "model": model,
            "subagent_prompt": subagent_prompt,
            "transcript": transcript,
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{session_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink(missing_ok=True)

    def load(self, session_id: str) -> dict[str, Any]:
        target = self.path(session_id)
        if not target.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Session file must contain an object")
        transcript = data.get("transcript")
        if not isinstance(transcript, list) or not all(
            isinstance(item, dict) for item in transcript
        ):
            raise ValueError("Session transcript is invalid")
        return data

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        self.directory.mkdir(parents=True, exist_ok=True)
        sessions: list[dict[str, Any]] = []
        paths = sorted(
            self.directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                transcript = data.get("transcript", [])
                first_user = next(
                    (
                        str(item.get("content", ""))
                        for item in transcript
                        if isinstance(item, dict) and item.get("role") == "user"
                    ),
                    "(empty)",
                )
                sessions.append(
                    {
                        "id": data.get("id", path.stem),
                        "workspace": data.get("workspace", "?"),
                        "updated_at": data.get("updated_at", "?"),
                        "preview": first_user[:72],
                    }
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sessions
