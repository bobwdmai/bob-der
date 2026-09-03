from __future__ import annotations

import json
import codecs
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Callable


class OllamaError(RuntimeError):
    pass


class OllamaSlowError(OllamaError):
    pass


@dataclass
class OllamaRunner:
    model: str = "nemotron-3-super:cloud"
    executable: str = "ollama"
    timeout: int = 0
    on_chunk: Callable[[str], None] | None = None
    thinking_level: str = "true"
    first_token_timeout: float = 0

    def generate(self, prompt: str) -> str:
        # Ollama's CLI can emit interactive cursor-control sequences when it
        # wraps output, even with stdout connected to a pipe. Those sequences
        # corrupt structured JSON. Hide the separate thinking stream and keep
        # the response on unwrapped lines so stdout remains machine-readable.
        command = [
            self.executable,
            "run",
            self.model,
            "--format",
            "json",
            "--think",
            self.thinking_level,
            "--nowordwrap",
        ]
        if self.on_chunk is not None:
            return self._generate_streaming(command, prompt)
        try:
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout if self.timeout > 0 else None,
                check=False,
            )
        except FileNotFoundError as exc:
            raise OllamaError(
                "Ollama is not installed or is not on PATH. Install it from https://ollama.com."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise OllamaError(f"Ollama timed out after {self.timeout} seconds") from exc

        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise OllamaError(f"ollama exited with status {process.returncode}: {detail}")
        if not process.stdout.strip():
            raise OllamaError("Ollama returned an empty response")
        return process.stdout.strip()

    def _generate_streaming(self, command: list[str], prompt: str) -> str:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise OllamaError(
                "Ollama is not installed or is not on PATH. Install it from https://ollama.com."
            ) from exc

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()

        chunks: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def pump(name: str, stream: BinaryIO) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            try:
                while True:
                    raw = os.read(stream.fileno(), 1024)
                    if not raw:
                        break
                    value = decoder.decode(raw)
                    if value:
                        chunks.put((name, value))
                tail = decoder.decode(b"", final=True)
                if tail:
                    chunks.put((name, tail))
            finally:
                chunks.put((name, None))

        threads = [
            threading.Thread(target=pump, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=pump, args=("stderr", process.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()

        stdout: list[str] = []
        stderr: list[str] = []
        finished: set[str] = set()
        started = time.monotonic()
        try:
            while len(finished) < 2:
                if (
                    self.first_token_timeout > 0
                    and not stdout
                    and time.monotonic() - started > self.first_token_timeout
                ):
                    process.kill()
                    process.wait()
                    raise OllamaSlowError(
                        f"{self.model} produced no output within "
                        f"{self.first_token_timeout:g} seconds"
                    )
                if self.timeout > 0 and time.monotonic() - started > self.timeout:
                    process.kill()
                    process.wait()
                    raise OllamaError(f"Ollama timed out after {self.timeout} seconds")
                try:
                    source, value = chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if value is None:
                    finished.add(source)
                elif source == "stdout":
                    stdout.append(value)
                    assert self.on_chunk is not None
                    self.on_chunk(value)
                else:
                    stderr.append(value)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

        return_code = process.wait()
        response = "".join(stdout).strip()
        error = "".join(stderr).strip()
        if return_code != 0:
            raise OllamaError(
                f"ollama exited with status {return_code}: {error or response}"
            )
        if not response:
            raise OllamaError("Ollama returned an empty response")
        return response


def parse_action(response: str) -> dict[str, object]:
    """Extract and validate the first JSON object in a model response."""
    action_response = response
    thinking_end = response.rfind("...done thinking.")
    if thinking_end >= 0:
        action_response = response[thinking_end + len("...done thinking.") :]
    candidates = [action_response.strip(), response.strip()]
    if "```" in response:
        for block in response.split("```"):
            cleaned = block.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
            if cleaned:
                candidates.append(cleaned)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        starts = [index for index, char in enumerate(candidate) if char == "{"]
        for start in starts:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                tool = value.get("tool")
                args = value.get("args")
                if isinstance(tool, str) and isinstance(args, dict):
                    return value
    raise OllamaError(f"Model did not return a valid tool action: {response[:500]}")
