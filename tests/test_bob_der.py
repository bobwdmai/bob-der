from __future__ import annotations

import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch

from bob_der.agent import Agent
from bob_der.cli import partial_final_message, partial_thinking
from bob_der.ollama import OllamaRunner, OllamaSlowError, parse_action
from bob_der.sessions import SessionStore
from bob_der.tools import EYES_MODEL, ToolExecutor, available_tool_schemas


class FakeRunner(OllamaRunner):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(model="fake")
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class ParserTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        action = parse_action('{"tool":"shell","args":{"command":"pwd"}}')
        self.assertEqual(action["tool"], "shell")

    def test_partial_final_message_stream(self) -> None:
        partial = '{"thought":"done","tool":"final","args":{"message":"hello\\nwor'
        self.assertEqual(partial_final_message(partial), "hello\nwor")
        self.assertIsNone(
            partial_final_message('{"tool":"shell","args":{"command":"echo hi"}}')
        )

    def test_thinking_is_separated_from_action(self) -> None:
        response = (
            'Thinking...\nConsidering {"tool":"shell","args":{"command":"wrong"}}'
            '\n...done thinking.\n'
            '{"tool":"final","args":{"message":"right"}}'
        )
        self.assertIn("Considering", partial_thinking(response))
        self.assertEqual(partial_final_message(response), "right")
        self.assertEqual(parse_action(response)["tool"], "final")

    def test_fenced_json(self) -> None:
        action = parse_action('```json\n{"tool":"final","args":{"message":"ok"}}\n```')
        self.assertEqual(action["args"]["message"], "ok")


class RunnerTests(unittest.TestCase):
    @patch("bob_der.ollama.subprocess.run")
    def test_runner_requests_machine_readable_output(self, run) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = '{"tool":"final","args":{"message":"ok"}}'
        run.return_value.stderr = ""
        runner = OllamaRunner()
        runner.generate("prompt")
        command = run.call_args.args[0]
        self.assertIn("--format", command)
        self.assertIn("--think", command)
        self.assertIn("true", command)
        self.assertNotIn("--hidethinking", command)
        self.assertIn("--nowordwrap", command)


class ToolTests(unittest.TestCase):
    def test_write_read_replace_and_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolExecutor(Path(directory))
            self.assertTrue(tools.execute("write_file", {"path": "a.txt", "content": "hello"})["ok"])
            self.assertEqual(tools.execute("read_file", {"path": "a.txt"})["content"], "hello")
            result = tools.execute(
                "replace_in_file", {"path": "a.txt", "old": "hello", "new": "world"}
            )
            self.assertEqual(result["replacements"], 1)
            shell = tools.execute("shell", {"command": "printf test"})
            self.assertEqual(shell["output"], "test")

    def test_absolute_paths_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolExecutor(Path(directory))
            absolute = Path(directory) / "absolute.txt"
            result = tools.execute("write_file", {"path": str(absolute), "content": "yes"})
            self.assertTrue(result["ok"])
            self.assertEqual(absolute.read_text(), "yes")

    def test_beta_tool_catalog_has_only_the_two_configured_agents(self) -> None:
        names = {
            schema["function"]["name"] for schema in available_tool_schemas()
        }
        self.assertEqual(len(names), 79)
        self.assertNotIn("sub_ai", names)
        self.assertIn("describe_image", names)
        self.assertNotIn("ask_user", names)
        self.assertNotIn("compact_conversation", names)
        self.assertTrue(
            {
                "grep",
                "git_diff",
                "test_run",
                "archive_extract",
                "http_request",
                "docker_exec",
                "clipboard_set",
                "sqlite_query",
                "create_pdf",
            }.issubset(names)
        )

    @patch("bob_der.tools.beta_tools.dispatch")
    def test_describe_image_uses_fixed_eyes_model(self, dispatch) -> None:
        dispatch.return_value = {"ok": True, "description": "a test image"}
        with tempfile.TemporaryDirectory() as directory:
            result = ToolExecutor(Path(directory)).execute(
                "describe_image", {"path": "screen.png", "mode": "code"}
            )
        self.assertTrue(result["ok"])
        self.assertEqual(dispatch.call_args.kwargs["cfg"].vision_model, EYES_MODEL)
        self.assertEqual(dispatch.call_args.args[1]["model"], EYES_MODEL)
        self.assertEqual(EYES_MODEL, "gemma4:31b-cloud")

    def test_safe_beta_tools_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("hello\nTODO: verify tools\n")
            tools = ToolExecutor(root)

            encoded = tools.execute("base64_encode", {"text": "hello"})
            self.assertTrue(encoded["ok"])
            decoded = tools.execute("base64_decode", {"text": encoded["encoded"]})
            self.assertEqual(decoded["text"], "hello")
            self.assertEqual(tools.execute("math_eval", {"expression": "2+3*4"})["result"], "14")
            self.assertTrue(tools.execute("regex_test", {"pattern": "ell", "text": "hello"})["ok"])

            copied = tools.execute("copy_file", {"src": "source.txt", "dst": "copy.txt"})
            self.assertTrue(copied["ok"])
            renamed = tools.execute("rename_file", {"src": "copy.txt", "dst": "moved.txt"})
            self.assertTrue(renamed["ok"])
            digest = tools.execute("hash_file", {"path": "moved.txt"})
            self.assertTrue(digest["ok"])
            self.assertEqual(len(digest["sha256"]), 64)
            self.assertTrue(tools.execute("delete_file", {"path": "moved.txt"})["ok"])

            todos = tools.execute("todo_scan", {"path": "."})
            self.assertTrue(todos["ok"])
            self.assertEqual(todos["count"], 1)

    def test_shell_has_no_timeout_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolExecutor(Path(directory))
            with patch("bob_der.tools.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "ok"
                tools.execute("shell", {"command": "anything"})
                self.assertIsNone(run.call_args.kwargs["timeout"])

    def test_tool_boundary_rejects_bad_calls_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolExecutor(Path(directory))
            self.assertFalse(tools.execute("", {})["ok"])
            self.assertFalse(tools.execute("missing", {})["ok"])
            self.assertFalse(tools.execute("read_file", ["not", "an", "object"])["ok"])

            missing = tools.execute("copy_file", {})
            self.assertFalse(missing["ok"])
            self.assertIn("missing required argument", missing["error"])

            with patch("bob_der.tools.shutil.which", return_value=None):
                invalid_regex = tools.execute("grep", {"pattern": "[", "path": "."})
            self.assertFalse(invalid_regex["ok"])
            self.assertIn("error", invalid_regex)

    def test_every_required_beta_tool_reports_missing_arguments_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolExecutor(Path(directory))
            checked = 0
            for schema in available_tool_schemas():
                function = schema["function"]
                required = function.get("parameters", {}).get("required", [])
                if not required:
                    continue
                result = tools.execute(function["name"], {})
                self.assertIsInstance(result, dict, function["name"])
                self.assertFalse(result["ok"], function["name"])
                self.assertIn("error", result, function["name"])
                checked += 1
            self.assertGreater(checked, 40)

    @patch("bob_der.tools.beta_tools.dispatch")
    def test_beta_tool_exceptions_are_isolated(self, dispatch) -> None:
        dispatch.side_effect = RuntimeError("beta exploded")
        with tempfile.TemporaryDirectory() as directory:
            result = ToolExecutor(Path(directory)).execute(
                "math_eval", {"expression": "1+1"}
            )
        self.assertFalse(result["ok"])
        self.assertIn("beta exploded", result["error"])

    @patch("bob_der.tools.beta_tools.dispatch")
    def test_tool_results_are_bounded_and_json_safe(self, dispatch) -> None:
        recursive: list[object] = []
        recursive.append(recursive)
        dispatch.return_value = {
            "ok": True,
            "path": Path("result.txt"),
            "binary": b"\xffhello",
            "score": float("nan"),
            "recursive": recursive,
            "payload": "x" * 20_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = ToolExecutor(Path(directory), max_output_chars=2_000).execute(
                "math_eval", {"expression": "1+1"}
            )
        encoded = json.dumps(result, allow_nan=False)
        self.assertLess(len(encoded), 2_500)
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])

    def test_atomic_write_preserves_original_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "important.txt"
            target.write_text("original")
            target.chmod(0o640)
            tools = ToolExecutor(Path(directory))

            with patch("bob_der.tools.os.replace", side_effect=OSError("disk failure")):
                failed = tools.execute(
                    "write_file", {"path": "important.txt", "content": "partial"}
                )
            self.assertFalse(failed["ok"])
            self.assertEqual(target.read_text(), "original")
            self.assertFalse(list(Path(directory).glob(".important.txt.*.tmp")))

            succeeded = tools.execute(
                "write_file", {"path": "important.txt", "content": "complete"}
            )
            self.assertTrue(succeeded["ok"])
            self.assertEqual(target.read_text(), "complete")
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o640)

    def test_shell_replaces_invalid_output_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = ToolExecutor(Path(directory)).execute(
                "shell", {"command": "printf '\\377'"}
            )
        self.assertTrue(result["ok"])
        self.assertIn("�", result["output"])

    def test_plan_shell_can_read_but_cannot_edit_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "protected.txt"
            target.write_text("original")
            tools = ToolExecutor(root)

            read = tools.execute_read_only(
                "shell", {"command": "cat protected.txt"}
            )
            self.assertTrue(read["ok"])
            self.assertEqual(read["exit_code"], 0)
            self.assertEqual(read["output"], "original")
            self.assertTrue(read["read_only"])

            write = tools.execute_read_only(
                "shell", {"command": "printf changed > protected.txt"}
            )
            self.assertTrue(write["ok"])
            self.assertNotEqual(write["exit_code"], 0)
            self.assertEqual(target.read_text(), "original")

            blocked = tools.execute_read_only(
                "write_file", {"path": "new.txt", "content": "no"}
            )
            self.assertFalse(blocked["ok"])
            self.assertFalse((root / "new.txt").exists())


class AgentTests(unittest.TestCase):
    def test_agent_executes_until_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    '{"thought":"create","tool":"write_file","args":{"path":"x.txt","content":"x"}}',
                    '{"thought":"done","tool":"final","args":{"message":"finished"}}',
                ]
            )
            agent = Agent(runner, Path(directory))
            self.assertEqual(agent.run("create x"), "finished")
            self.assertEqual((Path(directory) / "x.txt").read_text(), "x")

    def test_slow_main_falls_back_to_only_subagent(self) -> None:
        class SlowRunner(FakeRunner):
            def generate(self, prompt: str) -> str:
                raise OllamaSlowError("main was slow")

        with tempfile.TemporaryDirectory() as directory:
            events: list[tuple[str, dict[str, object]]] = []
            main = SlowRunner([])
            main.model = "nemotron-3-super:cloud"
            subagent = FakeRunner(
                ['{"tool":"final","args":{"message":"fast fallback"}}']
            )
            subagent.model = "gpt-oss:20b-cloud"
            agent = Agent(
                main,
                Path(directory),
                on_event=lambda event, data: events.append((event, data)),
                subagent_runner=subagent,
            )
            self.assertEqual(agent.run("respond"), "fast fallback")
            fallback = [data for event, data in events if event == "fallback"]
            self.assertEqual(fallback[0]["to_model"], "gpt-oss:20b-cloud")

    def test_workspace_subagent_prompt_and_undo_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            agent = Agent(FakeRunner([]), root)
            agent.transcript = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": {"tool": "final", "args": {}}},
                {"role": "user", "content": "second"},
                {"role": "tool", "content": {}},
            ]
            self.assertTrue(agent.undo_last_turn())
            self.assertEqual(len(agent.transcript), 2)
            agent.set_workspace(child)
            self.assertEqual(agent.workspace, child.resolve())
            agent.set_subagent_system_prompt("Review security boundaries first.")
            self.assertEqual(
                agent.subagent_system_prompt, "Review security boundaries first."
            )

    def test_main_agent_can_delegate_to_gpt_oss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main = FakeRunner(
                [
                    '{"thought":"delegate","tool":"subagent","args":{"task":"inspect the project"}}',
                    '{"thought":"done","tool":"final","args":{"message":"delegated"}}',
                ]
            )
            subagent = FakeRunner(
                ['{"thought":"done","tool":"final","args":{"message":"inspection complete"}}']
            )
            subagent.model = "gpt-oss:20b-cloud"
            agent = Agent(main, Path(directory), subagent_runner=subagent)
            agent.set_subagent_system_prompt("Audit authentication before editing.")
            self.assertEqual(agent.run("delegate this"), "delegated")
            tool_results = [item for item in agent.transcript if item["role"] == "tool"]
            result = tool_results[0]["content"]["result"]
            self.assertTrue(result["ok"])
            self.assertEqual(result["model"], "gpt-oss:20b-cloud")
            self.assertIn("Audit authentication before editing.", subagent.prompts[0])

    def test_plan_mode_rejects_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    '{"tool":"write_file","args":{"path":"blocked.txt","content":"no"}}',
                    '{"tool":"final","args":{"message":"1. Inspect\\n2. Implement\\n3. Test"}}',
                ]
            )
            agent = Agent(runner, Path(directory))
            result = agent.run("plan a feature", plan_mode=True)
            self.assertIn("Implement", result)
            self.assertFalse((Path(directory) / "blocked.txt").exists())
            feedback = [item for item in agent.transcript if item["role"] == "tool"]
            self.assertIn("disabled in plan mode", feedback[0]["content"]["result"]["error"])

    def test_plan_mode_can_inspect_files_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('hello')\n")
            runner = FakeRunner(
                [
                    '{"tool":"read_file","args":{"path":"app.py"}}',
                    '{"tool":"write_file","args":{"path":"blocked.txt","content":"no"}}',
                    '{"tool":"final","args":{"message":"1. Update app.py\\n2. Run tests"}}',
                ]
            )
            agent = Agent(runner, root)
            result = agent.run("plan a change", plan_mode=True)

            self.assertIn("Update app.py", result)
            self.assertFalse((root / "blocked.txt").exists())
            feedback = [
                item["content"]["result"]
                for item in agent.transcript
                if item["role"] == "tool"
            ]
            self.assertTrue(feedback[0]["ok"])
            self.assertIn("print('hello')", feedback[0]["content"])
            self.assertFalse(feedback[1]["ok"])

    def test_deep_mode_enforces_phases_then_executes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                [
                    '{"tool":"deep_phase","args":{"stage":"reason","content":"Inspect constraints."}}',
                    '{"tool":"deep_phase","args":{"stage":"plan","content":"Draft implementation."}}',
                    '{"tool":"deep_phase","args":{"stage":"revise","content":"Add verification."}}',
                    '{"tool":"write_file","args":{"path":"deep.txt","content":"done"}}',
                    '{"tool":"final","args":{"message":"Deep workflow complete."}}',
                ]
            )
            agent = Agent(runner, Path(directory), max_steps=1)
            result = agent.run("deeply implement this", deep_mode=True)
            self.assertEqual(result, "Deep workflow complete.")
            self.assertEqual(runner.thinking_level, "high")
            self.assertEqual(runner.timeout, 0)
            self.assertEqual((Path(directory) / "deep.txt").read_text(), "done")
            phases = [
                item["content"]["result"].get("stage")
                for item in agent.transcript
                if item["role"] == "tool"
                and item["content"].get("tool") == "deep_phase"
            ]
            self.assertEqual(phases, ["reason", "plan", "revise"])


class SessionTests(unittest.TestCase):
    def test_save_list_and_load_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session_id = store.new_id()
            transcript = [{"role": "user", "content": "fix the tests"}]
            store.save(
                session_id,
                workspace=Path(directory),
                model="nemotron-3-super:cloud",
                transcript=transcript,
                subagent_prompt="Focus on tests.",
            )
            loaded = store.load(session_id)
            self.assertEqual(loaded["transcript"], transcript)
            self.assertEqual(loaded["subagent_prompt"], "Focus on tests.")
            self.assertEqual(store.list()[0]["id"], session_id)

    def test_session_ids_cannot_escape_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            with self.assertRaises(ValueError):
                store.load("../outside")


if __name__ == "__main__":
    unittest.main()
