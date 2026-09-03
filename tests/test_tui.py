from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from bob_der.agent import Agent
from bob_der.ollama import OllamaRunner
from bob_der.sessions import SessionStore

try:
    from textual import events
    from textual.widgets import Button, Collapsible, OptionList

    from bob_der.tui import BobDerApp, CopyBox, PromptTextArea
except ImportError:
    Collapsible = None


class StreamingRunner(OllamaRunner):
    def generate(self, prompt: str) -> str:
        response = (
            "Thinking...\nI am reasoning live, one chunk at a time.\n"
            "...done thinking.\n"
            '{"thought":"done","tool":"final","args":{"message":"Visible answer"}}'
        )
        for start in range(0, len(response), 7):
            if self.on_chunk:
                self.on_chunk(response[start : start + 7])
        return response


@unittest.skipIf(Collapsible is None, "Textual is not installed in this test environment")
class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_slash_suggestions_and_plan_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                StreamingRunner(model="fake-primary"),
                Path(directory),
                on_event=lambda *_: None,
            )
            app = BobDerApp(agent, SessionStore(Path(directory) / "sessions"))
            async with app.run_test(size=(100, 36)) as pilot:
                prompt = app.query_one("#prompt", PromptTextArea)
                suggestions = app.query_one("#slash-suggestions", OptionList)
                prompt.load_text("/pl")
                await pilot.pause()
                self.assertEqual(suggestions.styles.display, "block")
                self.assertEqual(suggestions.options[0].id, "/plan")

                prompt.load_text("/plan")
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertTrue(app.plan_mode)
                self.assertTrue(prompt.has_class("plan-mode"))

                prompt.load_text("/deep")
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertTrue(app.deep_mode)
                self.assertFalse(app.plan_mode)

    async def test_thinking_panel_streams_and_toggles_on_click(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                StreamingRunner(model="fake-primary"),
                Path(directory),
                on_event=lambda *_: None,
            )
            app = BobDerApp(agent, SessionStore(Path(directory) / "sessions"))
            async with app.run_test(size=(100, 36)) as pilot:
                prompt = app.query_one("#prompt", PromptTextArea)
                prompt.load_text("test live thinking")
                await pilot.press("ctrl+enter")
                await pilot.pause(0.3)

                panel = app.query_one(Collapsible)
                self.assertTrue(panel.collapsed)
                self.assertIn("reasoning live", str(app.current_thinking.render()))

                title = panel.query_one("CollapsibleTitle")
                self.assertTrue(await pilot.click(title))
                await pilot.pause()
                self.assertFalse(panel.collapsed)
                self.assertEqual(app.current_answer_text, "Visible answer")

                copy_box = app.query_one(CopyBox)
                self.assertEqual(copy_box.copy_text, "Visible answer")
                self.assertTrue(await pilot.click(copy_box.query_one(Button)))
                await pilot.pause()
                self.assertEqual(app.clipboard, "Visible answer")

                prompt.focus()
                prompt.load_text("/copy")
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertEqual(app.clipboard, "Visible answer")

    async def test_input_stays_clickable_and_followups_are_queued(self) -> None:
        class SlowRunner(OllamaRunner):
            calls = 0

            def __init__(self, model: str) -> None:
                super().__init__(model=model)
                self.release_first = threading.Event()

            def generate(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    self.release_first.wait(timeout=2)
                response = (
                    "Thinking...\nworking\n...done thinking.\n"
                    f'{{"tool":"final","args":{{"message":"answer {self.calls}"}}}}'
                )
                if self.on_chunk:
                    self.on_chunk(response)
                return response

        with tempfile.TemporaryDirectory() as directory:
            runner = SlowRunner(model="fake-primary")
            agent = Agent(runner, Path(directory), on_event=lambda *_: None)
            app = BobDerApp(agent, SessionStore(Path(directory) / "sessions"))
            async with app.run_test(size=(100, 36)) as pilot:
                prompt = app.query_one("#prompt", PromptTextArea)
                prompt.load_text("first task")
                await pilot.press("ctrl+enter")
                await pilot.pause(0.05)
                self.assertTrue(app.busy)
                self.assertFalse(prompt.disabled)
                self.assertEqual(
                    app.query_one("#activity").styles.display, "block"
                )

                prompt.focus()
                prompt.load_text("follow-up task")
                await pilot.press("ctrl+enter")
                await pilot.pause(0.02)
                self.assertEqual(len(app.pending), 1)
                self.assertFalse(prompt.disabled)

                runner.release_first.set()
                await pilot.pause(0.5)
                self.assertEqual(runner.calls, 2)
                self.assertFalse(app.busy)
                self.assertEqual(len(app.pending), 0)

    async def test_enter_adds_newline_and_ctrl_enter_submits_paste(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                StreamingRunner(model="fake-primary"),
                Path(directory),
                on_event=lambda *_: None,
            )
            app = BobDerApp(agent, SessionStore(Path(directory) / "sessions"))
            async with app.run_test(size=(100, 36)) as pilot:
                prompt = app.query_one("#prompt", PromptTextArea)
                prompt.load_text("manual line")
                prompt.move_cursor((0, len("manual line")))
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(prompt.text, "manual line\n")
                prompt.load_text("")

                pasted = "Fix this function:\n\n    def example():\n        return 42"
                # This is the bracketed-paste event emitted by a real terminal,
                # not Textual's app-local clipboard shortcut.
                app.post_message(events.Paste(pasted))
                await pilot.pause()

                self.assertEqual(prompt.text, pasted)
                self.assertEqual(prompt.styles.height.value, 6)

                await pilot.press("ctrl+enter")
                await pilot.pause(0.3)
                self.assertEqual(agent.transcript[0]["content"], pasted)
                self.assertEqual(prompt.text, "")

    async def test_ubuntu_ctrl_enter_alias_submits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                StreamingRunner(model="fake-primary"),
                Path(directory),
                on_event=lambda *_: None,
            )
            app = BobDerApp(agent, SessionStore(Path(directory) / "sessions"))
            async with app.run_test(size=(100, 36)) as pilot:
                prompt = app.query_one("#prompt", PromptTextArea)
                prompt.load_text("sent through Ubuntu's Ctrl+Enter sequence")
                await pilot.press("ctrl+j")
                await pilot.pause(0.3)

                self.assertEqual(
                    agent.transcript[0]["content"],
                    "sent through Ubuntu's Ctrl+Enter sequence",
                )
                self.assertEqual(prompt.text, "")


if __name__ == "__main__":
    unittest.main()
