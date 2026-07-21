"""Subagent: self-contained ReAct executor for one subtask brief."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from server.events import EventType, aemit
from shared.types import SubtaskBrief, SubtaskResult, TaskStatus

from .context import (
    assemble_subagent_context,
    read_file_slice,
    regenerate_rolling_summary_prompt,
)

if TYPE_CHECKING:
    from memory import Memory
    from models.client import OllamaClient
    from tools.runtime import ToolRuntime

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a focused software engineering subagent. Your job is to complete exactly one subtask by reading and editing files in your assigned workspace. You work independently — you cannot communicate with other agents.

FIRST STEP — always start by listing the actual files in your workspace before doing anything else:
  run_shell with: find . -type f | grep -v '.git' | sort | head -60
This is mandatory. The file paths in your brief are suggestions only — the real files on disk are the ground truth. Never assume a file exists without verifying.

Rules:
- Only edit files that actually exist on disk (verified via the listing above).
- If a suggested scope file does not exist, find the closest matching real file and work on that instead.
- Do NOT call the 'test' or 'save' tools — those are coordinator-only.
- When you have completed the task, respond with DONE followed by a summary of your changes.
- If you cannot complete the task, respond with FAILED followed by the reason.
- Be precise and make minimal changes that accomplish the goal.
"""


class Subagent:
    """Executes a single SubtaskBrief in an isolated git worktree."""

    def __init__(
        self,
        brief: SubtaskBrief,
        baseline_commit: str,
        repo_path: str,
        worktree_root: str,
        client: "OllamaClient",
        tools: "ToolRuntime",
        memory: "Memory",
        config: dict[str, Any],
    ) -> None:
        self.brief = brief
        self.baseline_commit = baseline_commit
        self.repo_path = repo_path
        self.worktree_root = worktree_root
        self.client = client
        self.tools = tools
        self.memory = memory
        self.config = config

        self.step_cap: int = config.get("subagent_step_cap", 20)
        self.summary_every_n: int = config.get("summary_every_n", 5)
        self.token_budget: int = config.get("context_token_budget", 8192)
        self.ratios: dict[str, float] = config.get(
            "subagent_context_ratios",
            {"task_brief": 0.20, "files": 0.50, "memory": 0.15, "rolling_summary": 0.15},
        )

        self._worktree_path: Optional[str] = None
        self._steps_history: list[dict[str, Any]] = []
        self._rolling_summary: str = ""

    async def run(self) -> SubtaskResult:
        """Main entry: create worktree, run ReAct loop, return result."""
        await aemit(
            EventType.SUBAGENT_SPAWNED,
            {
                "subtask_id": self.brief.id,
                "goal": self.brief.goal,
                "scope": self.brief.scope,
                "model": self.brief.model.name if self.brief.model else "unknown",
                "matched_skills": self.brief.matched_skills,
                "fallback": self.brief.fallback,
            },
        )

        try:
            self._worktree_path = await self._create_worktree()
            result = await self._react_loop()
            return result
        except Exception as exc:
            logger.exception("Subagent %s crashed: %s", self.brief.id, exc)
            return SubtaskResult(
                subtask_id=self.brief.id,
                status=TaskStatus.FAILED,
                error=str(exc),
                steps_taken=len(self._steps_history),
            )
        finally:
            # Worktree cleanup is the coordinator's responsibility (it needs to read the diff).
            pass

    async def _create_worktree(self) -> str:
        """Create an isolated git worktree at baseline_commit."""
        os.makedirs(self.worktree_root, exist_ok=True)
        worktree_path = os.path.join(self.worktree_root, f"subagent-{self.brief.id}")

        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add", "--detach", worktree_path, self.baseline_commit,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to create worktree for {self.brief.id}: {stderr.decode()}"
            )
        logger.debug("Worktree created at %s", worktree_path)
        return worktree_path

    async def remove_worktree(self) -> None:
        """Remove worktree when coordinator is done with it."""
        if not self._worktree_path:
            return
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", "--force", self._worktree_path,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def get_diff(self) -> str:
        """Return the unified diff of changes made in this worktree."""
        if not self._worktree_path:
            return ""
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "HEAD",
            cwd=self._worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode(errors="replace")

    async def _react_loop(self) -> SubtaskResult:
        """Run the ReAct (Reason+Act) loop up to step_cap steps."""
        assert self.brief.model is not None
        from models.client import ChatMessage

        tool_schemas = self.tools.get_schemas(kinds=["action"])
        memory_entries = await self.memory.retrieve(
            self.brief.goal, k=3, include_failures=True
        )

        step = 0
        files_touched: list[str] = []

        while step < self.step_cap:
            # Regenerate rolling summary periodically
            if step > 0 and step % self.summary_every_n == 0:
                self._rolling_summary = await self._regenerate_summary()

            context = assemble_subagent_context(
                brief=self.brief,
                tool_schemas=tool_schemas,
                memory_entries=memory_entries,
                rolling_summary=self._rolling_summary,
                token_budget=self.token_budget,
                ratios=self.ratios,
                workspace=self._worktree_path,
            )

            messages = [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=context),
            ]
            # Add abbreviated step history (last 5 exchanges)
            for s in self._steps_history[-5:]:
                messages.append(ChatMessage(role="assistant", content=s.get("action", "")))
                messages.append(ChatMessage(role="user", content=str(s.get("result", ""))))

            await aemit(
                EventType.SUBAGENT_PROGRESS,
                {"subtask_id": self.brief.id, "step": step},
            )

            response = await self.client.chat(
                model_spec=self.brief.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )

            content = response.content.strip()
            step += 1

            # Check terminal conditions
            if content.upper().startswith("DONE"):
                diff = await self.get_diff()
                files_touched = self._extract_touched_files(diff)
                summary = content[4:].strip() if len(content) > 4 else "Task completed."
                await aemit(
                    EventType.SUBAGENT_DONE,
                    {
                        "subtask_id": self.brief.id,
                        "status": "success",
                        "files_touched": files_touched,
                        "steps": step,
                    },
                )
                return SubtaskResult(
                    subtask_id=self.brief.id,
                    diff=diff,
                    files_touched=files_touched,
                    summary=summary,
                    status=TaskStatus.SUCCESS,
                    steps_taken=step,
                )

            if content.upper().startswith("FAILED"):
                reason = content[6:].strip()
                await aemit(
                    EventType.SUBAGENT_DONE,
                    {"subtask_id": self.brief.id, "status": "failed", "reason": reason},
                )
                return SubtaskResult(
                    subtask_id=self.brief.id,
                    status=TaskStatus.FAILED,
                    error=reason,
                    steps_taken=step,
                )

            # Handle tool calls
            if response.tool_calls:
                result = await self._execute_tool_calls(response.tool_calls)
                for fname in result.get("files_written", []):
                    if fname not in files_touched:
                        files_touched.append(fname)
                self._steps_history.append(
                    {"action": content, "tool_calls": response.tool_calls, "result": result}
                )
            else:
                # Pure reasoning step — record and continue
                self._steps_history.append({"action": content, "result": "(no tool call)"})

        # Reached step cap without DONE/FAILED
        diff = await self.get_diff()
        files_touched = self._extract_touched_files(diff)
        await aemit(
            EventType.SUBAGENT_DONE,
            {"subtask_id": self.brief.id, "status": "partial", "steps": step},
        )
        return SubtaskResult(
            subtask_id=self.brief.id,
            diff=diff,
            files_touched=files_touched,
            summary="Reached step cap without explicit completion.",
            status=TaskStatus.PARTIAL,
            steps_taken=step,
        )

    async def _regenerate_summary(self) -> str:
        """Ask the model to summarize progress so far."""
        from models.client import ChatMessage

        prompt = regenerate_rolling_summary_prompt(self.brief, self._steps_history)
        assert self.brief.model is not None
        response = await self.client.chat(
            model_spec=self.brief.model,
            messages=[ChatMessage(role="user", content=prompt)],
        )
        return response.content.strip()

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, Any]:
        """Execute tool calls sequentially; return combined results."""
        results: dict[str, Any] = {"outputs": [], "files_written": []}
        for tc in tool_calls:
            fn_call = tc.get("function", tc)
            name = fn_call.get("name", "")
            args = fn_call.get("arguments", {})
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            # Inject workspace path for file operations
            if "workspace" not in args and self._worktree_path:
                args["workspace"] = self._worktree_path

            tool_result = await self.tools.call(name, caller="subagent", **args)
            results["outputs"].append(
                {"tool": name, "success": tool_result.success, "value": tool_result.value}
            )
        return results

    def _extract_touched_files(self, diff: str) -> list[str]:
        """Parse unified diff for modified file paths."""
        files: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:].strip()
                if path not in files:
                    files.append(path)
        return files

    @property
    def worktree_path(self) -> Optional[str]:
        return self._worktree_path
