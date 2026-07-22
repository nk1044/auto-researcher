"""Subagent: self-contained ReAct executor for one subtask brief."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from server.events import EventType, aemit
from shared.types import SubtaskBrief, SubtaskResult, TaskStatus

from .context import (
    assemble_subagent_context,
    regenerate_rolling_summary_prompt,
)

if TYPE_CHECKING:
    from memory import Memory
    from models.client import OllamaClient
    from tools.runtime import ToolRuntime

logger = logging.getLogger(__name__)


def _strip_workspace_param(schemas: list[dict]) -> list[dict]:
    """Return schemas with the 'workspace' parameter removed.

    workspace is always auto-injected by the runtime before the tool is called,
    so exposing it to the model causes confusion (it doesn't know the path).
    """
    result = []
    for s in schemas:
        s = copy.deepcopy(s)
        params = s.get("function", {}).get("parameters", {})
        params.get("properties", {}).pop("workspace", None)
        req = params.get("required", [])
        if "workspace" in req:
            req.remove("workspace")
        result.append(s)
    return result


def _parse_text_tool_call(content: str) -> dict | None:
    """Extract a tool call from text content.

    Smaller models (qwen2.5 series, etc.) often output tool-call JSON as fenced
    markdown instead of using Ollama's structured tool-call API.  This parser
    detects that pattern so the loop can execute the tool and feed back a result.

    Recognises:
      ```json
      {"name": "run_shell", "arguments": {"command": "..."}}
      ```
    and bare JSON objects with "name" + "arguments" keys anywhere in the text.
    """
    # 1. Fenced code block (```json … ``` or ``` … ```)
    block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if block:
        try:
            data = json.loads(block.group(1).strip())
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Scan for any JSON object using raw_decode (handles nested braces and } inside strings)
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        idx = content.find("{", pos)
        if idx == -1:
            break
        try:
            data, _ = decoder.raw_decode(content, idx)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                return data
            pos = idx + 1
        except (json.JSONDecodeError, ValueError):
            pos = idx + 1

    return None


SYSTEM_PROMPT = """You are a software engineering subagent. Your workspace is an isolated git worktree — a complete copy of the target repository. Every file you edit is captured as a diff and evaluated against a test oracle.

## Context already provided — use it
Your context below contains:
- "## Subtask Brief" — your exact goal, scope files, and constraints.
- "## In-Scope File Contents" — the CURRENT contents of every file you need to change.
- "## Actual Workspace Files" — a full listing of all files if you need to navigate.

Do NOT list or re-read files that are already shown in your context. Go straight to making the change.

## Workflow

Step 1 — Read your task:
  Your goal and the target files are in "## Subtask Brief". The current file contents are in "## In-Scope File Contents". Read them in your context — do NOT call read_file or run_shell just to rediscover information already given to you.
  Only call read_file if you need a file that is NOT in "## In-Scope File Contents".

Step 2 — Make the change:
  Option A — Surgical edit (preferred):
    Call edit_file with:
      path = the relative path of the file to change
      old_string = exact text copied verbatim from the file content shown in your context
      new_string = the replacement — MUST be genuinely different from old_string
    old_string must match character-for-character. Copy it directly from the shown content.

  Option B — Full rewrite:
    Call write_file with the COMPLETE new file content. Never truncate.

Step 3 — Verify syntax:
  Call run_shell: python3 -m py_compile <filename>
  If SyntaxError: call read_file to see current state, fix it, then re-verify.

Step 4 — Confirm with git:
  Call run_shell: git diff HEAD --stat
  - Files listed → edit landed. Proceed.
  - Empty output → edit was a no-op (new_string matched old_string exactly). Go back to Step 2 with a genuinely different change. Do NOT give up.

Step 5 — Signal completion:
  At least one real file changed: DONE — <one-line summary of exactly what changed>
  Genuinely impossible after multiple real attempts: FAILED — <specific reason>

## Hard rules
- Use ONLY relative paths from "## Actual Workspace Files" or "## In-Scope File Contents".
- workspace is auto-injected — never pass it yourself.
- Do NOT call run_tests or save_to_github.
- DONE requires a non-empty git diff. Never report DONE without one.
- Do NOT list all files with find/ls unless scope is empty and you have no other information.
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
        existing_workspace: Optional[str] = None,
    ) -> None:
        self.brief = brief
        self.baseline_commit = baseline_commit
        self.repo_path = repo_path
        self.worktree_root = worktree_root
        self.client = client
        self.tools = tools
        self.memory = memory
        self.config = config
        self._existing_workspace = existing_workspace

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
        short_id = self.brief.id[:8]
        goal_preview = self.brief.goal[:80].replace("\n", " ")
        logger.info("  [%s] START — %s", short_id, goal_preview)
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
            if self._existing_workspace:
                self._worktree_path = self._existing_workspace
            else:
                self._worktree_path = await self._create_worktree()
            result = await self._react_loop()
            logger.info("  [%s] DONE — status=%s  files=%s  steps=%d",
                        short_id, result.status.value,
                        ", ".join(result.files_touched) or "none", result.steps_taken)
            return result
        except Exception as exc:
            logger.exception("  [%s] CRASHED: %s", short_id, exc)
            return SubtaskResult(
                subtask_id=self.brief.id,
                status=TaskStatus.FAILED,
                error=str(exc),
                steps_taken=len(self._steps_history),
            )
        finally:
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
        if self._existing_workspace:
            return  # coordinator manages this worktree
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
        """Return the unified diff of changes made in this worktree.

        Stage all changes first (git add -A) so that new files created by
        write_file (which are untracked) are included in the diff.
        """
        if not self._worktree_path:
            return ""
        # Stage everything including new/untracked files
        add_proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A",
            cwd=self._worktree_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await add_proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--cached",
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

        raw_schemas = self.tools.get_schemas(kinds=["action"])
        # Strip 'workspace' from schemas — it's always auto-injected; the model
        # must not see it as a required argument or it will hallucinate paths.
        tool_schemas = _strip_workspace_param(raw_schemas)

        memory_entries = await self.memory.retrieve(
            self.brief.goal, k=3, include_failures=True
        )

        # Build one-time context (task brief + file listing)
        context = assemble_subagent_context(
            brief=self.brief,
            tool_schemas=tool_schemas,
            memory_entries=memory_entries,
            rolling_summary=self._rolling_summary,
            token_budget=self.token_budget,
            ratios=self.ratios,
            workspace=self._worktree_path,
        )

        # Running conversation: system + initial context, then tool exchanges
        conversation: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=context),
        ]

        step = 0
        files_touched: list[str] = []
        tool_calls_total = 0

        while step < self.step_cap:
            await aemit(
                EventType.SUBAGENT_PROGRESS,
                {"subtask_id": self.brief.id, "step": step},
            )

            response = await self.client.chat(
                model_spec=self.brief.model,
                messages=conversation,
                tools=tool_schemas if tool_schemas else None,
            )

            content = response.content.strip()
            step += 1

            # ── Tool calls ─────────────────────────────────────────────────
            if response.tool_calls:
                tool_calls_total += len(response.tool_calls)
                conversation.append(
                    ChatMessage(role="assistant", content=content, tool_calls=response.tool_calls)
                )

                for tc in response.tool_calls:
                    fn_call = tc.get("function", tc)
                    name = fn_call.get("name", "")
                    args = fn_call.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    tc_id = tc.get("id", name)

                    logger.info("  [%s] step %d → %s", self.brief.id[:8], step, name)
                    await aemit(
                        EventType.SUBAGENT_STEP,
                        {
                            "subtask_id": self.brief.id,
                            "step": step,
                            "kind": "tool_call",
                            "tool": name,
                            "args": {k: v for k, v in args.items() if k != "content"},
                        },
                    )

                    # Inject workspace before calling
                    if "workspace" not in args and self._worktree_path:
                        args["workspace"] = self._worktree_path

                    tool_result = await self.tools.call(
                        name, caller="subagent", requester_id=self.brief.id, **args
                    )
                    result_text = (
                        json.dumps(tool_result.value)
                        if tool_result.success
                        else f"ERROR: {tool_result.error}"
                    )

                    if not tool_result.success:
                        logger.warning("  [%s] %s failed: %s", self.brief.id[:8], name, tool_result.error)
                    await aemit(
                        EventType.SUBAGENT_STEP,
                        {
                            "subtask_id": self.brief.id,
                            "step": step,
                            "kind": "tool_result",
                            "tool": name,
                            "success": tool_result.success,
                            "result": result_text[:500],
                        },
                    )

                    # Track files written
                    if tool_result.success and isinstance(tool_result.value, dict):
                        written = tool_result.value.get("written")
                        if written and written not in files_touched:
                            files_touched.append(written)

                    # Append proper tool-result message back to conversation
                    conversation.append(
                        ChatMessage(
                            role="tool",
                            content=result_text[:2000],
                            tool_call_id=tc_id,
                        )
                    )
                continue  # next step after processing all tool calls

            # ── Text-based tool call fallback ──────────────────────────────
            # qwen2.5 and similar models output tool-call JSON as markdown text
            # rather than using Ollama's structured tool-call API.  Parse and
            # execute so the loop actually makes progress.
            # Skip parsing if the content is a terminal signal (DONE / FAILED) so
            # the terminal handler below fires correctly.
            _upper = content.upper().lstrip()
            parsed_tc = (
                _parse_text_tool_call(content)
                if content and not _upper.startswith("DONE") and not _upper.startswith("FAILED")
                else None
            )
            if parsed_tc:
                name = parsed_tc.get("name", "")
                args = parsed_tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                tool_calls_total += 1
                conversation.append(ChatMessage(role="assistant", content=content))

                logger.info("  [%s] step %d → %s (text)", self.brief.id[:8], step, name)
                await aemit(
                    EventType.SUBAGENT_STEP,
                    {
                        "subtask_id": self.brief.id,
                        "step": step,
                        "kind": "tool_call",
                        "tool": name,
                        "args": {k: v for k, v in args.items() if k != "content"},
                    },
                )

                if "workspace" not in args and self._worktree_path:
                    args["workspace"] = self._worktree_path

                tool_result = await self.tools.call(
                    name, caller="subagent", requester_id=self.brief.id, **args
                )
                result_text = (
                    json.dumps(tool_result.value)
                    if tool_result.success
                    else f"ERROR: {tool_result.error}"
                )

                if not tool_result.success:
                    logger.warning("  [%s] %s failed: %s", self.brief.id[:8], name, tool_result.error)
                await aemit(
                    EventType.SUBAGENT_STEP,
                    {
                        "subtask_id": self.brief.id,
                        "step": step,
                        "kind": "tool_result",
                        "tool": name,
                        "success": tool_result.success,
                        "result": result_text[:500],
                    },
                )

                if tool_result.success and isinstance(tool_result.value, dict):
                    written = tool_result.value.get("written")
                    if written and written not in files_touched:
                        files_touched.append(written)

                # Feed result back as a user message — text-mode models expect
                # the next human turn, not a structured "tool" role message.
                conversation.append(ChatMessage(
                    role="user",
                    content=(
                        f"Result of {name}:\n{result_text[:2000]}\n\n"
                        "Continue with the next step of the workflow."
                    ),
                ))
                continue

            # ── No tool calls — text-only response ─────────────────────────
            await aemit(
                EventType.SUBAGENT_STEP,
                {
                    "subtask_id": self.brief.id,
                    "step": step,
                    "kind": "thinking",
                    "content": content[:300],
                },
            )

            upper = content.upper()

            # Check terminal: DONE
            if upper.startswith("DONE"):
                diff = await self.get_diff()
                files_touched = self._extract_touched_files(diff) or files_touched
                if not diff.strip():
                    # Model said DONE without producing any real diff — push back.
                    # This covers both "no tool calls at all" and "tool calls that were
                    # all no-ops" (e.g. edit_file where new_string == old_string).
                    logger.warning(
                        "Subagent %s said DONE but git diff is empty (tool_calls=%d) — pushing back",
                        self.brief.id[:8], tool_calls_total,
                    )
                    conversation.append(ChatMessage(role="assistant", content=content))
                    if tool_calls_total == 0:
                        pushback = (
                            "You said DONE but have not made any changes yet. "
                            "You MUST call run_shell first to list files, then read_file to inspect "
                            "the target file, then write_file or edit_file to modify it. "
                            "Start over from Step 1."
                        )
                    else:
                        pushback = (
                            "You said DONE but git diff HEAD --stat shows no changes. "
                            "Your edits were no-ops — the new content is identical to the original. "
                            "Read the file again with read_file, then call edit_file with a "
                            "new_string that is GENUINELY DIFFERENT from old_string. "
                            "Make a real, substantive code change."
                        )
                    conversation.append(ChatMessage(role="user", content=pushback))
                    continue

                summary = content[4:].strip(" —:-") or "Task completed."
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

            # Check terminal: FAILED
            if upper.startswith("FAILED"):
                reason = content[6:].strip(" —:-")
                logger.warning("Subagent %s reported FAILED: %s", self.brief.id[:8], reason[:200])
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

            # Pure reasoning — append and continue
            conversation.append(ChatMessage(role="assistant", content=content))
            # Nudge if the model keeps reasoning without using tools
            if step % 3 == 0 and tool_calls_total == 0:
                conversation.append(ChatMessage(
                    role="user",
                    content="Use a tool now. Call run_shell to list files in the workspace.",
                ))

        # Step cap reached
        diff = await self.get_diff()
        files_touched = self._extract_touched_files(diff) or files_touched
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
