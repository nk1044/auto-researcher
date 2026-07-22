"""Coordinator: the infinite hypothesis→decompose→execute→integrate→test→learn loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from server.events import EventType, aemit
from shared.types import (
    AgentState,
    Hypothesis,
    IntegrationResult,
    IterationRecord,
    OutcomeType,
    SubtaskBrief,
    SubtaskResult,
    TaskStatus,
)

from .context import assemble_coordinator_context, assemble_integration_context
from .decomposer import _extract_json, build_subtask_briefs, make_decompose_messages
from .integrator import (
    apply_diff_to_worktree,
    build_integration_messages,
    create_integration_worktree,
    get_worktree_diff,
    naive_merge,
    remove_worktree,
)

logger = logging.getLogger(__name__)

HYPOTHESIS_SYSTEM = """You are an intelligent software engineering researcher. Your goal is to iteratively improve a codebase's test pass-rate score by forming one concrete, actionable hypothesis per iteration.

Reasoning strategy — follow this in order:
1. FOLLOW THE GRADIENT — if the score has been improving recently, continue refining that direction.
2. BUILD ON SUCCESS — if a past hypothesis scored well, improve upon it rather than abandoning it entirely.
3. AVOID DEAD ENDS — do not repeat an approach that already failed and showed no score improvement.
4. STAY INCREMENTAL — prefer a small targeted change over a large rewrite; small wins compound.

Output ONLY JSON: {"hypothesis": "...", "rationale": "...", "target_files": ["..."]}"""

HYPOTHESIS_USER = """Iteration: {iteration}  |  Current baseline score: {baseline:.4f}

Recent score trajectory (newest first):
{trajectory}

Best approach so far (highest scoring win):
{best_win}

{context}

Form ONE hypothesis that improves the score.
- If the trajectory is trending upward, refine or extend the best approach.
- If progress has stalled, try a meaningfully different angle.
- Name the specific files and functions you intend to change.
Output JSON only."""

NOVELTY_SYSTEM = """You are an intelligent software engineering researcher. The agent is stuck — the last hypothesis was nearly identical to a past failure with no improvement signal.

Your job is to generate a NOVEL hypothesis that breaks out of the current dead end.

Output ONLY JSON: {"hypothesis": "...", "rationale": "...", "target_files": [...]}"""

NOVELTY_USER = """Stuck hypothesis (too similar to a past failure, no win nearby):
{rejected_hypothesis}

Recent failures to AVOID repeating:
{past_failures}

Break out by trying a completely different area of the codebase or a different type of improvement.
Output JSON only."""


class Coordinator:
    """Runs the infinite improvement loop. Stops only when stop_requested is set."""

    def __init__(
        self,
        config: dict[str, Any],
        memory: Any,
        client: Any,
        router: Any,
        tools: Any,
    ) -> None:
        self.config = config
        self.memory = memory
        self.client = client
        self.router = router
        self.tools = tools

        self.stop_requested: bool = False
        self.pause_gate: asyncio.Event = asyncio.Event()
        self.pause_gate.set()  # starts unpaused

        self.state: AgentState = AgentState()
        self.baseline: float = 0.0
        self._sem = asyncio.Semaphore(config.get("max_subagents", 4))

        self.repo_path: str = config["target_repo"]
        self.worktree_root: str = config.get("worktree_root", "/tmp/auto-researcher/worktrees")
        self.max_subagents: int = config.get("max_subagents", 4)
        self.dup_threshold: float = config.get("dup_threshold", 0.97)
        self.novelty_boost: float = config.get("novelty_boost", 0.3)
        self.token_budget: int = config.get("context_token_budget", 8192)
        self.coord_ratios: dict[str, float] = config.get(
            "coordinator_context_ratios",
            {"task_spec": 0.20, "memory": 0.30, "files": 0.35, "rolling_summary": 0.15},
        )
        self.sub_ratios: dict[str, float] = config.get(
            "subagent_context_ratios",
            {"task_brief": 0.20, "files": 0.50, "memory": 0.15, "rolling_summary": 0.15},
        )
        self.protected_patterns: list[str] = config.get(
            "protected_patterns",
            ["tests/", "test/", "held_out/", "eval/", "benchmark/"],
        )

        self._current_hypothesis: Optional[Hypothesis] = None
        self._iteration_rolling_summary: str = ""

    async def run(self) -> None:
        """The infinite loop. Never returns until stop_requested is set."""
        await self._load_state()
        await aemit(
            EventType.LOOP_STARTED,
            {"baseline": self.baseline, "iteration": self.state.iteration},
        )
        logger.info("Coordinator loop starting at iteration %d, baseline=%.4f",
                    self.state.iteration, self.baseline)

        while not self.stop_requested:
            await self.pause_gate.wait()  # blocks when paused; never exits the loop
            if self.stop_requested:
                break

            try:
                await self._run_one_iteration()
            except Exception as exc:
                logger.exception("Unhandled error in iteration %d: %s",
                                 self.state.iteration, exc)
                await aemit(
                    EventType.ERROR,
                    {"error": str(exc), "iteration": self.state.iteration},
                    iteration=self.state.iteration,
                )
                await asyncio.sleep(5)  # backoff before retrying

        await self._drain_and_flush()

    async def _run_one_iteration(self) -> None:
        n = self.state.iteration
        logger.info("=== Iteration %d ===", n)
        self._iteration_rolling_summary = ""

        # 1. Form one hypothesis
        hyp = await self.form_hypothesis()
        self._current_hypothesis = hyp

        # 2. Anti-repetition gate — only block near-exact failure duplicates that
        #    are NOT building on a past win (which would be an incremental improvement).
        is_dup = await self.memory.is_duplicate_failure(hyp.text)
        if is_dup:
            building_on_win = await self.memory.is_building_on_win(hyp.text)
            if building_on_win:
                logger.info(
                    "Hypothesis is similar to a failure but also close to a past win — "
                    "allowing through as an incremental improvement attempt"
                )
            else:
                logger.info("Hypothesis is a near-exact failure duplicate with no win nearby; reforming")
                await aemit(EventType.DUP_REJECTED, {"hypothesis": hyp.text}, iteration=n)
                hyp = await self.reform_with_novelty(hyp)
                self._current_hypothesis = hyp

        # 3. Decompose + route models
        briefs = await self.decompose(hyp)
        for b in briefs:
            model_spec, matched, fallback = self.router.select(b.required_skills)
            b.model = model_spec
            b.matched_skills = matched
            b.fallback = fallback
            await aemit(
                EventType.MODEL_ROUTED,
                {
                    "subtask_id": b.id,
                    "model": model_spec.name,
                    "matched_skills": matched,
                    "fallback": fallback,
                },
                iteration=n,
            )

        # 4. Dispatch subagents concurrently
        results = await self._dispatch_subagents(briefs)
        ok_results = [r for r in results if r is not None and not isinstance(r, BaseException)]

        # 5. Review + integrate
        integrated = await self.review_and_integrate(hyp, ok_results)

        # 6. Test once
        await aemit(EventType.REVIEW_INTEGRATE, {"hypothesis": hyp.text, "diff_len": len(integrated.diff)}, iteration=n)
        score, remark = await self._run_test(integrated.path)
        await aemit(
            EventType.TEST_SCORED,
            {"score": score, "remark": remark, "baseline": self.baseline},
            iteration=n,
        )

        # 7. Record
        outcome = OutcomeType.WIN if score > self.baseline else OutcomeType.MISTAKE
        record = IterationRecord(
            id=str(uuid.uuid4()),
            hypothesis=hyp.text,
            integrated_diff_hash=hashlib.sha256(integrated.diff.encode()).hexdigest()[:16],
            subagent_contribs=[
                {"id": r.subtask_id, "status": r.status.value, "files": r.files_touched}
                for r in ok_results
            ],
            score=score,
            remark=remark,
            outcome=outcome,
            baseline_before=self.baseline,
            iteration=n,
        )
        await self.memory.record(record)
        await aemit(
            EventType.MEMORY_RECORDED,
            {"outcome": outcome.value, "score": score, "iteration": n},
            iteration=n,
        )

        # 8. Save on improvement
        if score > self.baseline and integrated.diff.strip():
            saved = await self._maybe_save(integrated, hyp, score, n)
            if saved:
                self.baseline = score
                await self._advance_working_commit(integrated.path)

        # 9. Cleanup worktrees
        await self._cleanup_worktrees(results, integrated.path)

        # Advance iteration counter
        self.state.iteration += 1
        self.state.baseline_score = self.baseline
        self.memory.save_state(self.state)

    async def form_hypothesis(self) -> Hypothesis:
        """Assemble coordinator context and ask the model for one hypothesis."""
        n = self.state.iteration

        wins = await self.memory.retrieve(
            "improve test score", k=5, include_failures=False
        )
        failures = await self.memory.top_failures("improvement hypothesis", k=5)
        best_wins = await self.memory.get_best_wins(k=3)
        file_slices = self._load_file_slices(max_files=8)

        context = assemble_coordinator_context(
            task_spec=self._task_spec(),
            tool_schemas=[],
            memory_wins=wins,
            memory_failures=failures,
            file_slices=file_slices,
            rolling_summary=self._iteration_rolling_summary,
            token_budget=self.token_budget,
            ratios=self.coord_ratios,
        )

        from models.client import ChatMessage

        messages = [
            ChatMessage(role="system", content=HYPOTHESIS_SYSTEM),
            ChatMessage(
                role="user",
                content=HYPOTHESIS_USER.format(
                    iteration=n,
                    baseline=self.baseline,
                    trajectory=self._format_trajectory(last_n=7),
                    best_win=self._format_best_win(best_wins),
                    context=context,
                ),
            ),
        ]

        response = await self.client.chat(
            model_spec=self.client.registry.coordinator,
            messages=messages,
        )

        try:
            data = _extract_json(response.content)
        except ValueError:
            data = {"hypothesis": response.content.strip(), "rationale": ""}

        hyp = Hypothesis(
            text=data.get("hypothesis", response.content.strip()),
            rationale=data.get("rationale", ""),
            iteration=n,
        )
        await aemit(
            EventType.HYPOTHESIS_FORMED,
            {"hypothesis": hyp.text, "rationale": hyp.rationale},
            iteration=n,
        )
        logger.info("Hypothesis formed: %s", hyp.text[:120])
        return hyp

    async def reform_with_novelty(self, rejected: Hypothesis) -> Hypothesis:
        """Re-form hypothesis with novelty boost after duplicate rejection."""
        from models.client import ChatMessage

        failures = await self.memory.top_failures(rejected.text, k=8)
        failures_text = "\n".join(f"- {f.text[:200]}" for f in failures)

        # Boost temperature for novelty
        options_override = {}
        base_temp = self.client.registry.coordinator.options.get("temperature", 0.7)
        options_override["temperature"] = min(1.0, base_temp + self.novelty_boost)

        messages = [
            ChatMessage(role="system", content=NOVELTY_SYSTEM),
            ChatMessage(
                role="user",
                content=NOVELTY_USER.format(
                    rejected_hypothesis=rejected.text,
                    past_failures=failures_text or "(none on record)",
                ),
            ),
        ]

        response = await self.client.chat(
            model_spec=self.client.registry.coordinator,
            messages=messages,
            options_override=options_override,
        )

        try:
            data = _extract_json(response.content)
        except ValueError:
            data = {"hypothesis": response.content.strip(), "rationale": "novelty-forced"}

        hyp = Hypothesis(
            text=data.get("hypothesis", response.content.strip()),
            rationale=data.get("rationale", "novelty-forced"),
            iteration=rejected.iteration,
        )
        logger.info("Novelty hypothesis: %s", hyp.text[:120])
        return hyp

    async def decompose(self, hyp: Hypothesis) -> list[SubtaskBrief]:
        """Ask coordinator model to decompose hypothesis into subtask briefs."""
        from models.client import ChatMessage

        file_listing = self._list_repo_files(max_files=50)
        messages_raw = make_decompose_messages(
            hypothesis=hyp,
            repo_path=self.repo_path,
            file_listing=file_listing,
            max_subagents=self.max_subagents,
        )
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_raw]

        response = await self.client.chat(
            model_spec=self.client.registry.coordinator,
            messages=messages,
        )

        try:
            decomposition = _extract_json(response.content)
        except ValueError:
            # Fallback: single subtask covering all files
            logger.warning("Decompose JSON parse failed; using fallback single subtask")
            decomposition = {
                "subtasks": [
                    {
                        "goal": hyp.text,
                        "scope": [],
                        "constraints": "Do not modify test files.",
                        "expected_output": "Improved code with no regressions",
                        "required_skills": ["code"],
                    }
                ]
            }

        briefs = build_subtask_briefs(hyp, decomposition, self.max_subagents)
        await aemit(
            EventType.DECOMPOSED,
            {"hypothesis": hyp.text, "n_subtasks": len(briefs)},
            iteration=hyp.iteration,
        )
        return briefs

    async def _dispatch_subagents(
        self, briefs: list[SubtaskBrief]
    ) -> list[SubtaskResult | BaseException | None]:
        """Run all subagents concurrently under the semaphore."""
        from subagent.subagent import Subagent

        async def run_one(brief: SubtaskBrief) -> SubtaskResult:
            async with self._sem:
                agent = Subagent(
                    brief=brief,
                    baseline_commit=self.state.working_commit,
                    repo_path=self.repo_path,
                    worktree_root=self.worktree_root,
                    client=self.client,
                    tools=self.tools,
                    memory=self.memory,
                    config=self.config,
                )
                return await agent.run()

        results = await asyncio.gather(
            *[run_one(b) for b in briefs],
            return_exceptions=True,
        )

        # Log any exceptions (but don't crash the loop)
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.warning(
                    "Subagent %s raised exception: %s", briefs[i].id, r
                )

        return list(results)

    async def review_and_integrate(
        self, hyp: Hypothesis, results: list[SubtaskResult]
    ) -> IntegrationResult:
        """Coordinator reviews each result and merges into one integration worktree."""
        integration_id = str(uuid.uuid4())[:8]
        integration_path = await create_integration_worktree(
            baseline_commit=self.state.working_commit,
            repo_path=self.repo_path,
            worktree_root=self.worktree_root,
            integration_id=integration_id,
        )

        ok_results = [r for r in results if r.diff and r.status != TaskStatus.FAILED]
        if not ok_results:
            return IntegrationResult(
                hypothesis_id=hyp.id,
                path=integration_path,
                summary="No successful subagent results to integrate.",
            )

        # Try LLM-guided integration
        try:
            from models.client import ChatMessage

            file_slices = self._load_file_slices(max_files=5)
            context = assemble_integration_context(
                hypothesis=hyp.text,
                subagent_results=ok_results,
                file_slices=file_slices,
                token_budget=self.token_budget,
                ratios=self.coord_ratios,
            )
            messages = build_integration_messages(hyp.text, ok_results)
            messages_typed = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

            response = await self.client.chat(
                model_spec=self.client.registry.coordinator,
                messages=messages_typed,
            )

            try:
                data = _extract_json(response.content)
            except ValueError:
                raise ValueError("Integration LLM returned unparseable response")

            decisions = data.get("decisions", [])
            merged_diff = data.get("merged_diff", "")
            summary = data.get("summary", "")

            accepted_ids = [d["subtask_id"] for d in decisions if d.get("decision") == "ACCEPT"]
            rejected_ids = [d["subtask_id"] for d in decisions if d.get("decision") == "REJECT"]

            if merged_diff.strip():
                ok = await apply_diff_to_worktree(merged_diff, integration_path)
                if not ok:
                    # LLM diff failed to apply — fall back to naive merge
                    logger.warning("LLM-merged diff failed to apply; falling back to naive merge")
                    merged_diff, accepted_ids, rejected_ids = await naive_merge(
                        ok_results, integration_path
                    )
                    summary = "Naive sequential merge (LLM diff failed to apply)."
            else:
                # Empty merged diff from LLM — use naive merge
                merged_diff, accepted_ids, rejected_ids = await naive_merge(
                    ok_results, integration_path
                )
                summary = summary or "Naive merge applied."

        except Exception as exc:
            logger.warning("LLM integration failed (%s); falling back to naive merge", exc)
            merged_diff, accepted_ids, rejected_ids = await naive_merge(
                ok_results, integration_path
            )
            summary = "Naive sequential merge."

        # Re-read actual diff from worktree (ground truth)
        actual_diff = await get_worktree_diff(integration_path)
        files_touched: list[str] = []
        for line in actual_diff.splitlines():
            if line.startswith("+++ b/"):
                f = line[6:].strip()
                if f not in files_touched:
                    files_touched.append(f)

        return IntegrationResult(
            hypothesis_id=hyp.id,
            diff=actual_diff,
            files_touched=files_touched,
            summary=summary,
            path=integration_path,
            accepted_subtasks=accepted_ids,
            rejected_subtasks=rejected_ids,
        )

    async def _run_test(self, workspace: str) -> tuple[float, Optional[str]]:
        """Run the opaque test tool once on the integration workspace."""
        result = await self.tools.call("run_tests", caller="coordinator", workspace=workspace)
        if result.success and isinstance(result.value, dict):
            score = float(result.value.get("score", 0.0))
            remark = result.value.get("remark")
            return score, remark
        # Test tool failed to run — score 0
        logger.warning("Test tool error: %s", result.error)
        return 0.0, f"test error: {result.error}"

    async def _maybe_save(
        self,
        integrated: IntegrationResult,
        hyp: Hypothesis,
        score: float,
        iteration: int,
    ) -> bool:
        """Validate and save on score improvement."""
        from tools.validator import validate_diff

        valid, reason = validate_diff(
            diff=integrated.diff,
            repo_path=self.repo_path,
            protected_patterns=self.protected_patterns,
        )
        if not valid:
            logger.warning("Diff failed validation (reward-hack guard): %s", reason)
            await aemit(
                EventType.REWARD_HACK_REJECTED,
                {"reason": reason, "iteration": iteration},
                iteration=iteration,
            )
            return False

        try:
            result = await self.tools.call(
                "save_to_github",
                caller="coordinator",
                diff=integrated.diff,
                repo_path=self.repo_path,
                remote=self.config.get("github_remote", "origin"),
                branch_prefix=self.config.get("github_branch_prefix", "auto-researcher"),
                iteration=iteration,
                meta={"hypothesis": hyp.text, "score": score},
            )
            if result.success:
                await aemit(
                    EventType.SAVED,
                    {"branch": result.value.get("branch", ""), "score": score},
                    iteration=iteration,
                )
                logger.info("Saved to GitHub: %s", result.value)
                return True
            else:
                logger.warning("Save failed: %s", result.error)
                return False
        except Exception as exc:
            logger.warning("Save exception: %s", exc)
            return False

    async def _advance_working_commit(self, worktree_path: str) -> None:
        """Update working_commit to the HEAD of the real target repo after a successful save.

        The integration worktree has no new commit (changes are staged but not committed
        there), so we read HEAD from the real repo where save_to_github committed the diff.
        """
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            self.state.working_commit = stdout.decode().strip()

    async def _cleanup_worktrees(
        self,
        results: list[Any],
        integration_path: str,
    ) -> None:
        """Remove subagent and integration worktrees."""
        # Remove integration worktree
        try:
            await remove_worktree(integration_path, self.repo_path)
        except Exception as exc:
            logger.warning("Could not remove integration worktree %s: %s", integration_path, exc)

        # Note: subagent worktrees were created by Subagent instances;
        # we remove them via the worktree_root directory scan
        worktree_root = Path(self.worktree_root)
        if worktree_root.exists():
            for entry in worktree_root.iterdir():
                if entry.name.startswith("subagent-") and entry.is_dir():
                    try:
                        await remove_worktree(str(entry), self.repo_path)
                    except Exception:
                        pass

    async def _load_state(self) -> None:
        """Load persisted state (baseline, iteration counter, working commit)."""
        saved = self.memory.load_state()
        if saved:
            self.state = saved
            self.baseline = saved.baseline_score
            logger.info(
                "Resumed from state: iteration=%d baseline=%.4f commit=%s",
                self.state.iteration,
                self.baseline,
                self.state.working_commit[:8] if self.state.working_commit else "none",
            )
        else:
            # First run: get HEAD commit
            self.state.working_commit = await self._get_head_commit()

    async def _get_head_commit(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() if proc.returncode == 0 else ""

    async def _drain_and_flush(self) -> None:
        """Clean up on shutdown: flush memory, emit shutdown event."""
        self.memory.save_state(self.state)
        await aemit(EventType.SHUTDOWN, {"iteration": self.state.iteration})
        logger.info("Coordinator shut down cleanly at iteration %d", self.state.iteration)

    def _task_spec(self) -> str:
        prompt_file = self.config.get("system_prompt_file", "system_prompt.txt")
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = Path.cwd() / prompt_path
        if prompt_path.exists():
            goal = prompt_path.read_text().strip()
        else:
            goal = "Continuously improve the codebase's test pass-rate score."
            logger.warning("system_prompt_file not found at %s — using default goal", prompt_path)
        return (
            f"Target repository: {self.repo_path}\n"
            f"Goal:\n{goal}\n"
            f"Constraints:\n"
            f"  - Do not modify test files or held-out evaluation data.\n"
            f"  - All changes must pass the test oracle.\n"
            f"  - Each iteration produces exactly one hypothesis.\n"
        )

    _CODE_EXTS = {
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go",
        ".rs", ".rb", ".swift", ".kt", ".cs", ".md",
    }

    def _load_file_slices(self, max_files: int = 8) -> dict[str, str]:
        """Load the most recently modified code files from the target repo."""
        repo = Path(self.repo_path)
        if not repo.exists():
            return {}
        code_files = sorted(
            [
                f for f in repo.rglob("*")
                if f.is_file()
                and f.suffix in self._CODE_EXTS
                and ".git" not in f.parts
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        slices: dict[str, str] = {}
        for f in code_files[:max_files]:
            try:
                slices[str(f.relative_to(repo))] = f.read_text(
                    encoding="utf-8", errors="replace"
                )[:2000]
            except OSError:
                pass
        return slices

    def _list_repo_files(self, max_files: int = 80) -> str:
        """Return a full directory tree of the target repo (dirs + code files)."""
        repo = Path(self.repo_path)
        if not repo.exists():
            return "(repo not found)"
        lines: list[str] = []
        for path in sorted(repo.rglob("*")):
            if ".git" in path.parts:
                continue
            rel = path.relative_to(repo)
            if path.is_dir():
                lines.append(f"{rel}/")
            elif path.suffix in self._CODE_EXTS:
                lines.append(str(rel))
            if len(lines) >= max_files:
                lines.append("... (truncated)")
                break
        return "\n".join(lines) if lines else "(no code files found)"

    def _format_trajectory(self, last_n: int = 7) -> str:
        """Return a compact score history string for the hypothesis prompt."""
        records = self.memory.get_recent_iterations(last_n)
        if not records:
            return "(no history yet — this is the first iteration)"
        lines = []
        for r in records:  # already sorted newest-first by get_recent_iterations
            arrow = "↑" if r.outcome.value == "win" else "↓"
            lines.append(
                f"  iter {r.iteration:>3}  score={r.score:.4f} {arrow}  {r.hypothesis[:80]}"
            )
        return "\n".join(lines)

    def _format_best_win(self, best_wins: list) -> str:
        """Format the highest-scoring past win for the hypothesis prompt."""
        if not best_wins:
            return "(none yet)"
        top = best_wins[0]
        return f'score={top.score:.4f}: "{top.text}"'

    def pause(self) -> None:
        self.pause_gate.clear()
        logger.info("Coordinator paused")

    def resume(self) -> None:
        self.pause_gate.set()
        logger.info("Coordinator resumed")

    def stop(self) -> None:
        self.stop_requested = True
        self.pause_gate.set()  # unblock if paused so the loop can exit
        logger.info("Coordinator stop requested")

    @property
    def current_hypothesis(self) -> Optional[str]:
        return self._current_hypothesis.text if self._current_hypothesis else None
