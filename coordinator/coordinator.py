"""Coordinator: the infinite hypothesis loop with an inner fix loop per hypothesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from server.events import EventType, aemit
from shared.types import (
    AgentState,
    ExplorationResult,
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

# ── Hypothesis formation ──────────────────────────────────────────────────────

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

## Project Understanding (from explorer subagents)

### Architecture
{arch}

### Test Mechanism
{test_structure}

### Key Patterns / Bottlenecks
{patterns}

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

# ── Project exploration ───────────────────────────────────────────────────────

EXPLORE_SYSTEM = """You are a software architecture analyst. Read the provided project files carefully and answer the specific question below. Be concise, specific, and technical. Output plain text."""

EXPLORE_ARCH_Q = """Analyze this project's architecture. Identify:
1. Main modules/files and their responsibilities
2. Key data structures and data flows
3. Entry points and major execution paths
4. How modules depend on each other

Repository file listing:
{file_listing}

File contents:
{file_contents}"""

EXPLORE_TESTS_Q = """Analyze this project's test and evaluation structure. Identify:
1. What the test oracle measures (pass rate, accuracy, score, etc.)
2. Which files are test-related and must NOT be modified
3. How the score is computed and what "improvement" means
4. Any held-out data or benchmark files

Repository file listing:
{file_listing}

File contents:
{file_contents}"""

EXPLORE_PATTERNS_Q = """Analyze this project's algorithms, patterns, and improvement opportunities. Identify:
1. The primary algorithms and their computational complexity
2. Obvious performance bottlenecks or correctness issues
3. Code quality issues (duplication, poor abstractions, etc.)
4. Concrete areas most likely to yield a score improvement and why

Repository file listing:
{file_listing}

File contents:
{file_contents}"""

# ── Coordinator decision ──────────────────────────────────────────────────────

COORDINATOR_DECIDE_SYSTEM = """You are coordinating an AI research agent. A hypothesis was implemented and tested but failed to beat the baseline.

Decide whether to CONTINUE fixing this hypothesis (same hypothesis, new implementation attempt) or ACCEPT the result and move on to a new hypothesis.

CONTINUE when: the error is a runtime exception, syntax error, import failure, or other technical bug that a targeted fix can resolve — the hypothesis idea itself is sound.
ACCEPT when: the hypothesis scored higher than baseline (any improvement counts), OR the hypothesis idea is logically flawed, OR the error suggests a fundamental design mismatch that fresh retries won't fix.

Output JSON only: {"decision": "CONTINUE" | "ACCEPT", "reason": "..."}"""

COORDINATOR_DECIDE_USER = """Hypothesis: {hypothesis}

Test result: score={score:.4f}  baseline={baseline:.4f}
Test remark: {remark}

Inner loop attempts so far: {attempt}

Should we CONTINUE fixing this hypothesis or ACCEPT this result and move on?
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
        self.max_inner_iterations: int = config.get("max_inner_iterations", 20)
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
            await self.pause_gate.wait()
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
                await asyncio.sleep(5)

        await self._drain_and_flush()

    async def _run_one_iteration(self) -> None:
        n = self.state.iteration
        logger.info("━━━ Iteration %d  (baseline=%.4f) ━━━", n, self.baseline)
        self._iteration_rolling_summary = ""

        # 0. Establish baseline on first iteration
        if self.baseline == 0.0:
            logger.info("[%d] Measuring baseline on original repo…", n)
            await aemit(
                EventType.TEST_SCORED,
                {"score": 0.0, "remark": "Measuring baseline…", "baseline": 0.0},
                iteration=n,
            )
            real_baseline, bl_remark = await self._run_test(self.repo_path)
            if real_baseline > 0.0:
                self.baseline = real_baseline
                self.state.baseline_score = real_baseline
                self.memory.save_state(self.state)
                logger.info("[%d] Baseline established: %.4f", n, real_baseline)
                await aemit(EventType.LOOP_STARTED, {"baseline": self.baseline, "iteration": n})
            else:
                logger.warning("[%d] Baseline score=0 — check test.py. %s", n, bl_remark)

        # 1. Explore project architecture via parallel explorer subagents
        logger.info("[%d] Spawning explorer subagents…", n)
        exploration = await self._explore_project(n)

        # 2. Form hypothesis (informed by exploration)
        logger.info("[%d] Forming hypothesis…", n)
        hyp = await self.form_hypothesis(exploration=exploration)
        self._current_hypothesis = hyp
        logger.info("[%d] Hypothesis: %s", n, hyp.text)

        # 3. Anti-repetition gate
        is_dup = await self.memory.is_duplicate_failure(hyp.text)
        if is_dup:
            building_on_win = await self.memory.is_building_on_win(hyp.text)
            if building_on_win:
                logger.info("[%d] Resembles past failure but also a win — allowing as incremental improvement", n)
            else:
                logger.info("[%d] Duplicate failure — reforming with novelty boost", n)
                await aemit(EventType.DUP_REJECTED, {"hypothesis": hyp.text}, iteration=n)
                hyp = await self.reform_with_novelty(hyp)
                self._current_hypothesis = hyp
                logger.info("[%d] Novel hypothesis: %s", n, hyp.text)

        # 4. Inner loop: implement → integrate → test → coordinator decides
        #    Runs until coordinator is satisfied (success or giving up on this hypothesis).
        inner_attempt = 0
        score: float = 0.0
        remark: Optional[str] = None
        integrated: Optional[IntegrationResult] = None
        last_ok_results: list[SubtaskResult] = []

        while not self.stop_requested:
            await self.pause_gate.wait()

            logger.info("[%d] Inner attempt %d — decomposing hypothesis…", n, inner_attempt)

            # 4a. Decompose (passes failure context on retries so coordinator picks different approach)
            briefs = await self.decompose(
                hyp,
                exploration=exploration,
                fix_attempt=inner_attempt,
                prev_remark=remark,
            )
            for b in briefs:
                model_spec, matched, fallback = self.router.select(b.required_skills)
                b.model = model_spec
                b.matched_skills = matched
                b.fallback = fallback
                await aemit(
                    EventType.MODEL_ROUTED,
                    {"subtask_id": b.id, "model": model_spec.name,
                     "matched_skills": matched, "fallback": fallback},
                    iteration=n,
                )
            logger.info("[%d] %d subtask(s): %s", n, len(briefs),
                        " | ".join(b.goal[:60] for b in briefs))

            # 4b. Dispatch subagents (each runs its own read/edit/debug ReAct loop)
            logger.info("[%d] Dispatching %d subagent(s) (inner attempt %d)…",
                        n, len(briefs), inner_attempt)
            results = await self._dispatch_subagents(briefs)
            ok_results = [r for r in results if r is not None and not isinstance(r, BaseException)]
            failed_count = len(results) - len(ok_results)
            last_ok_results = ok_results
            logger.info("[%d] Subagents done — %d succeeded, %d failed",
                        n, len(ok_results), failed_count)

            # 4c. Integrate all subagent diffs
            logger.info("[%d] Integrating diffs…", n)
            if integrated is not None:
                # Clean previous integration worktree before creating a new one
                try:
                    await remove_worktree(integrated.path, self.repo_path)
                except Exception as exc:
                    logger.debug("Could not remove previous integration worktree: %s", exc)
            integrated = await self.review_and_integrate(hyp, ok_results)
            await aemit(
                EventType.REVIEW_INTEGRATE,
                {"hypothesis": hyp.text, "diff_len": len(integrated.diff),
                 "inner_attempt": inner_attempt},
                iteration=n,
            )
            logger.info("[%d] Integration done — diff: %d lines, files: %s",
                        n, integrated.diff.count("\n"),
                        ", ".join(integrated.files_touched) or "none")

            # 4d. Test hypothesis
            logger.info("[%d] Testing (inner attempt %d)…", n, inner_attempt)
            score, remark = await self._run_test(integrated.path)
            await aemit(
                EventType.TEST_SCORED,
                {"score": score, "remark": remark, "baseline": self.baseline,
                 "inner_attempt": inner_attempt},
                iteration=n,
            )
            logger.info("[%d] Inner %d → score=%.4f  baseline=%.4f  %s",
                        n, inner_attempt, score, self.baseline,
                        "↑ WIN" if score > self.baseline else "↓ below baseline")

            # 4e. Coordinator decides: satisfied or continue inner loop?
            satisfied = await self._coordinator_decide(hyp, score, remark, inner_attempt)

            if satisfied:
                await aemit(
                    EventType.INNER_LOOP_EXIT,
                    {"inner_attempt": inner_attempt, "score": score,
                     "reason": "coordinator_accepted"},
                    iteration=n,
                )
                logger.info("[%d] Coordinator accepted result after %d inner attempt(s)",
                            n, inner_attempt + 1)
                break

            # Safety net: don't spin forever if the LLM keeps saying CONTINUE
            inner_attempt += 1
            if inner_attempt >= self.max_inner_iterations:
                logger.warning("[%d] Inner loop safety cap (%d) reached — accepting result",
                               n, self.max_inner_iterations)
                await aemit(
                    EventType.INNER_LOOP_EXIT,
                    {"inner_attempt": inner_attempt, "score": score,
                     "reason": "safety_cap"},
                    iteration=n,
                )
                break

            await aemit(
                EventType.INNER_LOOP_CONTINUE,
                {"inner_attempt": inner_attempt, "score": score,
                 "hypothesis": hyp.text[:80]},
                iteration=n,
            )
            logger.info("[%d] Inner loop continuing — attempt %d", n, inner_attempt)

            # Clean up subagent worktrees before next inner attempt
            await self._cleanup_subagent_worktrees(results)

        # 5. Record outcome
        outcome = OutcomeType.WIN if score > self.baseline else OutcomeType.MISTAKE
        record = IterationRecord(
            id=str(uuid.uuid4()),
            hypothesis=hyp.text,
            integrated_diff_hash=hashlib.sha256(
                (integrated.diff if integrated else "").encode()
            ).hexdigest()[:16],
            subagent_contribs=[
                {"id": r.subtask_id, "status": r.status.value, "files": r.files_touched}
                for r in last_ok_results
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

        # 6. Git checkpoint on improvement
        if score > self.baseline and integrated and integrated.diff.strip():
            saved = await self._maybe_save(integrated, hyp, score, n)
            if saved:
                self.baseline = score
                await self._advance_working_commit(integrated.path)

        # 7. Cleanup all worktrees for this iteration
        if integrated:
            await self._cleanup_worktrees(last_ok_results, integrated.path)

        self.state.iteration += 1
        self.state.baseline_score = self.baseline
        self.memory.save_state(self.state)
        logger.info("[%d] Iteration complete — outcome=%s  score=%.4f  new_baseline=%.4f",
                    n, outcome.value.upper(), score, self.baseline)

    # ── Exploration ──────────────────────────────────────────────────────────

    async def _explore_project(self, n: int) -> ExplorationResult:
        """Spawn 3 parallel explorer subagents to understand the target project."""
        await aemit(
            EventType.EXPLORATION_STARTED,
            {"iteration": n, "aspects": ["architecture", "tests", "patterns"]},
            iteration=n,
        )

        file_listing = self._list_repo_files(max_files=200)
        file_slices = self._load_file_slices(max_files=25, chars_per_file=3000)
        file_contents = "\n\n".join(
            f"### {p}\n```\n{c}\n```" for p, c in file_slices.items()
        ) or "(no file contents available)"

        arch_task = self._explore_aspect(EXPLORE_ARCH_Q, file_listing, file_contents)
        tests_task = self._explore_aspect(EXPLORE_TESTS_Q, file_listing, file_contents)
        patterns_task = self._explore_aspect(EXPLORE_PATTERNS_Q, file_listing, file_contents)

        arch, tests, patterns = await asyncio.gather(arch_task, tests_task, patterns_task)

        result = ExplorationResult(
            architecture=arch,
            test_structure=tests,
            key_patterns=patterns,
            key_files=list(file_slices.keys())[:15],
        )
        await aemit(
            EventType.EXPLORATION_DONE,
            {"iteration": n, "key_files": result.key_files[:5]},
            iteration=n,
        )
        logger.info("[%d] Exploration done — arch=%d chars, tests=%d chars, patterns=%d chars",
                    n, len(arch), len(tests), len(patterns))
        return result

    async def _explore_aspect(
        self, question_template: str, file_listing: str, file_contents: str
    ) -> str:
        """Single explorer subagent: answer one architectural question about the project."""
        from models.client import ChatMessage

        question = question_template.format(
            file_listing=file_listing,
            file_contents=file_contents[:6000],
        )
        messages = [
            ChatMessage(role="system", content=EXPLORE_SYSTEM),
            ChatMessage(role="user", content=question),
        ]
        try:
            response = await self.client.chat(
                model_spec=self.client.registry.coordinator,
                messages=messages,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Explorer subagent failed: %s", exc)
            return f"(exploration failed: {exc})"

    # ── Coordinator decision ─────────────────────────────────────────────────

    async def _coordinator_decide(
        self,
        hyp: Hypothesis,
        score: float,
        remark: Optional[str],
        inner_attempt: int,
    ) -> bool:
        """Returns True if the coordinator is satisfied (exit inner loop).

        Auto-accepts on any score improvement. On failure, asks LLM only after
        the first attempt (first failure always retries automatically).
        """
        if score > self.baseline:
            return True  # any improvement → accept and checkpoint

        if inner_attempt == 0:
            return False  # always give at least one retry on first failure

        # Ask the coordinator model whether to continue fixing or give up
        from models.client import ChatMessage

        messages = [
            ChatMessage(role="system", content=COORDINATOR_DECIDE_SYSTEM),
            ChatMessage(
                role="user",
                content=COORDINATOR_DECIDE_USER.format(
                    hypothesis=hyp.text,
                    score=score,
                    baseline=self.baseline,
                    remark=remark or "(no remark)",
                    attempt=inner_attempt,
                ),
            ),
        ]
        try:
            response = await self.client.chat(
                model_spec=self.client.registry.coordinator,
                messages=messages,
            )
            data = _extract_json(response.content)
            decision = data.get("decision", "ACCEPT").upper()
            reason = data.get("reason", "")
            logger.info("Coordinator decision: %s — %s", decision, reason[:120])
            return decision == "ACCEPT"
        except Exception as exc:
            logger.warning("Coordinator decision LLM failed (%s) — defaulting to CONTINUE", exc)
            return False  # default: keep trying

    # ── Hypothesis formation ─────────────────────────────────────────────────

    async def form_hypothesis(
        self, exploration: Optional[ExplorationResult] = None
    ) -> Hypothesis:
        """Assemble coordinator context and ask the model for one hypothesis."""
        n = self.state.iteration

        wins = await self.memory.retrieve("improve test score", k=5, include_failures=False)
        failures = await self.memory.top_failures("improvement hypothesis", k=5)
        best_wins = await self.memory.get_best_wins(k=3)
        file_slices = self._load_file_slices(max_files=20, chars_per_file=4000)

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

        arch = exploration.architecture[:600] if exploration else "(not available)"
        test_structure = exploration.test_structure[:400] if exploration else "(not available)"
        patterns = exploration.key_patterns[:400] if exploration else "(not available)"

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
                    arch=arch,
                    test_structure=test_structure,
                    patterns=patterns,
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

    # ── Decomposition ────────────────────────────────────────────────────────

    async def decompose(
        self,
        hyp: Hypothesis,
        exploration: Optional[ExplorationResult] = None,
        fix_attempt: int = 0,
        prev_remark: Optional[str] = None,
    ) -> list[SubtaskBrief]:
        """Decompose hypothesis into subtask briefs, aware of exploration and any prior failure."""
        from models.client import ChatMessage

        file_listing = self._list_repo_files(max_files=200)
        file_slices = self._load_file_slices(max_files=20, chars_per_file=1500)
        messages_raw = make_decompose_messages(
            hypothesis=hyp,
            repo_path=self.repo_path,
            file_listing=file_listing,
            file_slices=file_slices,
            max_subagents=self.max_subagents,
            exploration=exploration,
            fix_attempt=fix_attempt,
            prev_remark=prev_remark,
        )
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_raw]

        response = await self.client.chat(
            model_spec=self.client.registry.coordinator,
            messages=messages,
        )

        try:
            decomposition = _extract_json(response.content)
        except ValueError:
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
            {"hypothesis": hyp.text, "n_subtasks": len(briefs), "fix_attempt": fix_attempt},
            iteration=hyp.iteration,
        )
        return briefs

    # ── Subagent dispatch ────────────────────────────────────────────────────

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

        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.warning("Subagent %s raised exception: %s", briefs[i].id, r)

        return list(results)

    # ── Integration ──────────────────────────────────────────────────────────

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

            if file_slices:
                file_context = "\n\n## Current File State (for resolving conflicts)\n"
                file_context += "\n".join(
                    f"### {p}\n```\n{c[:800]}\n```" for p, c in file_slices.items()
                )
                messages_typed[-1] = ChatMessage(
                    role=messages_typed[-1].role,
                    content=messages_typed[-1].content + file_context,
                )

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
                    logger.warning("LLM-merged diff failed to apply; falling back to naive merge")
                    merged_diff, accepted_ids, rejected_ids = await naive_merge(
                        ok_results, integration_path
                    )
                    summary = "Naive sequential merge (LLM diff failed to apply)."
            else:
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

    # ── Test runner ──────────────────────────────────────────────────────────

    async def _run_test(self, workspace: str) -> tuple[float, Optional[str]]:
        """Run the opaque test tool once on the integration workspace."""
        result = await self.tools.call("run_tests", caller="coordinator", workspace=workspace)
        if result.success and isinstance(result.value, dict):
            score = float(result.value.get("score", 0.0))
            remark = result.value.get("remark")
            return score, remark
        logger.warning("Test tool error: %s", result.error)
        return 0.0, f"test error: {result.error}"

    # ── Save / checkpoint ────────────────────────────────────────────────────

    async def _maybe_save(
        self,
        integrated: IntegrationResult,
        hyp: Hypothesis,
        score: float,
        iteration: int,
    ) -> bool:
        """Validate and save to git on score improvement."""
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
                branch = result.value.get("branch", "") if isinstance(result.value, dict) else ""
                await aemit(
                    EventType.SAVED,
                    {"branch": branch, "score": score},
                    iteration=iteration,
                )
                logger.info("Saved to git: %s", result.value)
                return True
            else:
                logger.warning("Save failed: %s", result.error)
                return False
        except Exception as exc:
            logger.warning("Save exception: %s", exc)
            return False

    async def _advance_working_commit(self, worktree_path: str) -> None:
        """Update working_commit to HEAD of the real repo after a successful save."""
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            self.state.working_commit = stdout.decode().strip()

    # ── Cleanup ──────────────────────────────────────────────────────────────

    async def _cleanup_subagent_worktrees(
        self, results: list[Any]
    ) -> None:
        """Remove only the subagent worktrees (keep integration worktree alive)."""
        worktree_root = Path(self.worktree_root)
        if worktree_root.exists():
            for entry in worktree_root.iterdir():
                if entry.name.startswith("subagent-") and entry.is_dir():
                    try:
                        await remove_worktree(str(entry), self.repo_path)
                    except Exception:
                        pass

    async def _cleanup_worktrees(
        self,
        results: list[Any],
        integration_path: str,
    ) -> None:
        """Remove both subagent worktrees and the integration worktree."""
        try:
            await remove_worktree(integration_path, self.repo_path)
        except Exception as exc:
            logger.warning("Could not remove integration worktree %s: %s", integration_path, exc)

        await self._cleanup_subagent_worktrees(results)

    # ── State management ─────────────────────────────────────────────────────

    async def _load_state(self) -> None:
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
        self.memory.save_state(self.state)
        await aemit(EventType.SHUTDOWN, {"iteration": self.state.iteration})
        logger.info("Coordinator shut down cleanly at iteration %d", self.state.iteration)

    # ── Context helpers ──────────────────────────────────────────────────────

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
        ".rs", ".rb", ".swift", ".kt", ".cs", ".md", ".yaml", ".toml",
    }

    _SKIP_DIRS = {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        "site-packages", ".eggs", "*.egg-info",
    }

    def _is_project_file(self, path: Path, repo: Path) -> bool:
        return not any(part in self._SKIP_DIRS for part in path.relative_to(repo).parts)

    def _load_file_slices(self, max_files: int = 20, chars_per_file: int = 4000) -> dict[str, str]:
        repo = Path(self.repo_path)
        if not repo.exists():
            return {}
        code_files = sorted(
            [
                f for f in repo.rglob("*")
                if f.is_file()
                and f.suffix in self._CODE_EXTS
                and self._is_project_file(f, repo)
            ],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        slices: dict[str, str] = {}
        for f in code_files[:max_files]:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                rel = str(f.relative_to(repo))
                slices[rel] = content[:chars_per_file] + (
                    f"\n... [truncated at {chars_per_file} chars]"
                    if len(content) > chars_per_file else ""
                )
            except OSError:
                pass
        return slices

    def _list_repo_files(self, max_files: int = 200) -> str:
        repo = Path(self.repo_path)
        if not repo.exists():
            return "(repo not found)"
        lines: list[str] = []
        for path in sorted(repo.rglob("*")):
            if not self._is_project_file(path, repo):
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
        records = self.memory.get_recent_iterations(last_n)
        if not records:
            return "(no history yet — this is the first iteration)"
        lines = []
        for r in records:
            arrow = "↑" if r.outcome.value == "win" else "↓"
            lines.append(
                f"  iter {r.iteration:>3}  score={r.score:.4f} {arrow}  {r.hypothesis[:80]}"
            )
        return "\n".join(lines)

    def _format_best_win(self, best_wins: list) -> str:
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
        self.pause_gate.set()
        logger.info("Coordinator stop requested")

    @property
    def current_hypothesis(self) -> Optional[str]:
        return self._current_hypothesis.text if self._current_hypothesis else None
