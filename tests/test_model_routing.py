"""Tests for skill-based model routing and registry validation."""

from __future__ import annotations

import pytest

from models.registry import ModelRegistry
from models.router import ModelRouter
from shared.types import ModelSpec


def _make_registry(workers: list[dict] | None = None) -> ModelRegistry:
    """Build a ModelRegistry without calling validate() (no Ollama needed)."""
    config = {
        "models": {
            "coordinator": "coordinator-model",
            "default": "default-model",
            "embed": "embed-model",
            "workers": workers if workers is not None else [],
        }
    }
    return ModelRegistry(config)


# ── Registry construction ─────────────────────────────────────────────────────

def test_registry_missing_coordinator_raises():
    with pytest.raises(ValueError, match="coordinator"):
        ModelRegistry({"models": {"default": "x"}})


def test_registry_missing_default_raises():
    with pytest.raises(ValueError, match="default"):
        ModelRegistry({"models": {"coordinator": "x"}})


def test_registry_workers_absent_returns_empty():
    reg = _make_registry(workers=[])
    assert reg.workers == []


def test_registry_workers_parsed_correctly():
    reg = _make_registry(workers=[
        {"name": "math-model", "skills": ["math", "proof"], "options": {"temperature": 0.1}},
        {"name": "code-model", "skills": ["code"]},
    ])
    assert len(reg.workers) == 2
    assert reg.workers[0].name == "math-model"
    assert reg.workers[0].skills == ["math", "proof"]
    assert reg.workers[0].options == {"temperature": 0.1}
    assert reg.workers[1].name == "code-model"
    assert reg.workers[1].options == {}


def test_registry_coordinator_and_default_properties():
    reg = _make_registry()
    assert reg.coordinator.name == "coordinator-model"
    assert reg.default.name == "default-model"
    assert reg.embed_model == "embed-model"


# ── Router.select ─────────────────────────────────────────────────────────────

def test_exact_skill_match_returns_specialist():
    reg = _make_registry(workers=[
        {"name": "math-model", "skills": ["math", "proof"]},
        {"name": "code-model", "skills": ["code", "refactor"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["math"])
    assert spec.name == "math-model"
    assert "math" in matched
    assert fallback is False


def test_best_overlap_wins():
    reg = _make_registry(workers=[
        {"name": "partial-math", "skills": ["math"]},
        {"name": "full-math", "skills": ["math", "proof", "numeric"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["math", "proof", "numeric"])
    assert spec.name == "full-math"
    assert len(matched) == 3
    assert fallback is False


def test_no_skill_overlap_returns_default():
    reg = _make_registry(workers=[
        {"name": "math-model", "skills": ["math"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["code"])
    assert spec.name == "default-model"
    assert matched == []
    assert fallback is True


def test_empty_required_skills_returns_default():
    reg = _make_registry(workers=[
        {"name": "math-model", "skills": ["math"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select([])
    assert spec.name == "default-model"
    assert fallback is True


def test_no_workers_configured_returns_default():
    reg = _make_registry(workers=[])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["code", "math"])
    assert spec.name == "default-model"
    assert fallback is True


def test_tie_broken_by_registry_order():
    reg = _make_registry(workers=[
        {"name": "first-model", "skills": ["code"]},
        {"name": "second-model", "skills": ["code"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["code"])
    assert spec.name == "first-model"


def test_multiple_skills_partial_match_picks_best():
    reg = _make_registry(workers=[
        {"name": "one-skill", "skills": ["code"]},
        {"name": "two-skills", "skills": ["code", "refactor"]},
    ])
    router = ModelRouter(reg)
    spec, matched, fallback = router.select(["code", "refactor", "debug"])
    assert spec.name == "two-skills"
    assert set(matched) == {"code", "refactor"}


def test_router_never_raises_on_strange_input():
    reg = _make_registry(workers=[])
    router = ModelRouter(reg)
    spec, _, _ = router.select(["a", "b", "c", "d", "e"])
    assert spec.name == "default-model"


def test_worker_model_spec_has_correct_options():
    reg = _make_registry(workers=[
        {"name": "precise", "skills": ["math"], "options": {"temperature": 0.1, "seed": 42}},
    ])
    router = ModelRouter(reg)
    spec, _, _ = router.select(["math"])
    assert spec.options["temperature"] == 0.1
    assert spec.options["seed"] == 42
