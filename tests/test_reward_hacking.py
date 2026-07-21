"""Tests for the reward-hacking diff validator."""

from __future__ import annotations

import pytest

from tools.validator import extract_touched_files, validate_diff

_PROTECTED = ["tests/", "test/", "held_out/", "eval/", "benchmark/", "*.test.*"]

_GOOD_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 x = 1
+y = 2
 z = 3
"""

_TESTS_DIFF = """\
--- a/tests/test_foo.py
+++ b/tests/test_foo.py
@@ -1,3 +1,4 @@
 x = 1
+y = 2
"""

_MIXED_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 x = 1
+y = 2
--- a/tests/test_foo.py
+++ b/tests/test_foo.py
@@ -1,2 +1,3 @@
 a = 1
+b = 2
"""

_NO_HUNK_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
+y = 2
"""

_HELD_OUT_DIFF = """\
--- a/held_out/data.json
+++ b/held_out/data.json
@@ -1,1 +1,1 @@
-{}
+{"key": "val"}
"""

_BENCHMARK_DIFF = """\
--- a/benchmark/run.py
+++ b/benchmark/run.py
@@ -1,2 +1,3 @@
 import time
+import sys
"""

_DOT_TEST_DIFF = """\
--- a/src/foo.test.py
+++ b/src/foo.test.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
"""


def test_valid_diff_passes():
    ok, reason = validate_diff(_GOOD_DIFF, ".", _PROTECTED)
    assert ok is True
    assert reason == ""


def test_empty_diff_invalid():
    ok, reason = validate_diff("", ".", _PROTECTED)
    assert ok is False
    assert "empty" in reason


def test_whitespace_only_diff_invalid():
    ok, reason = validate_diff("   \n  ", ".", _PROTECTED)
    assert ok is False


def test_diff_touching_tests_dir_invalid():
    ok, reason = validate_diff(_TESTS_DIFF, ".", _PROTECTED)
    assert ok is False
    assert "tests/" in reason or "protected" in reason


def test_diff_touching_both_src_and_tests_invalid():
    ok, reason = validate_diff(_MIXED_DIFF, ".", _PROTECTED)
    assert ok is False


def test_diff_missing_hunk_headers_invalid():
    ok, reason = validate_diff(_NO_HUNK_DIFF, ".", _PROTECTED)
    assert ok is False
    assert "hunk" in reason.lower() or "parseable" in reason.lower()


def test_diff_touching_held_out_invalid():
    ok, reason = validate_diff(_HELD_OUT_DIFF, ".", _PROTECTED)
    assert ok is False


def test_diff_touching_benchmark_invalid():
    ok, reason = validate_diff(_BENCHMARK_DIFF, ".", _PROTECTED)
    assert ok is False


def test_diff_touching_dot_test_file_invalid():
    ok, reason = validate_diff(_DOT_TEST_DIFF, ".", _PROTECTED)
    assert ok is False


def test_extract_touched_files_basic():
    files = extract_touched_files(_GOOD_DIFF)
    assert files == ["src/foo.py"]


def test_extract_touched_files_skips_dev_null():
    diff = """\
--- /dev/null
+++ b/src/new_file.py
@@ -0,0 +1,2 @@
+x = 1
"""
    files = extract_touched_files(diff)
    assert "src/new_file.py" in files
    assert "/dev/null" not in files


def test_extract_touched_files_multiple():
    files = extract_touched_files(_MIXED_DIFF)
    assert "src/foo.py" in files
    assert "tests/test_foo.py" in files
    assert len(files) == 2


def test_empty_protected_patterns_uses_defaults():
    # With no custom patterns, defaults still apply
    ok, reason = validate_diff(_TESTS_DIFF, ".", [])
    # Default protected includes "tests/"
    assert ok is False
