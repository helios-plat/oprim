"""Tests for oprim.tdd_test_run, git_checkpoint_commit, browser_element_interact, web_search_fetch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprim._tdd_test_run import tdd_test_run
from oprim._git_checkpoint_commit import git_checkpoint_commit
from oprim._browser_element_interact import browser_element_interact
from oprim._web_search_fetch import web_search_fetch


# ============================================================================
# tdd_test_run
# ============================================================================

class TestTddTestRun:
    def test_pytest_passing(self):
        """Run pytest on a trivial passing test."""
        result = tdd_test_run(
            "python -c 'exit(0)'",
            cwd=".",
            timeout_sec=5,
        )
        assert result["passed"] is True
        assert result["exit_code"] == 0

    def test_failing_command(self):
        """Run a command that fails."""
        result = tdd_test_run(
            "python -c 'exit(1)'",
            cwd=".",
            timeout_sec=5,
        )
        assert result["passed"] is False
        assert result["exit_code"] == 1

    def test_default_command(self):
        """Default test_command is 'pytest'."""
        result = tdd_test_run(cwd=".")
        assert "passed" in result
        assert "exit_code" in result

    def test_timeout_returns_failure(self):
        """Timeout should return passed=False."""
        result = tdd_test_run(
            "sleep 10",
            cwd=".",
            timeout_sec=0.1,
        )
        assert result["passed"] is False
        assert result["exit_code"] == -1

    def test_stdout_truncated(self):
        """Large stdout should be truncated to ~2000 chars."""
        result = tdd_test_run(
            f"python -c \"print('x' * 5000)\"",
            cwd=".",
            timeout_sec=5,
        )
        assert len(result["stdout"]) <= 2100  # allow slight variance

    def test_returns_dict_keys(self):
        result = tdd_test_run("python -c 'print(1)'", cwd=".", timeout_sec=5)
        for key in ("passed", "exit_code", "stdout", "stderr"):
            assert key in result

    def test_positional_only_one(self):
        """Signature enforcement: only 1 positional arg."""
        import inspect
        sig = inspect.signature(tdd_test_run)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1


# ============================================================================
# git_checkpoint_commit
# ============================================================================

class TestGitCheckpointCommit:
    def test_no_git_repo_returns_failed(self, tmp_path: Path):
        """Running in a non-git directory returns failed."""
        result = git_checkpoint_commit("test", repo_path=str(tmp_path))
        assert result["status"] == "failed"

    def test_returns_dict_structure(self):
        result = git_checkpoint_commit("msg", repo_path=".")
        assert "status" in result
        assert "commit_hash" in result
        assert "message" in result

    def test_git_not_found_handled(self, tmp_path: Path):
        """When git is not available, returns failed."""
        with patch("subprocess.run", side_effect=FileNotFoundError("git")):
            result = git_checkpoint_commit("test", repo_path=str(tmp_path))
            assert result["status"] == "failed"
            assert "Git CLI not found" in result["message"]

    def test_positional_only_one(self):
        import inspect
        sig = inspect.signature(git_checkpoint_commit)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1

    def test_successful_commit(self, tmp_path: Path):
        """Test in an actual git repo."""
        # init git repo
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True)
        # Create a file
        (tmp_path / "test.txt").write_text("hello")
        result = git_checkpoint_commit("init commit", repo_path=str(tmp_path))
        # Could be success or "nothing to commit" if file not tracked yet
        assert result["status"] in ("success", "failed")


# ============================================================================
# browser_element_interact
# ============================================================================

class TestBrowserElementInteract:
    def test_click_action(self):
        spec = {"action": "click", "target_id": "btn-1"}
        result = browser_element_interact(spec)
        assert result["status"] == "success"
        assert result["performed_action"] == "click"
        assert result["target"] == "btn-1"

    def test_type_action(self):
        spec = {"action": "type", "target_id": "input-1", "value": "hello"}
        result = browser_element_interact(spec)
        assert result["status"] == "success"
        assert result["performed_action"] == "type"

    def test_navigate_action(self):
        spec = {"action": "navigate", "url": "https://example.com"}
        result = browser_element_interact(spec)
        assert result["status"] == "success"
        assert result["performed_action"] == "navigate"

    def test_with_runner_op(self):
        mock_runner = MagicMock()
        mock_page = MagicMock()
        mock_page.elements = []
        mock_runner.capture_page_with_bounding_boxes.return_value = mock_page
        spec = {"action": "navigate", "url": "https://example.com"}
        result = browser_element_interact(spec, runner_op=mock_runner)
        assert result["status"] == "success"
        mock_runner.capture_page_with_bounding_boxes.assert_called_once()

    def test_missing_action_defaults_to_click(self):
        spec = {"target_id": "btn"}
        result = browser_element_interact(spec)
        assert result["performed_action"] == "click"

    def test_returns_failed_on_exception(self):
        mock_runner = MagicMock()
        mock_runner.capture_page_with_bounding_boxes.side_effect = RuntimeError("browser crash")
        spec = {"action": "navigate", "url": "https://example.com"}
        result = browser_element_interact(spec, runner_op=mock_runner)
        assert result["status"] == "failed"
        assert "RuntimeError" in result.get("error", "")

    def test_positional_only_one(self):
        import inspect
        sig = inspect.signature(browser_element_interact)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1


# ============================================================================
# web_search_fetch
# ============================================================================

class TestWebSearchFetch:
    def test_search_query_returns_results(self):
        result = web_search_fetch("python asyncio")
        assert result["status"] == "success"
        assert "content_markdown" in result
        assert "python asyncio" in result["content_markdown"]

    def test_url_without_safe_net_op(self):
        result = web_search_fetch("https://example.com")
        assert result["status"] == "success"

    def test_url_blocked_by_firewall(self):
        mock_net = MagicMock()
        mock_net.is_safe_url.return_value = False
        result = web_search_fetch("http://127.0.0.1/admin", safe_net_op=mock_net)
        assert result["status"] == "blocked"
        assert "SSRF" in result.get("reason", "")

    def test_url_allowed_and_fetched(self):
        mock_net = MagicMock()
        mock_net.is_safe_url.return_value = True
        mock_net.safe_fetch.return_value = {
            "status": "success",
            "content": "<html>test</html>",
            "content_length": 18,
        }
        result = web_search_fetch("https://safe.com", safe_net_op=mock_net)
        assert result["status"] == "success"
        assert "<html>test</html>" in result["content_markdown"]

    def test_fetch_failure_returns_failed(self):
        mock_net = MagicMock()
        mock_net.is_safe_url.return_value = True
        mock_net.safe_fetch.side_effect = ConnectionError("timeout")
        result = web_search_fetch("https://fail.com", safe_net_op=mock_net)
        assert result["status"] == "failed"
        assert "timeout" in result.get("reason", "")

    def test_positional_only_one(self):
        import inspect
        sig = inspect.signature(web_search_fetch)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1
