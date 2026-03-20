"""Tests for otter.role module."""

from pathlib import Path

import pytest

from otter.episode import Episode, InputManifest, OutputManifest, Turn
from otter.role import (
    extract_for_chat_llm,
    extract_for_docker,
    pack_chat_llm,
    pack_docker,
    EXTRACT_DISPATCH,
    PACK_DISPATCH,
    BaseRole,
    ProposerRole,
    ExecutorRole,
    EvaluatorRole,
)
from otter.backend import Result, ChatLLMRunResult, DockerRunResult


class TestExtractForChatLLM:
    """Test extract_for_chat_llm function."""

    def test_single_turn_extracts_prompt(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Hello, solve this problem", encoding="utf-8")

        manifest = InputManifest(prompt_file=prompt_file)
        turn = Turn(turn_dir=tmp_path, exec_input_manifest=manifest)
        ep = Episode(task_id="t", sample_id=0, turns=[turn])

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = extract_for_chat_llm(manifest, ep, output_dir)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, solve this problem"
        assert result["output_file"] == output_dir / "response.txt"

    def test_multi_turn_builds_history(self, tmp_path):
        """Should include history from previous turns."""
        # Turn 1: prompt + response
        t1_dir = tmp_path / "turn_1"
        t1_dir.mkdir()
        t1_prompt = t1_dir / "prompt.txt"
        t1_prompt.write_text("First prompt", encoding="utf-8")
        t1_response = t1_dir / "response.txt"
        t1_response.write_text("First response", encoding="utf-8")

        turn1 = Turn(
            turn_dir=t1_dir,
            exec_input_manifest=InputManifest(prompt_file=t1_prompt),
            exec_output_manifest=OutputManifest(exec_output_file=t1_response),
        )

        # Turn 2: current prompt
        t2_dir = tmp_path / "turn_2"
        t2_dir.mkdir()
        t2_prompt = t2_dir / "prompt.txt"
        t2_prompt.write_text("Second prompt", encoding="utf-8")

        current_manifest = InputManifest(prompt_file=t2_prompt)
        turn2 = Turn(turn_dir=t2_dir, exec_input_manifest=current_manifest)

        ep = Episode(task_id="t", sample_id=0, turns=[turn1, turn2])

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = extract_for_chat_llm(current_manifest, ep, output_dir)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "First prompt"}
        assert messages[1] == {"role": "assistant", "content": "First response"}
        assert messages[2] == {"role": "user", "content": "Second prompt"}


class TestExtractForDocker:
    """Test extract_for_docker function."""

    def test_basic_extraction(self, tmp_path):
        manifest = InputManifest(
            image_tag="test:latest",
            commands=["echo hello"],
        )
        ep = Episode(task_id="t", sample_id=0)
        result = extract_for_docker(manifest, ep, tmp_path)
        assert result["image_tag"] == "test:latest"
        assert result["commands"] == ["echo hello"]
        assert "copy_in" not in result

    def test_with_script_file(self, tmp_path):
        script = tmp_path / "script.py"
        script.write_text("print('hi')")
        manifest = InputManifest(
            image_tag="test:latest",
            script_file=script,
            commands=["python /tmp/script.py"],
        )
        ep = Episode(task_id="t", sample_id=0)
        result = extract_for_docker(manifest, ep, tmp_path)
        assert result["copy_in"] == [(script, "/tmp")]

    def test_with_timeout(self, tmp_path):
        manifest = InputManifest(
            image_tag="test:latest",
            commands=["sleep 100"],
            timeout=30,
        )
        ep = Episode(task_id="t", sample_id=0)
        result = extract_for_docker(manifest, ep, tmp_path)
        assert result["timeout"] == 30

    def test_none_commands_defaults_to_empty_list(self, tmp_path):
        manifest = InputManifest(image_tag="test:latest", commands=None)
        ep = Episode(task_id="t", sample_id=0)
        result = extract_for_docker(manifest, ep, tmp_path)
        assert result["commands"] == []


class TestPackChatLLM:
    """Test pack_chat_llm function."""

    def test_successful_result(self, tmp_path):
        output_file = tmp_path / "response.txt"
        output_file.write_text("Hello world response", encoding="utf-8")
        run_result = ChatLLMRunResult(
            products=[output_file],
            retries=[Result(stdout="Hello world response", stderr="", returncode=0, timed_out=False)],
        )
        result = pack_chat_llm(run_result, tmp_path)
        assert isinstance(result, OutputManifest)
        assert result.exec_output_file == output_file

    def test_failed_result(self, tmp_path):
        run_result = ChatLLMRunResult(
            products=[],
            retries=[Result(stdout="", stderr="API error", returncode=-1, timed_out=False)],
            error="ChatLLMBackend failed after 2 attempts",
        )
        result = pack_chat_llm(run_result, tmp_path)
        assert result.exec_output_file is None


class TestPackDocker:
    """Test pack_docker function."""

    def test_writes_stdout_stderr_files(self, tmp_path):
        docker_result = DockerRunResult(
            copy_in=[],
            commands=[Result(stdout="output text", stderr="error text", returncode=0, timed_out=False)],
            copy_out=[],
        )
        result = pack_docker(docker_result, tmp_path)
        assert isinstance(result, OutputManifest)
        assert result.stdout_file == tmp_path / "stdout.txt"
        assert result.stderr_file == tmp_path / "stderr.txt"
        assert result.stdout_file.read_text(encoding="utf-8") == "output text"
        assert result.stderr_file.read_text(encoding="utf-8") == "error text"
        assert result.returncode == 0
        assert result.timed_out is False

    def test_failed_execution(self, tmp_path):
        docker_result = DockerRunResult(
            copy_in=[],
            commands=[Result(stdout="", stderr="command not found", returncode=127, timed_out=False)],
            copy_out=[],
        )
        result = pack_docker(docker_result, tmp_path)
        assert result.returncode == 127
        assert result.timed_out is False

    def test_timed_out(self, tmp_path):
        docker_result = DockerRunResult(
            copy_in=[],
            commands=[Result(stdout="", stderr="timeout", returncode=-1, timed_out=True)],
            copy_out=[],
        )
        result = pack_docker(docker_result, tmp_path)
        assert result.timed_out is True

    def test_error_result(self, tmp_path):
        docker_result = DockerRunResult(
            copy_in=[],
            commands=[],
            copy_out=[],
            error="container creation failed",
        )
        result = pack_docker(docker_result, tmp_path)
        assert result.returncode == -1
        assert result.stderr_file.read_text(encoding="utf-8") == "container creation failed"

    def test_empty_commands(self, tmp_path):
        docker_result = DockerRunResult(
            copy_in=[],
            commands=[],
            copy_out=[],
        )
        result = pack_docker(docker_result, tmp_path)
        assert result.returncode == 0
        assert result.timed_out is False


class TestDispatchRegistries:
    """Test EXTRACT_DISPATCH and PACK_DISPATCH registries."""

    def test_extract_dispatch_has_chat_llm(self):
        from otter.backend.chat_llm import ChatLLMBackend
        assert EXTRACT_DISPATCH[ChatLLMBackend] is extract_for_chat_llm

    def test_extract_dispatch_has_docker(self):
        from otter.backend.docker import DockerBackend
        assert EXTRACT_DISPATCH[DockerBackend] is extract_for_docker

    def test_pack_dispatch_has_chat_llm(self):
        from otter.backend.chat_llm import ChatLLMBackend
        assert PACK_DISPATCH[ChatLLMBackend] is pack_chat_llm

    def test_pack_dispatch_has_docker(self):
        from otter.backend.docker import DockerBackend
        assert PACK_DISPATCH[DockerBackend] is pack_docker


class TestRoleSubclasses:
    """Test ProposerRole, ExecutorRole, EvaluatorRole field access."""

    def _make_episode_with_turn(self, tmp_path):
        turn_dir = tmp_path / "turn_1"
        turn_dir.mkdir()
        prop_in = turn_dir / "prop_input"
        prop_out = turn_dir / "prop_output"
        exec_in = turn_dir / "exec_input"
        exec_out = turn_dir / "exec_output"
        eval_in = turn_dir / "eval_input"
        eval_out = turn_dir / "eval_output"
        for d in [prop_in, prop_out, exec_in, exec_out, eval_in, eval_out]:
            d.mkdir()

        turn = Turn(
            turn_dir=turn_dir,
            prop_input_path=prop_in,
            prop_output_path=prop_out,
            exec_input_path=exec_in,
            exec_output_path=exec_out,
            eval_input_path=eval_in,
            eval_output_path=eval_out,
            prop_input_manifest=InputManifest(prompt_file=Path("/tmp/p.txt")),
            exec_input_manifest=InputManifest(prompt_file=Path("/tmp/e.txt")),
            eval_input_manifest=InputManifest(image_tag="test:v1"),
        )
        ep = Episode(task_id="t", sample_id=0, turns=[turn], base_dir=tmp_path)
        return ep

    def test_proposer_role_reads_prop_fields(self, tmp_path, mocker):
        ep = self._make_episode_with_turn(tmp_path)
        backend = mocker.MagicMock()
        backend.__class__ = type("FakeBackend", (), {})
        # We can't instantiate ProposerRole with a fake backend due to dispatch lookup,
        # so test the abstract methods directly
        role = ProposerRole.__new__(ProposerRole)
        manifest = role._get_input_manifest(ep)
        assert manifest is ep.turns[-1].prop_input_manifest
        assert role._get_output_dir(ep) == ep.turns[-1].prop_output_path

    def test_executor_role_reads_exec_fields(self, tmp_path, mocker):
        ep = self._make_episode_with_turn(tmp_path)
        role = ExecutorRole.__new__(ExecutorRole)
        manifest = role._get_input_manifest(ep)
        assert manifest is ep.turns[-1].exec_input_manifest
        assert role._get_output_dir(ep) == ep.turns[-1].exec_output_path

    def test_evaluator_role_reads_eval_fields(self, tmp_path, mocker):
        ep = self._make_episode_with_turn(tmp_path)
        role = EvaluatorRole.__new__(EvaluatorRole)
        manifest = role._get_input_manifest(ep)
        assert manifest is ep.turns[-1].eval_input_manifest
        assert role._get_output_dir(ep) == ep.turns[-1].eval_output_path

    def test_set_output_manifest_proposer(self, tmp_path):
        ep = self._make_episode_with_turn(tmp_path)
        role = ProposerRole.__new__(ProposerRole)
        out = OutputManifest(exec_output_file=Path("/tmp/out.txt"))
        role._set_output_manifest(ep, out)
        assert ep.turns[-1].prop_output_manifest is out

    def test_set_output_manifest_executor(self, tmp_path):
        ep = self._make_episode_with_turn(tmp_path)
        role = ExecutorRole.__new__(ExecutorRole)
        out = OutputManifest(exec_output_file=Path("/tmp/out.txt"))
        role._set_output_manifest(ep, out)
        assert ep.turns[-1].exec_output_manifest is out

    def test_set_output_manifest_evaluator(self, tmp_path):
        ep = self._make_episode_with_turn(tmp_path)
        role = EvaluatorRole.__new__(EvaluatorRole)
        out = OutputManifest(returncode=0)
        role._set_output_manifest(ep, out)
        assert ep.turns[-1].eval_output_manifest is out


class TestBaseRoleRun:
    """Test BaseRole.run template method end-to-end with mock backend."""

    async def test_executor_role_run_with_chat_llm(self, tmp_path, mocker):
        """Full run: extract → backend.run → pack → save → set manifest."""
        import otter.role as role_mod

        # Setup prompt file
        exec_in = tmp_path / "exec_input"
        exec_out = tmp_path / "exec_output"
        exec_in.mkdir()
        exec_out.mkdir()

        prompt_file = exec_in / "prompt.txt"
        prompt_file.write_text("Solve this", encoding="utf-8")

        turn = Turn(
            turn_dir=tmp_path,
            exec_input_path=exec_in,
            exec_output_path=exec_out,
            exec_input_manifest=InputManifest(prompt_file=prompt_file),
        )
        ep = Episode(task_id="t", sample_id=0, turns=[turn], base_dir=tmp_path)

        # Build role manually, bypassing __init__ dispatch
        response_file = exec_out / "response.txt"
        response_file.write_text("def solution(): return 42", encoding="utf-8")

        mock_backend = mocker.AsyncMock()
        mock_backend.run.return_value = ChatLLMRunResult(
            products=[response_file],
            retries=[Result(stdout="def solution(): return 42", stderr="", returncode=0, timed_out=False)],
        )

        role = ExecutorRole.__new__(ExecutorRole)
        role.backend = mock_backend
        role._extract = role_mod.extract_for_chat_llm
        role._pack = role_mod.pack_chat_llm

        await role.run(ep)

        # Verify backend was called
        mock_backend.run.assert_called_once()
        call_kwargs = mock_backend.run.call_args[1]
        assert "messages" in call_kwargs
        assert call_kwargs["messages"][-1]["content"] == "Solve this"
        assert call_kwargs["output_file"] == exec_out / "response.txt"

        # Verify output manifest was saved and set
        assert ep.turns[-1].exec_output_manifest is not None
        assert (exec_out / "manifest.json").exists()
        assert response_file.exists()
        assert response_file.read_text(encoding="utf-8") == "def solution(): return 42"
