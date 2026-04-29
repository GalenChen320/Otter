"""Tests for otter.episode module."""

import json
from pathlib import Path


from otter.episode import (
    _is_path_field,
    InputManifest,
    OutputManifest,
    Turn,
    Episode,
    META_FILENAME,
)


class TestIsPathField:
    """Test _is_path_field helper function."""

    def test_plain_path(self):
        assert _is_path_field(Path) is True

    def test_path_or_none(self):
        assert _is_path_field(Path | None) is True

    def test_plain_str(self):
        assert _is_path_field(str) is False

    def test_str_or_none(self):
        assert _is_path_field(str | None) is False

    def test_int(self):
        assert _is_path_field(int) is False

    def test_list_str(self):
        assert _is_path_field(list[str]) is False


class TestInputManifest:
    """Test InputManifest serialization and deserialization."""

    def test_to_dict_all_none(self):
        m = InputManifest()
        d = m.to_dict()
        assert d == {
            "prompt_file": None,
            "image_tag": None,
            "script_file": None,
            "commands": None,
            "timeout": None,
        }

    def test_to_dict_with_path_fields(self):
        m = InputManifest(
            prompt_file=Path("/tmp/prompt.txt"),
            image_tag="test:latest",
            script_file=Path("/tmp/script.py"),
            commands=["python /tmp/script.py"],
            timeout=30,
        )
        d = m.to_dict()
        assert d["prompt_file"] == "/tmp/prompt.txt"
        assert d["script_file"] == "/tmp/script.py"
        assert d["image_tag"] == "test:latest"
        assert d["commands"] == ["python /tmp/script.py"]
        assert d["timeout"] == 30

    def test_from_dict_restores_paths(self):
        data = {
            "prompt_file": "/tmp/prompt.txt",
            "image_tag": "test:latest",
            "script_file": "/tmp/script.py",
            "commands": ["echo hello"],
            "timeout": 10,
        }
        m = InputManifest.from_dict(data)
        assert isinstance(m.prompt_file, Path)
        assert m.prompt_file == Path("/tmp/prompt.txt")
        assert isinstance(m.script_file, Path)
        assert m.image_tag == "test:latest"
        assert m.commands == ["echo hello"]
        assert m.timeout == 10

    def test_from_dict_with_none_values(self):
        data = {"prompt_file": None, "image_tag": None}
        m = InputManifest.from_dict(data)
        assert m.prompt_file is None
        assert m.image_tag is None

    def test_from_dict_ignores_unknown_keys(self):
        data = {"prompt_file": "/tmp/p.txt", "unknown_field": "value"}
        m = InputManifest.from_dict(data)
        assert m.prompt_file == Path("/tmp/p.txt")
        assert not hasattr(m, "unknown_field")

    def test_to_dict_from_dict_roundtrip(self):
        original = InputManifest(
            prompt_file=Path("/a/b.txt"),
            image_tag="img:v1",
            commands=["cmd1", "cmd2"],
            timeout=60,
        )
        restored = InputManifest.from_dict(original.to_dict())
        assert restored.prompt_file == original.prompt_file
        assert restored.image_tag == original.image_tag
        assert restored.commands == original.commands
        assert restored.timeout == original.timeout

    def test_save_writes_json(self, tmp_path):
        m = InputManifest(prompt_file=Path("/tmp/p.txt"), image_tag="test:v1")
        m.save(tmp_path)
        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.exists()
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert data["prompt_file"] == "/tmp/p.txt"
        assert data["image_tag"] == "test:v1"


class TestOutputManifest:
    """Test OutputManifest serialization and deserialization."""

    def test_to_dict_all_none(self):
        m = OutputManifest()
        d = m.to_dict()
        assert d == {
            "exec_output_file": None,
            "stdout_file": None,
            "stderr_file": None,
            "returncode": None,
            "timed_out": None,
        }

    def test_to_dict_with_values(self):
        m = OutputManifest(
            exec_output_file=Path("/tmp/out.txt"),
            stdout_file=Path("/tmp/stdout.txt"),
            stderr_file=Path("/tmp/stderr.txt"),
            returncode=0,
            timed_out=False,
        )
        d = m.to_dict()
        assert d["exec_output_file"] == "/tmp/out.txt"
        assert d["stdout_file"] == "/tmp/stdout.txt"
        assert d["returncode"] == 0
        assert d["timed_out"] is False

    def test_from_dict_restores_paths(self):
        data = {
            "exec_output_file": "/tmp/out.txt",
            "stdout_file": "/tmp/stdout.txt",
            "stderr_file": "/tmp/stderr.txt",
            "returncode": 1,
            "timed_out": True,
        }
        m = OutputManifest.from_dict(data)
        assert isinstance(m.exec_output_file, Path)
        assert isinstance(m.stdout_file, Path)
        assert m.returncode == 1
        assert m.timed_out is True

    def test_roundtrip(self):
        original = OutputManifest(
            stdout_file=Path("/a/stdout.txt"),
            stderr_file=Path("/a/stderr.txt"),
            returncode=0,
            timed_out=False,
        )
        restored = OutputManifest.from_dict(original.to_dict())
        assert restored.stdout_file == original.stdout_file
        assert restored.stderr_file == original.stderr_file
        assert restored.returncode == original.returncode
        assert restored.timed_out == original.timed_out


class TestTurn:
    """Test Turn directory setup and meta saving."""

    def test_save_meta_writes_json(self, tmp_path):
        turn_dir = tmp_path / "turn_1"
        turn_dir.mkdir()
        turn = Turn(turn_dir=turn_dir, passed=True)
        turn.save_meta()
        meta_path = turn_dir / META_FILENAME
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["passed"] is True

    def test_save_meta_with_none_passed(self, tmp_path):
        turn_dir = tmp_path / "turn_1"
        turn_dir.mkdir()
        turn = Turn(turn_dir=turn_dir, passed=None)
        turn.save_meta()
        data = json.loads((turn_dir / META_FILENAME).read_text(encoding="utf-8"))
        assert data["passed"] is None

    def test_save_meta_with_false_passed(self, tmp_path):
        turn_dir = tmp_path / "turn_1"
        turn_dir.mkdir()
        turn = Turn(turn_dir=turn_dir, passed=False)
        turn.save_meta()
        data = json.loads((turn_dir / META_FILENAME).read_text(encoding="utf-8"))
        assert data["passed"] is False

    def test_setup_dirs_with_executor_only(self, tmp_path, mocker):
        """When only executor is configured, only exec dirs should be created."""
        mock_settings = mocker.MagicMock()
        mock_settings.proposer = None
        mock_settings.executor = mocker.MagicMock()  # not None
        mock_settings.evaluator = None
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        turn_dir = tmp_path / "turn_1"
        turn = Turn(turn_dir=turn_dir)
        turn.setup_dirs()

        assert turn_dir.exists()
        assert turn.exec_input_path == turn_dir / "exec_input"
        assert turn.exec_output_path == turn_dir / "exec_output"
        assert (turn_dir / "exec_input").is_dir()
        assert (turn_dir / "exec_output").is_dir()
        # proposer and evaluator dirs should not exist
        assert turn.prop_input_path is None
        assert turn.eval_input_path is None

    def test_setup_dirs_with_all_components(self, tmp_path, mocker):
        """When all components configured, all dirs should be created."""
        mock_settings = mocker.MagicMock()
        mock_settings.proposer = mocker.MagicMock()
        mock_settings.executor = mocker.MagicMock()
        mock_settings.evaluator = mocker.MagicMock()
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        turn_dir = tmp_path / "turn_1"
        turn = Turn(turn_dir=turn_dir)
        turn.setup_dirs()

        assert (turn_dir / "prop_input").is_dir()
        assert (turn_dir / "prop_output").is_dir()
        assert (turn_dir / "exec_input").is_dir()
        assert (turn_dir / "exec_output").is_dir()
        assert (turn_dir / "eval_input").is_dir()
        assert (turn_dir / "eval_output").is_dir()
        # Verify path fields are set correctly
        assert turn.prop_input_path == turn_dir / "prop_input"
        assert turn.prop_output_path == turn_dir / "prop_output"
        assert turn.exec_input_path == turn_dir / "exec_input"
        assert turn.exec_output_path == turn_dir / "exec_output"
        assert turn.eval_input_path == turn_dir / "eval_input"
        assert turn.eval_output_path == turn_dir / "eval_output"

    def test_setup_dirs_with_no_components(self, tmp_path, mocker):
        """When no components configured, only turn_dir should be created."""
        mock_settings = mocker.MagicMock()
        mock_settings.proposer = None
        mock_settings.executor = None
        mock_settings.evaluator = None
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        turn_dir = tmp_path / "turn_1"
        turn = Turn(turn_dir=turn_dir)
        turn.setup_dirs()

        assert turn_dir.exists()
        assert turn.prop_input_path is None
        assert turn.exec_input_path is None
        assert turn.eval_input_path is None


class TestEpisode:
    """Test Episode properties and methods."""

    def test_make_eid(self):
        assert Episode.make_eid("task_1", 0) == "task_1#0"
        assert Episode.make_eid("HumanEval_0", 2) == "HumanEval_0#2"

    def test_eid_property(self):
        ep = Episode(task_id="task_1", sample_id=3)
        assert ep.eid == "task_1#3"

    def test_resolved_true(self):
        turn = Turn(turn_dir=Path("/fake"), passed=True)
        ep = Episode(task_id="t", sample_id=0, turns=[turn])
        assert ep.resolved is True

    def test_resolved_true_after_earlier_failure(self):
        t1 = Turn(turn_dir=Path("/fake1"), passed=False)
        t2 = Turn(turn_dir=Path("/fake2"), passed=True)
        ep = Episode(task_id="t", sample_id=0, turns=[t1, t2])
        assert ep.resolved is True

    def test_resolved_false_when_last_turn_failed(self):
        turn = Turn(turn_dir=Path("/fake"), passed=False)
        ep = Episode(task_id="t", sample_id=0, turns=[turn])
        assert ep.resolved is False

    def test_resolved_false_when_no_turns(self):
        ep = Episode(task_id="t", sample_id=0)
        assert ep.resolved is False

    def test_resolved_false_when_last_turn_none(self):
        turn = Turn(turn_dir=Path("/fake"), passed=None)
        ep = Episode(task_id="t", sample_id=0, turns=[turn])
        assert ep.resolved is False

    def test_total_turns(self):
        ep = Episode(task_id="t", sample_id=0)
        assert ep.total_turns == 0
        ep.turns.append(Turn(turn_dir=Path("/fake")))
        assert ep.total_turns == 1

    def test_exhausted(self):
        ep = Episode(task_id="t", sample_id=0)
        assert ep.exhausted(1) is False
        ep.turns.append(Turn(turn_dir=Path("/fake")))
        assert ep.exhausted(1) is True
        assert ep.exhausted(2) is False

    def test_exhausted_zero_max_turns(self):
        ep = Episode(task_id="t", sample_id=0)
        assert ep.exhausted(0) is True

    def test_next_turn(self, tmp_path, mocker):
        """next_turn should create turn dir, call setup_dirs, and append."""
        mock_settings = mocker.MagicMock()
        mock_settings.proposer = None
        mock_settings.executor = mocker.MagicMock()
        mock_settings.evaluator = None
        mocker.patch("otter.episode.get_settings", return_value=mock_settings)

        ep = Episode(task_id="t", sample_id=0, base_dir=tmp_path)
        ep.next_turn()
        assert ep.total_turns == 1
        assert ep.turns[0].turn_dir == tmp_path / "turn_1"
        assert (tmp_path / "turn_1").is_dir()

        ep.next_turn()
        assert ep.total_turns == 2
        assert ep.turns[1].turn_dir == tmp_path / "turn_2"


class TestEpisodeSyncAll:
    """Test Episode.sync_all directory scanning and state rebuild."""

    def _make_turn_dir(self, ep_dir, turn_num, passed, with_exec=True, with_eval=True):
        """Helper to create a turn directory with meta.json and optional subdirs."""
        turn_dir = ep_dir / f"turn_{turn_num}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        meta = {"passed": passed}
        (turn_dir / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
        if with_exec:
            (turn_dir / "exec_input").mkdir()
            (turn_dir / "exec_output").mkdir()
        if with_eval:
            (turn_dir / "eval_input").mkdir()
            (turn_dir / "eval_output").mkdir()
        return turn_dir

    def test_empty_directory(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        result = Episode.sync_all(tmp_path)
        assert result == {}

    def test_nonexistent_directory(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        result = Episode.sync_all(tmp_path / "nonexistent")
        assert result == {}

    def test_single_episode_single_turn(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        ep_dir = tmp_path / "task_1#0"
        ep_dir.mkdir()
        self._make_turn_dir(ep_dir, 1, passed=True)

        result = Episode.sync_all(tmp_path)
        assert "task_1#0" in result
        ep = result["task_1#0"]
        assert ep.task_id == "task_1"
        assert ep.sample_id == 0
        assert len(ep.turns) == 1
        assert ep.turns[0].passed is True

    def test_multiple_turns(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        ep_dir = tmp_path / "task_1#0"
        ep_dir.mkdir()
        self._make_turn_dir(ep_dir, 1, passed=False)
        self._make_turn_dir(ep_dir, 2, passed=True)

        result = Episode.sync_all(tmp_path)
        ep = result["task_1#0"]
        assert len(ep.turns) == 2
        assert ep.turns[0].passed is False
        assert ep.turns[1].passed is True

    def test_incomplete_turn_cleaned(self, tmp_path, mocker):
        """Turn without meta.json should be deleted."""
        mocker.patch("otter.logger.get_logger")
        ep_dir = tmp_path / "task_1#0"
        ep_dir.mkdir()
        # Complete turn
        self._make_turn_dir(ep_dir, 1, passed=False)
        # Incomplete turn (no meta.json)
        incomplete = ep_dir / "turn_2"
        incomplete.mkdir()
        (incomplete / "exec_input").mkdir()

        result = Episode.sync_all(tmp_path)
        ep = result["task_1#0"]
        assert len(ep.turns) == 1
        assert not incomplete.exists()  # cleaned up

    def test_skips_non_episode_dirs(self, tmp_path, mocker):
        """Directories without '#' in name should be skipped."""
        mocker.patch("otter.logger.get_logger")
        (tmp_path / "not_an_episode").mkdir()
        (tmp_path / "some_file.txt").write_text("hi")
        ep_dir = tmp_path / "task_1#0"
        ep_dir.mkdir()
        self._make_turn_dir(ep_dir, 1, passed=True)

        result = Episode.sync_all(tmp_path)
        assert len(result) == 1
        assert "task_1#0" in result

    def test_manifests_loaded(self, tmp_path, mocker):
        """Manifests in subdirectories should be loaded."""
        mocker.patch("otter.logger.get_logger")
        ep_dir = tmp_path / "task_1#0"
        ep_dir.mkdir()
        turn_dir = ep_dir / "turn_1"
        turn_dir.mkdir()
        (turn_dir / META_FILENAME).write_text('{"passed": true}')

        exec_input = turn_dir / "exec_input"
        exec_input.mkdir()
        manifest_data = {"prompt_file": "/tmp/prompt.txt"}
        (exec_input / "manifest.json").write_text(json.dumps(manifest_data))

        result = Episode.sync_all(tmp_path)
        ep = result["task_1#0"]
        assert ep.turns[0].exec_input_manifest is not None
        assert ep.turns[0].exec_input_manifest.prompt_file == Path("/tmp/prompt.txt")

    def test_multiple_episodes(self, tmp_path, mocker):
        mocker.patch("otter.logger.get_logger")
        for task_id in ["task_1", "task_2"]:
            for sample in [0, 1]:
                ep_dir = tmp_path / f"{task_id}#{sample}"
                ep_dir.mkdir()
                self._make_turn_dir(ep_dir, 1, passed=(sample == 0))

        result = Episode.sync_all(tmp_path)
        assert len(result) == 4
        assert result["task_1#0"].turns[0].passed is True
        assert result["task_1#1"].turns[0].passed is False
