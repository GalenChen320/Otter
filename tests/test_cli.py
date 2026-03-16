"""Tests for otter.cli module."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from otter.cli import app, EXPERIMENTS_DIR, _resolve_experiment_dir

runner = CliRunner()


class TestApp:
    """Typer 应用基本结构验证。"""

    def test_app_is_typer_instance(self):
        assert isinstance(app, typer.Typer)

    def test_app_has_run_command(self):
        """app 应注册了 run 命令。"""
        command_names = [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]
        assert "run" in command_names

    def test_app_has_summary_command(self):
        """app 应注册了 summary 命令。"""
        command_names = [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]
        assert "summary" in command_names

    def test_app_has_version_command(self):
        """app 应注册了 version 命令。"""
        command_names = [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]
        assert "version" in command_names

    def test_experiments_dir_is_under_root(self):
        """EXPERIMENTS_DIR 应该是 ROOT_DIR / 'experiments'。"""
        assert EXPERIMENTS_DIR.name == "experiments"
        assert EXPERIMENTS_DIR.is_absolute()

    def test_help_exits_zero(self):
        """--help 应正常退出。"""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0


class TestRunCommand:
    """run 命令测试。"""

    def test_run_env_not_found(self, tmp_path):
        """指定不存在的 env 文件应报错退出。"""
        result = runner.invoke(app, ["run", "--env", str(tmp_path / "nonexistent.env")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_run_env_not_found_default(self):
        """默认 .env 不存在时应报错退出（CWD 已被 conftest 切到 tmp_path）。"""
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_run_calls_pipeline(self, tmp_path, mocker):
        """env 文件存在时，应依次调用 init_settings、init_logger、asyncio.run(main())。"""
        env_file = tmp_path / "test.env"
        env_file.touch()

        mock_init_settings = mocker.patch("otter.config.setting.init_settings")
        mock_init_logger = mocker.patch("otter.logger.init_logger")
        mock_main = mocker.patch("otter.pipeline.main")
        mock_asyncio_run = mocker.patch("otter.cli.asyncio.run")

        result = runner.invoke(app, ["run", "--env", str(env_file)])
        assert result.exit_code == 0
        mock_init_settings.assert_called_once_with(str(env_file))
        mock_init_logger.assert_called_once()
        mock_asyncio_run.assert_called_once()

    def test_run_only_accepts_env_option(self):
        """run 命令应只有 --env 一个选项，不应有其他参数。"""
        import re
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--env" in result.output
        # 提取所有 --xxx 选项名（排除 --help）
        options = re.findall(r"--(\S+)", result.output)
        custom_options = [o for o in options if o != "help"]
        assert custom_options == ["env"]


class TestResolveExperimentDir:
    """_resolve_experiment_dir 函数测试。"""

    def test_explicit_id_valid(self, tmp_path, monkeypatch):
        """指定存在的 experiment_id 应返回对应目录。"""
        exp_dir = tmp_path / "experiments" / "exp_001"
        exp_dir.mkdir(parents=True)
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", tmp_path / "experiments")

        result = _resolve_experiment_dir("exp_001")
        assert result == exp_dir

    def test_explicit_id_not_found(self, tmp_path, monkeypatch):
        """指定不存在的 experiment_id 应 raise typer.Exit(1)。"""
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", exp_dir)

        with pytest.raises(typer.Exit) as exc_info:
            _resolve_experiment_dir("nonexistent")
        assert exc_info.value.exit_code == 1

    def test_no_id_experiments_dir_missing(self, tmp_path, monkeypatch):
        """未指定 id 且 experiments 目录不存在应报错退出。"""
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", tmp_path / "no_such_dir")

        with pytest.raises(typer.Exit) as exc_info:
            _resolve_experiment_dir(None)
        assert exc_info.value.exit_code == 1

    def test_no_id_no_experiments(self, tmp_path, monkeypatch):
        """未指定 id 且 experiments 目录为空应报错退出。"""
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", exp_dir)

        with pytest.raises(typer.Exit) as exc_info:
            _resolve_experiment_dir(None)
        assert exc_info.value.exit_code == 1

    def test_no_id_single_experiment(self, tmp_path, monkeypatch):
        """未指定 id 且只有一个实验目录时应自动选中。"""
        exp_dir = tmp_path / "experiments"
        only_exp = exp_dir / "the_only_one"
        only_exp.mkdir(parents=True)
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", exp_dir)

        result = _resolve_experiment_dir(None)
        assert result == only_exp

    def test_no_id_multiple_experiments(self, tmp_path, monkeypatch):
        """未指定 id 且有多个实验目录时应报错退出。"""
        exp_dir = tmp_path / "experiments"
        (exp_dir / "exp_a").mkdir(parents=True)
        (exp_dir / "exp_b").mkdir(parents=True)
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", exp_dir)

        with pytest.raises(typer.Exit) as exc_info:
            _resolve_experiment_dir(None)
        assert exc_info.value.exit_code == 1

    def test_no_id_ignores_files(self, tmp_path, monkeypatch):
        """自动检测时应只考虑子目录，忽略文件。"""
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        (exp_dir / "readme.txt").touch()  # 文件，不是目录
        only_exp = exp_dir / "real_exp"
        only_exp.mkdir()
        monkeypatch.setattr("otter.cli.EXPERIMENTS_DIR", exp_dir)

        result = _resolve_experiment_dir(None)
        assert result == only_exp


class TestSummaryCommand:
    """summary 命令测试。"""

    def test_summary_help_exits_zero(self):
        """summary --help 应正常退出。"""
        result = runner.invoke(app, ["summary", "--help"])
        assert result.exit_code == 0
        assert "--exp" in result.output

    def test_summary_calls_summarize_and_show(self, tmp_path, mocker):
        """summary 命令应调用 _resolve_experiment_dir、summarize、show_summary。"""
        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        mocker.patch("otter.cli.EXPERIMENTS_DIR", tmp_path / "experiments")

        mock_summarize = mocker.patch("otter.summary.summarize", return_value="fake_result")
        mock_show = mocker.patch("otter.summary.show_summary")

        result = runner.invoke(app, ["summary", "--exp", "test_exp"])
        assert result.exit_code == 0
        mock_summarize.assert_called_once_with(exp_dir)
        mock_show.assert_called_once_with("fake_result")

    def test_summary_nonexistent_exp(self, tmp_path, mocker):
        """指定不存在的实验 ID 应报错退出。"""
        mocker.patch("otter.cli.EXPERIMENTS_DIR", tmp_path / "experiments")
        (tmp_path / "experiments").mkdir()

        result = runner.invoke(app, ["summary", "--exp", "ghost"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_summary_auto_detect_single(self, tmp_path, mocker):
        """不指定 --exp 且只有一个实验时应自动选中。"""
        exp_dir = tmp_path / "experiments" / "only_one"
        exp_dir.mkdir(parents=True)
        mocker.patch("otter.cli.EXPERIMENTS_DIR", tmp_path / "experiments")

        mock_summarize = mocker.patch("otter.summary.summarize", return_value="result")
        mock_show = mocker.patch("otter.summary.show_summary")

        result = runner.invoke(app, ["summary"])
        assert result.exit_code == 0
        mock_summarize.assert_called_once_with(exp_dir)
        mock_show.assert_called_once_with("result")

    def test_summary_auto_detect_multiple_fails(self, tmp_path, mocker):
        """不指定 --exp 且有多个实验时应报错退出。"""
        exps = tmp_path / "experiments"
        (exps / "a").mkdir(parents=True)
        (exps / "b").mkdir(parents=True)
        mocker.patch("otter.cli.EXPERIMENTS_DIR", exps)

        result = runner.invoke(app, ["summary"])
        assert result.exit_code == 1


class TestVersionCommand:
    """version 命令测试。"""

    def test_version_exits_zero(self):
        """version 命令应正常退出。"""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_output_format(self):
        """输出应包含 'Otter v' 前缀和版本号。"""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        output = result.output.strip()
        assert output.startswith("Otter v")
        # 版本号部分应非空
        version_str = output.removeprefix("Otter v")
        assert len(version_str) > 0

    def test_version_matches_package_metadata(self):
        """输出的版本号应与 importlib.metadata 一致。"""
        from importlib.metadata import version as get_version
        expected = get_version("otter")
        result = runner.invoke(app, ["version"])
        assert f"Otter v{expected}" in result.output

    def test_version_help(self):
        """version --help 应正常退出。"""
        result = runner.invoke(app, ["version", "--help"])
        assert result.exit_code == 0
