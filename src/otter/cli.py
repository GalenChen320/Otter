import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer

from otter.config.setting import ROOT_DIR

app = typer.Typer(help="Otter - LLM 代码评测框架")

EXPERIMENTS_DIR = ROOT_DIR / "experiments"


@app.command()
def run(
    env: Annotated[Path, typer.Option(help="环境变量文件路径")] = Path(".env"),
):
    """运行评测"""
    if not env.exists():
        typer.echo(f"Error: env file not found: {env}", err=True)
        raise typer.Exit(1)

    from otter.config.setting import set_env_file, init_settings
    from otter.logger import init_logger
    from otter.pipeline import main

    set_env_file(str(env))
    init_settings()
    init_logger()
    asyncio.run(main())


def _resolve_experiment_dir(experiment_id: str | None) -> Path:
    """解析实验目录：传了 --exp 就校验，没传就自动检测。"""
    if experiment_id is not None:
        exp_dir = EXPERIMENTS_DIR / experiment_id
        if not exp_dir.is_dir():
            typer.echo(f"Error: experiment not found: {exp_dir}", err=True)
            raise typer.Exit(1)
        return exp_dir

    if not EXPERIMENTS_DIR.is_dir():
        typer.echo(f"Error: experiments directory not found: {EXPERIMENTS_DIR}", err=True)
        raise typer.Exit(1)

    subdirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        typer.echo("Error: no experiments found", err=True)
        raise typer.Exit(1)
    if len(subdirs) == 1:
        return subdirs[0]

    names = ", ".join(sorted(d.name for d in subdirs))
    typer.echo(f"Error: multiple experiments found, specify one with --exp: {names}", err=True)
    raise typer.Exit(1)


@app.command()
def summary(
    experiment_id: Annotated[Optional[str], typer.Option("--exp", help="实验 ID")] = None,
):
    """查看评测结果摘要"""
    from otter.summary import summarize, show_summary

    exp_dir = _resolve_experiment_dir(experiment_id)
    result = summarize(exp_dir)
    show_summary(result)


@app.command()
def version():
    """查看版本"""
    typer.echo("Otter v0.1.0")


if __name__ == "__main__":
    app()
