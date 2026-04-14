import asyncio
from pathlib import Path
from typing import Annotated

import typer

from otter.config.setting import ROOT_DIR

app = typer.Typer(help="Otter - 智能体代码能力评测框架")

EXPERIMENTS_DIR = ROOT_DIR / "experiments"


@app.command()
def run(
    env: Annotated[Path, typer.Option(help="环境变量文件路径")] = Path(".env"),
):
    """运行评测"""
    if not env.exists():
        typer.echo(f"Error: env file not found: {env}", err=True)
        raise typer.Exit(1)

    from otter.config.setting import init_settings
    from otter.logger import init_logger
    from otter.pipeline import main

    init_settings(str(env))
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
    env: Annotated[Path, typer.Option(help="环境变量文件路径")] = Path(".env"),
):
    """运行评测"""
    if not env.exists():
        typer.echo(f"Error: env file not found: {env}", err=True)
        raise typer.Exit(1)

    from otter.config.setting import init_settings
    from otter.summary import summarize

    init_settings(str(env))
    summarize()


@app.command()
def version():
    """查看版本"""
    from importlib.metadata import version as get_version
    typer.echo(f"Otter v{get_version('otter')}")


if __name__ == "__main__":
    app()
