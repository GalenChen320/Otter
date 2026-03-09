import asyncio
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Otter - LLM 代码评测框架")


@app.command()
def run(
    env: Annotated[Path, typer.Option(help="环境变量文件路径")] = Path(".env"),
):
    """运行评测"""
    from otter.config.setting import set_env_file, init_settings
    from otter.logger import init_logger
    from otter.pipeline import main

    set_env_file(str(env))
    init_settings()
    init_logger()
    asyncio.run(main())
