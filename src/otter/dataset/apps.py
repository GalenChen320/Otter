from pathlib import Path

from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings


class APPSDataset(BaseDataset):

    async def setup(self, output_dir: Path) -> None:
        await super().setup(output_dir)
        dataset = load_dataset("codeparrot/apps")


if __name__ == "__main__":
    apps = APPSDataset()