from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings


class APPSDataset(BaseDataset):

    async def setup(self) -> None:
        dataset = load_dataset("codeparrot/apps")

