from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings


class EvalPlusDataset(BaseDataset):

    async def setup(self) -> None:
        settings = get_settings()
        dataset = load_dataset(
            "openai/openai_humaneval",
            cache_dir=str(settings.dataset.cache_dir)
        )


if __name__ == "__main__":
    h = EvalPlusDataset()