from datasets import load_dataset

from otter.datasets.base import BaseDataset
from otter.config.setting import settings, ROOT_DIR


class HumanEvalDataset(BaseDataset):

    def load(self):
        dataset = load_dataset(
            settings.dataset.dataset_name,
            cache_dir=str(settings.dataset.cache_dir)
        )



if __name__ == "__main__":
    h = HumanEvalDataset()