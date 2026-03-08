from datasets import load_dataset

from code_eval.datasets.base import BaseDataset
from code_eval.config.setting import settings, ROOT_DIR


class HumanEvalDataset(BaseDataset):

    def load(self):
        dataset = load_dataset(
            settings.dataset.dataset_name,
            cache_dir=str(settings.dataset.cache_dir)
        )



if __name__ == "__main__":
    h = HumanEvalDataset()