from datasets import load_dataset

from code_eval.config.setting import settings, ROOT_DIR


class HumanEvalDataset:
    dataset = load_dataset(
        settings.dataset.dataset_name,
        cache_dir=str(settings.dataset.cache_dir)
    )

    def explore(self):
        print(self.dataset)
        print(self.dataset["test"][0])


h = HumanEvalDataset()
h.explore()