from datasets import load_dataset

from code_eval.datasets.base import BaseDataset
from code_eval.config.setting import settings, ROOT_DIR


class APPSDataset(BaseDataset):


    def load(self):
        cfg = settings.dataset.apps
        dataset = load_dataset("codeparrot/apps")


    def explore(self):
        print(self.dataset)
        print(self.dataset["test"][0])


if __name__ == "__main__":
    apps = APPSDataset()
    apps.load()
    apps.explore()