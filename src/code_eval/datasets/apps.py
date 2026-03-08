from datasets import load_dataset

from code_eval.datasets.base import BaseDataset
from code_eval.config.setting import settings, ROOT_DIR


class APPSDataset(BaseDataset):


    def load(self):
        cfg = settings.dataset.apps
        dataset = load_dataset("codeparrot/apps")



if __name__ == "__main__":
    apps = APPSDataset()
    apps.load()