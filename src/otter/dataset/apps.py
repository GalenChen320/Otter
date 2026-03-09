from datasets import load_dataset

from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings, ROOT_DIR


class APPSDataset(BaseDataset):

    def load(self):
        dataset = load_dataset("codeparrot/apps")



if __name__ == "__main__":
    apps = APPSDataset()
    apps.load()