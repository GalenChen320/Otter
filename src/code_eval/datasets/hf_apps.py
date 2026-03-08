from datasets import load_dataset

from code_eval.config.setting import settings, ROOT_DIR


class APPSDataset:


    def load(self):
        cfg = settings.dataset.apps
        dataset = load_dataset("codeparrot/apps")


    def explore(self):
        print(self.dataset)
        print(self.dataset["test"][0])



h = APPSDataset()
h.load()
h.explore()