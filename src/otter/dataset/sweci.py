import csv
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from otter.dataset.base import BaseDataset
from otter.dataset.utils import read_csv, download_hf_file, download_hf_folder
from otter.config.setting import get_settings
from otter.episode import Episode, InputManifest
from otter.backend.docker import DockerBackend
from otter.logger import get_logger



def download_sweci(splitting: str, save_root_dir: Path) -> None:
    
    # Validate splitting
    hf_repo_id = "skylenage/SWE-CI"
    api = HfApi()
    files = api.list_repo_tree(
        repo_id=hf_repo_id,
        path_in_repo="metadata",
        repo_type="dataset",
        recursive=True,
        token=None
    )
    all_split = [
        Path(f.path).stem for f in files
        if f.path.endswith('.csv')
    ]
    if splitting not in all_split:
        raise ValueError(f"Expected splitting in {all_split}, but got {splitting}")
    
    # Download metadata
    disable_progress_bars()
    metadata_path = download_hf_file(
        repo_id=hf_repo_id,
        remote_file_path=f"metadata/{splitting}.csv",
        local_root_dir=save_root_dir,
        hf_token=None
    )
    metadata = read_csv(metadata_path)
    task_ids = [task['task_id'] for task in metadata]
    
    # Download tasks
    total = len(task_ids)
    for idx, task_id in enumerate(task_ids):
        print(f"({idx+1}/{total}) Preparing {task_id}...", end="    ", flush=True)
        download_hf_folder(
            repo_id=hf_repo_id,
            remote_folder_path=f"data/{task_id}",
            local_root_dir=save_root_dir,
            hf_token=None
        )
        print("Done.", flush=True)


def initialize_sweci():
    pass


class SWECIDataset(BaseDataset):

    async def setup(self) -> None:
        settings = get_settings()
        logger = get_logger()
        download_sweci("default", "./.temp")
        print("hello")


    @property
    def task_ids(self) -> list[str]:
        pass

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        pass

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        pass

    async def _judge(self, episode: Episode) -> None:
        pass
    

