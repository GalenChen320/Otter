import re
import csv
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


def extract_code(response: str) -> str:
    """从 LLM response 中提取 Python 代码块。

    优先匹配 ```python ... ``` 包裹的代码块，
    如果没有匹配到则返回原始文本（去除首尾空白）。
    """
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def read_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def download_hf_file(
        repo_id: str,
        remote_file_path: str,
        local_root_dir: Path,
        hf_token: str | None = None,
        repo_type: str = "dataset"
        ) -> Path:
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=remote_file_path,
        repo_type=repo_type,
        local_dir=local_root_dir,
        token=hf_token
    ))


def download_hf_folder(
        repo_id: str,
        remote_folder_path: str,
        local_root_dir: Path,
        hf_token: str | None = None,
        repo_type: str = "dataset"
        ) -> Path:
    return Path(snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=[f"{remote_folder_path}/**"],
        local_dir=local_root_dir,
        token=hf_token
    ))


__all__ = [
    "extract_code",
    "read_csv",
    "download_hf_file",
    "download_hf_folder",
]