import re
import csv
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from jinja2 import Environment, FileSystemLoader, StrictUndefined


from otter.episode import Episode


def build_messages(episode: Episode, current_prompt: str) -> list[dict]:
    """构建完整的多轮 messages 列表（用于 ChatLLM 场景）。

    遍历历史轮次的 exec_input/output，追加当前轮的 prompt。
    """
    messages = []
    for turn in episode.turns[:-1]:
        if turn.exec_input_manifest and turn.exec_input_manifest.msg_file:
            hist_messages = json.loads(
                turn.exec_input_manifest.msg_file.read_text(encoding="utf-8")
            )
            last_user_msg = hist_messages[-1]
            messages.append(last_user_msg)
        if (turn.exec_output_manifest
                and turn.exec_output_manifest.products
                and turn.exec_output_manifest.products[0]):
            response = turn.exec_output_manifest.products[0].read_text(encoding="utf-8")
            messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": current_prompt})
    return messages


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


def unzip(
        zip_file: Path, 
        output_dir: Path,
        ) -> None:
    if not zip_file.is_file():
        raise FileNotFoundError(f"File not found: {str(zip_file)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(output_dir)


def checkout(
        repo_dir: Path, 
        commit_sha: str,
        ) -> None:
    repo_dir = repo_dir.resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {str(repo_dir)}")
    subprocess.run([
        "git", "-C", str(repo_dir), "checkout", "--force", "--detach", commit_sha
        ], check=True, capture_output=True, text=True)


def remove_pattern_files(
        target_dir: Path, 
        patterns: list[str], 
        *,
        recursive: bool = False
        ) -> None:
    target_dir = target_dir.resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {str(target_dir)}")
    for pattern in patterns:
        matches = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
        for item in sorted(matches, key=lambda x: len(x.parts), reverse=True):
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def load_prompt(
        template_path: str | Path,
        template_args: dict | None
        ) -> str:
    template_path = Path(template_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"File not found: {template_path}")
    # autoescape=False is intentional: templates are used as LLM prompts (not HTML),
    # so escaping < > & would corrupt the prompt content.
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,  # nosec B701
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    prompt = template.render(**(template_args or {}))
    return prompt


__all__ = [
    "build_messages",
    "extract_code",
    "read_csv",
    "download_hf_file",
    "download_hf_folder",
    "unzip",
    "checkout",
    "remove_pattern_files",
    "load_prompt",
]