import json
import shutil
import asyncio
import hashlib
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars


from otter.logger import get_logger
from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings
from otter.episode import Episode, InputManifest
from otter.backend.docker import DockerBackend
from otter.dataset.utils import (
    read_csv, download_hf_file, download_hf_folder,
    unzip, checkout, remove_pattern_files
)

def safe_name(node_id: str) -> str:
    safe_name = "".join(c if c.isalnum() or c == '.' else '_' for c in node_id)
    if len(safe_name) > 100:
        hash_suffix = hashlib.sha256(node_id.encode('utf-8')).hexdigest()[:12]
        safe_name = f"{safe_name[:100]}_{hash_suffix}"
    return safe_name


def generate_nonpassed_dir(
        current_report: str | Path,
        target_report: str | Path,
        output_root: str | Path
        ) -> int:

    current = json.loads(Path(current_report).read_text(encoding="utf-8"))
    target = json.loads(Path(target_report).read_text(encoding="utf-8"))
    
    current_passed_ids = set([t['nodeid'] for t in current['tests'] if t['outcome'] == 'passed'])
    target_passed_ids = set([t['nodeid'] for t in target['tests'] if t['outcome'] == 'passed'])
    nonpassed_ids = target_passed_ids - current_passed_ids
    
    current_tests = {t['nodeid']: t for t in current['tests']}
    nonpassed_dir = Path(output_root) / "non-passed"
    shutil.rmtree(nonpassed_dir, ignore_errors=True)
    nonpassed_dir.mkdir(parents=True, exist_ok=True)
    with (nonpassed_dir / "summary.jsonl").open('w', encoding='utf-8') as f:
        for node_id in sorted(nonpassed_ids):
            if node_id in current_tests:
                stage, reason = "", ""
                test_node = current_tests[node_id]
                detail_path = nonpassed_dir / f"{safe_name(node_id)}.json"
                for s in ['setup', 'call', 'teardown']:
                    if s in test_node and 'longrepr' in test_node[s]:
                        reason = "Reason: " + test_node[s]['longrepr'].strip().split('\n')[-1]
                        stage = " at " + s + " stage"
                        break
                f.write(json.dumps({
                    'test': node_id,
                    'description': f"Test {test_node['outcome']}{stage}. {reason}. Detailed traceback is recorded in {str(detail_path)}"
                    }, ensure_ascii=False) + '\n')
                with detail_path.open('w', encoding='utf-8') as d:
                    json.dump(test_node, d, ensure_ascii=False, indent=4)
            else:
                f.write(json.dumps({
                    'test': node_id,
                    'description': f"This test was not properly collected or executed in the current run, so no detailed traceback is available. However, it was recorded as 'passed' in the target report."
                }, ensure_ascii=False) + '\n')
    return len(nonpassed_ids)



def download_sweci() -> None:
    settings = get_settings()
    logger = get_logger()
    splitting = settings.dataset.splitting
    cache_dir = settings.dataset.cache_dir
    logger.info("downloading sweci dataset (splitting=%s)", splitting)

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
        local_root_dir=cache_dir,
        hf_token=None
    )
    metadata = read_csv(metadata_path)
    task_ids = [task['task_id'] for task in metadata]
    
    # Download tasks
    total = len(task_ids)
    logger.info("downloading %d tasks", total)
    for idx, task_id in enumerate(task_ids):
        logger.info("(%d/%d) downloading %s", idx + 1, total, task_id)
        download_hf_folder(
            repo_id=hf_repo_id,
            remote_folder_path=f"data/{task_id}",
            local_root_dir=cache_dir,
            hf_token=None
        )
    logger.info("all tasks downloaded")



async def initialize_sweci():
    settings = get_settings()
    logger = get_logger()
    assert settings.evaluator_type == "docker", "..."
    splitting = settings.dataset.splitting
    cache_dir = Path(settings.dataset.cache_dir)
    metadata_path = cache_dir / "metadata" / f"{splitting}.csv"
    metadata = read_csv(metadata_path)

    logger.info("initializing %d tasks", len(metadata))

    backend = DockerBackend(
        timeout=settings.evaluator.timeout,
        cpus=settings.evaluator.cpus,
        memory=settings.evaluator.memory,
        memory_swap=settings.evaluator.memory_swap,
        memory_reservation=settings.evaluator.memory_reservation,
        network_mode=settings.evaluator.network_mode,
        device_read_bps=settings.evaluator.device_read_bps,
        device_write_bps=settings.evaluator.device_write_bps,
    )
    semaphore = asyncio.Semaphore(settings.evaluator_concurrency)
    loaded_images: set[str] = set()
    total, completed = len(metadata), 0

    async def process_task(task):
        nonlocal completed
        async with semaphore:
            tid = task['task_id']
            task_dir = cache_dir / "processed" / tid
            if (task_dir / ".done").exists():
                completed += 1
                logger.info("(%d/%d) skipping %s (already done)", completed, total, tid)
                return
            if task_dir.exists():
                shutil.rmtree(task_dir)
            logger.info("initializing %s", tid)
            task_dir.mkdir(parents=True, exist_ok=True)
            current_dir = task_dir / "current"
            target_dir = task_dir / "target"
            current_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            data_dir = cache_dir / "data" / tid
            unzip(data_dir / "code.zip", current_dir / "code")
            shutil.copytree(current_dir / "code", target_dir / "code")
            checkout(current_dir / "code", task["current_sha"])
            checkout(target_dir / "code", task["target_sha"])
            remove_pattern_files(current_dir / "code", [".git*", "test", "tests"])
            remove_pattern_files(target_dir / "code", [".git*", "test"])
            shutil.copytree(target_dir / "code" / "tests", current_dir / "code" / "tests")
            image_tag = await DockerBackend.load_image(data_dir / "image.tar.gz")
            loaded_images.add(image_tag)
            container_report = "/tmp/test_report.json"
            current_result = await backend._run(
                image_tag,
                commands=[
                    (
                        f"python -m pytest tests --color=no --tb=short --disable-warnings -rfE --rootdir=/app/code --json-report --json-report-file={container_report}",
                        {"workdir": "/app/code", "environment": {"PYTHONPATH": "src:."}},
                    )
                ],
                copy_in=[(current_dir/"code", "/app")],
                copy_out=[(container_report, current_dir)],
            )
            target_result = await backend._run(
                image_tag,
                commands=[
                    (
                        f"python -m pytest tests --color=no --tb=short --disable-warnings -rfE --rootdir=/app/code --json-report --json-report-file={container_report}",
                        {"workdir": "/app/code", "environment": {"PYTHONPATH": "src:."}},
                    )   
                ],
                copy_in=[(target_dir/"code", "/app")],
                copy_out=[(container_report, target_dir)],
            )
            current_report = current_dir / "test_report.json"
            target_report = target_dir / "test_report.json"
            assert current_report.is_file() and target_report.is_file(), "..."
            generate_nonpassed_dir(
                current_dir / "test_report.json", 
                target_dir / "test_report.json",
                task_dir
            )
            (task_dir / ".done").touch()
            completed += 1
            logger.info("(%d/%d) initialized %s", completed, total, tid)

    await asyncio.gather(*[process_task(task) for task in metadata])

    for tag in loaded_images:
        await DockerBackend.remove_image(tag)
    logger.info("all tasks initialized")
    return [task['task_id'] for task in metadata]





class SWECIDataset(BaseDataset):

    def __init__(self, base_dir: Path) -> None:
        super().__init__(base_dir)
        self.base_image_tags = []
        self.agnet_image_tages = []

    async def setup(self) -> None:
        download_sweci()
        self._taskids = await initialize_sweci()        

    @property
    def task_ids(self) -> list[str]:
        return self._taskids

    async def setup_episode(self, episode: Episode) -> None:
        """Episode 级别初始化，每道题开始前调用。"""
        settings = get_settings()
        logger = get_logger()

        base_image = settings.dataset.cache_dir / "data" / episode.task_id / "image.tar.gz"
        base_image_tag = await DockerBackend.load_image(base_image)
        await DockerBackend.build_image(extra_params={})

        await DockerBackend.build_image(
            exist_ok = True 
        )

    async def teardown_episode(self, episode: Episode) -> None:
        """Episode 级别资源回收，每道题结束后调用。"""
        pass

    def _prepare_prop_input(self, episode: Episode) -> InputManifest:
        pass

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        pass

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        pass

    async def _judge(self, episode: Episode) -> None:
        pass
    

