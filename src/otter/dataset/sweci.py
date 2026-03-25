import json
import shutil
import asyncio
import hashlib
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

from docker_cli.base import BaseAgentDriver
from docker_cli.claude.claude import ClaudeConfig, ClaudeDriver
from docker_cli.codex.codex import CodexConfig, CodexDriver
from docker_cli.opencode.opencode import OpenCodeConfig, OpenCodeDriver
from docker_cli.openhands.openhands import OpenHandsConfig, OpenHandsDriver

from otter.logger import get_logger
from otter.dataset.base import BaseDataset
from otter.config.setting import get_settings
from otter.episode import Episode, InputManifest, OutputManifest, Turn
from otter.backend.docker import DockerBackend
from otter.dataset.utils import (
    read_csv, download_hf_file, download_hf_folder,
    unzip, checkout, remove_pattern_files, load_prompt
)


# agent_name → (DriverClass, ConfigClass)
AGENT_REGISTRY: dict[str, tuple[type[BaseAgentDriver], type]] = {
    "claude":   (ClaudeDriver, ClaudeConfig),
    "codex":    (CodexDriver, CodexConfig),
    "opencode": (OpenCodeDriver, OpenCodeConfig),
    "openhands": (OpenHandsDriver, OpenHandsConfig),
}

# agent name → Dockerfile 路径（相对于项目根目录）
AGENT_DOCKERFILE_MAP: dict[str, Path] = {
    "claude":   Path(__file__).resolve().parents[2] / "docker_cli" / "claude" / "Dockerfile.claude",
    "codex":    Path(__file__).resolve().parents[2] / "docker_cli" / "codex" / "Dockerfile.codex",
    "opencode": Path(__file__).resolve().parents[2] / "docker_cli" / "opencode" / "Dockerfile.opencode",
    "openhands": Path(__file__).resolve().parents[2] / "docker_cli" / "openhands" / "Dockerfile.openhands",
}

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
        # task_id → (base_image_tag, agent_image_tag)
        self._task_images: dict[str, tuple[str, str]] = {}
        self._driver: BaseAgentDriver | None = None
        self._setup_semaphore = asyncio.Semaphore(1)
        prompt_file = Path(__file__).parent / "sweci_prompt.jinja2"
        self.architect_prompt = load_prompt(prompt_file, {"role": "architect"})
        self.programmer_prompt = load_prompt(prompt_file, {"role": "programmer"})

    async def setup(self) -> None:
        download_sweci()
        self._taskids = await initialize_sweci()

        # 实例化 driver
        settings = get_settings()
        agent_name = settings.dataset.agent_name
        if agent_name not in AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent '{agent_name}', "
                f"expected one of {list(AGENT_REGISTRY.keys())}"
            )
        driver_cls, config_cls = AGENT_REGISTRY[agent_name]
        cfg = config_cls(
            api_key=settings.dataset.agent_api_key,
            model_name=settings.dataset.agent_model_name,
            base_url=settings.dataset.agent_base_url,
        )
        self._driver = driver_cls(cfg)

    async def teardown(self) -> None:
        """Dataset 级别资源回收：清理所有构建的 image。"""
        logger = get_logger()
        cleaned: set[str] = set()
        for base_tag, agent_tag in self._task_images.values():
            for tag in (agent_tag, base_tag):
                if tag not in cleaned:
                    await DockerBackend.remove_image(tag, missing_ok=True)
                    cleaned.add(tag)
                    logger.info("removed image: %s", tag)

    @property
    def task_ids(self) -> list[str]:
        return self._taskids

    async def setup_episode(self, episode: Episode) -> None:
        """Episode 级别初始化。

        加载 base image + 构建 agent image，exist_ok=True 自动跳过已存在的。
        """
        async with self._setup_semaphore:
            settings = get_settings()
            logger = get_logger()
            agent_name = settings.dataset.agent_name

            if agent_name not in AGENT_DOCKERFILE_MAP:
                raise ValueError(
                    f"Unknown agent '{agent_name}', "
                    f"expected one of {list(AGENT_DOCKERFILE_MAP.keys())}"
                )

            # 加载 base image
            tar_path = Path(settings.dataset.cache_dir) / "data" / episode.task_id / "image.tar.gz"
            base_image_tag = await DockerBackend.load_image(tar_path, exist_ok=True)

            # 构建 agent image
            agent_image_tag = f"sweci-{agent_name}-{base_image_tag.replace(':', '-').replace('/', '-')}:latest"
            dockerfile_path = AGENT_DOCKERFILE_MAP[agent_name]
            await DockerBackend.build_image(
                agent_image_tag,
                dockerfile_path,
                exist_ok=True,
                extra_params={"buildargs": {"BASE_IMAGE": base_image_tag}},
            )
            self._task_images[episode.task_id] = (base_image_tag, agent_image_tag)

            task_dir = Path(settings.dataset.cache_dir) / "processed" / episode.task_id
            current = json.loads((task_dir / "current" / "test_report.json").read_text(encoding="utf-8"))
            target = json.loads((task_dir / "target" / "test_report.json").read_text(encoding="utf-8"))
            current_passed_ids = set([t['nodeid'] for t in current['tests'] if t['outcome'] == 'passed'])
            target_passed_ids = set([t['nodeid'] for t in target['tests'] if t['outcome'] == 'passed'])
            episode.meta = {
                "base_passed": len(current_passed_ids),
                "target_passed": len(target_passed_ids),
                }
            episode.base_dir.mkdir(parents=True, exist_ok=True)
            (episode.base_dir / "meta.json").write_text(
                json.dumps(episode.meta, ensure_ascii=False, indent=4), encoding="utf-8"
                )

            logger.info("episode image ready: %s", agent_image_tag)

    async def teardown_episode(self, episode: Episode) -> None:
        pass

    def last_valid_turn(self, episode: Episode) -> Turn | None:
        for turn in reversed(episode.turns):
            if (turn.turn_dir / "non-passed").is_dir():
                return turn
        return None

    def _prepare_prop_input(self, episode: Episode) -> InputManifest:
        settings = get_settings()
        logger = get_logger() 

        _, agent_image_tag = self._task_images[episode.task_id]
        setup_cmds = self._driver.build_setup_commands()
        cmd, cmd_params = self._driver.build_command(
            self.architect_prompt,
            work_dir="/app/code"
            )
        
        task_dir = Path(settings.dataset.cache_dir) / "processed" / episode.task_id 
        
        code_dir, nonpassed_dir = None, None
        prev_turn = self.last_valid_turn(episode)
        if prev_turn:
            code_dir = prev_turn.turn_dir / "exec_output" / "code"
            nonpassed_dir = prev_turn.turn_dir / "non-passed"
        else:
            code_dir = task_dir / "current" / "code"
            nonpassed_dir = task_dir / "non-passed"

        return InputManifest(params={
            "image_tag": agent_image_tag,
            "commands": setup_cmds + [(cmd, cmd_params)],
            "copy_in": [
                (str(code_dir), "/app"), (str(nonpassed_dir), "/app"),
            ],
            "copy_out": [
                ("/app/requirement.xml", episode.turns[-1].prop_output_path)
            ]
        })

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        settings = get_settings()
        logger = get_logger() 

        _, agent_image_tag = self._task_images[episode.task_id]
        setup_cmds = self._driver.build_setup_commands()
        cmd, cmd_params = self._driver.build_command(
            self.programmer_prompt, 
            work_dir="/app/code"
            )

        task_dir = Path(settings.dataset.cache_dir) / "processed" / episode.task_id 

        code_dir = None
        prev_turn = self.last_valid_turn(episode)
        if prev_turn:
            code_dir = prev_turn.turn_dir / "exec_output" / "code"
        else:
            code_dir = task_dir / "current" / "code"

        return InputManifest(params={
            "image_tag": agent_image_tag,
            "commands": setup_cmds + [(cmd, cmd_params)],
            "copy_in": [
                (str(code_dir), "/app"),
                (str(episode.turns[-1].prop_output_path  / "requirement.xml"), "/app"),
                ],
            "copy_out": [
                ("/app/code", episode.turns[-1].exec_output_path)
            ]
        })

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        settings = get_settings()
        logger = get_logger() 

        _, agent_image_tag = self._task_images[episode.task_id]
        task_dir = Path(settings.dataset.cache_dir) / "processed" / episode.task_id 
        container_report = "/tmp/test_report.json"

        return InputManifest(params={
            "image_tag": agent_image_tag,
            "commands": [
                (
                    f"python -m pytest tests --color=no --tb=short --disable-warnings -rfE --rootdir=/app/code --json-report --json-report-file={container_report}",
                    {"workdir": "/app/code", "environment": {"PYTHONPATH": "src:."}},
                ),
            ],
            "copy_in": [
                (str(episode.turns[-1].turn_dir / "exec_output" / "code"), "/app"),
                ],
            "copy_out": [
                (container_report, episode.turns[-1].eval_output_path),
            ]
        })

    def validate_prop_output(self, manifest: OutputManifest) -> bool:
        if manifest.unexpected != "":
            return False
        for result in manifest.debug_info.copy_in:
            if result.returncode != 0:
                return False
        for result in manifest.debug_info.commands:
            if result.returncode != 0:
                return False
        for result in manifest.debug_info.copy_out:
            if result.returncode != 0:
                return False
        return True
    
    def validate_exec_output(self, manifest: OutputManifest) -> bool:
        if manifest.unexpected != "":
            return False
        for result in manifest.debug_info.copy_in:
            if result.returncode != 0:
                return False
        for result in manifest.debug_info.commands:
            if result.returncode != 0:
                return False
        for result in manifest.debug_info.copy_out:
            if result.returncode != 0:
                return False
        return True

    def validate_eval_output(self, manifest: OutputManifest) -> bool:
        return True

    async def _conclude(self, episode: Episode) -> dict:      
        test_returncode = episode.turns[-1].eval_output_manifest.debug_info.commands[0].returncode
        current_report_path = episode.turns[-1].eval_output_path / "test_report.json"
        if test_returncode >= 2 or not current_report_path.exists():
            return {"passed": False, "collapse": True, "gap": -1}

        settings = get_settings()
        cache_dir = Path(settings.dataset.cache_dir)
        target_report_path = cache_dir / "processed" / episode.task_id / "target" / "test_report.json"
        diff = generate_nonpassed_dir(
            current_report = current_report_path, 
            target_report = target_report_path,
            output_root = episode.turns[-1].turn_dir,
            )
        return {"passed": diff == 0, "collapse": False, "gap": diff}