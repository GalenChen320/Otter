# 开发者扩展指南

本文档说明如何为 Otter 添加新的 **Backend** 或 **Dataset**。

Otter 的三个角色（Proposer / Executor / Evaluator）是固定的，不需要扩展。真正变化的是角色绑定的 Backend 和评测使用的 Dataset。

---

## 添加新 Backend

Backend 是纯能力层，接收原生类型参数、返回原生类型结果，不感知框架概念。

以添加一个名为 `my_api` 的 Backend 为例，需要修改 4 个文件：

### 第一步：实现 Backend 类

在 `src/otter/backend/` 下新建 `my_api.py`：

```python
class MyAPIBackend:
    def __init__(self, endpoint: str, token: str, timeout: int):
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout

    async def run(self, prompt: str) -> str:
        # 调用外部 API，返回结果
        ...
```

**约定**：
- `__init__` 接收配置参数（纯 Python 类型）
- `run` 方法是 async 的，参数和返回值都是原生类型
- 不要在 Backend 中导入任何框架模块

### 第二步：注册 Backend 配置

在 `src/otter/config/backend_settings.py` 中：

1. 新增一个继承 `BackendSettings` 的配置类
2. 将其注册到 `BACKEND_SETTINGS_REGISTRY`

```python
class MyAPISettings(BackendSettings):
    endpoint: str = tracked_field(
        description="API endpoint URL"
    )
    token: str = untracked_field(
        description="Authentication token"
    )
    timeout: int = tracked_field(
        default=30,
        description="Request timeout in seconds"
    )


BACKEND_SETTINGS_REGISTRY: dict[str, type[BackendSettings]] = {
    "chat_llm": ChatLLMSettings,
    "docker": DockerSettings,
    "my_api": MyAPISettings,       # ← 新增
}
```

注册后，框架会根据角色前缀自动从环境变量读取配置。例如当 `EXECUTOR_TYPE=my_api` 时，会从 `EXECUTOR__endpoint`、`EXECUTOR__token`、`EXECUTOR__timeout` 读取参数。

### 第三步：注册 Backend 工厂

在 `src/otter/backend/__init__.py` 的 `create_backend` 函数中添加一个 `case` 分支：

```python
from otter.backend.my_api import MyAPIBackend

def create_backend(backend_type: str, settings):
    match backend_type:
        case "chat_llm":
            ...
        case "docker":
            ...
        case "my_api":                    # ← 新增
            return MyAPIBackend(
                endpoint=settings.endpoint,
                token=settings.token,
                timeout=settings.timeout,
            )
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")
```

### 第四步：注册 extract / pack 函数

在 `src/otter/role.py` 中，需要为新 Backend 定义两个函数并注册到分发字典：

- **extract**：从 `InputManifest` 读取文件内容，转换为 Backend `run` 方法的参数
- **pack**：将 Backend 返回的结果写入文件，构建 `OutputManifest`

```python
from otter.backend.my_api import MyAPIBackend

def extract_for_my_api(manifest: InputManifest, episode: Episode) -> dict:
    prompt = manifest.prompt_file.read_text(encoding="utf-8")
    return {"prompt": prompt}

def pack_my_api(result: str, output_dir: Path) -> OutputManifest:
    output_file = output_dir / "response.txt"
    output_file.write_text(result, encoding="utf-8")
    return OutputManifest(exec_output_file=output_file)

EXTRACT_DISPATCH = {
    ChatLLMBackend: extract_for_chat_llm,
    DockerBackend: extract_for_docker,
    MyAPIBackend: extract_for_my_api,     # ← 新增
}

PACK_DISPATCH = {
    ChatLLMBackend: pack_chat_llm,
    DockerBackend: pack_docker,
    MyAPIBackend: pack_my_api,            # ← 新增
}
```

### 完成

此时在 `.env` 中将任意角色的 type 设为 `my_api` 即可使用：

```env
EXECUTOR_TYPE=my_api
EXECUTOR__endpoint=https://api.example.com
EXECUTOR__token=xxx
EXECUTOR__timeout=30
```

---

## 添加新 Dataset

Dataset 负责加载数据、构建 Manifest、判定通过与否。

以添加一个名为 `leetcode` 的 Dataset 为例，需要修改 4 个文件：

### 第一步：实现 Dataset 类

在 `src/otter/dataset/` 下新建 `leetcode.py`，继承 `BaseDataset`：

```python
from otter.dataset.base import BaseDataset
from otter.episode import Episode, InputManifest

class LeetCodeDataset(BaseDataset):

    # ── 必须实现的属性 ──

    @property
    def task_ids(self) -> list[str]:
        """返回所有题目 ID 列表。"""
        return list(self._problems.keys())

    # ── 生命周期（按需重写）──

    async def setup(self) -> None:
        """整个评测开始前调用一次。加载数据集、构建 Docker 镜像等。"""
        ...

    async def teardown(self) -> None:
        """整个评测结束后调用一次。清理镜像等资源。"""
        ...

    async def setup_episode(self, episode: Episode) -> None:
        """每道题开始前调用。"""
        ...

    async def teardown_episode(self, episode: Episode) -> None:
        """每道题结束后调用。"""
        ...

    # ── 必须实现的 Pipeline 接口 ──

    def _prepare_exec_input(self, episode: Episode) -> InputManifest:
        """构建 Executor 的输入。

        第一轮写题目 prompt，后续轮次写反馈信息。
        将文件写入 turn.exec_input_path，返回 InputManifest。
        """
        ...

    def _prepare_eval_input(self, episode: Episode) -> InputManifest:
        """构建 Evaluator 的输入。

        从 turn.exec_output_manifest 读取 Executor 的输出，
        提取代码、拼接测试用例，写入脚本文件，返回 InputManifest。
        """
        ...

    async def _judge(self, episode: Episode) -> None:
        """判定本轮是否通过。

        从 turn.eval_output_manifest 读取 Evaluator 的输出，
        设置 turn.passed = True / False。
        """
        ...
```

**`BaseDataset` 接口说明**：

| 方法 | 类型 | 说明 |
|------|------|------|
| `task_ids` | 抽象属性 | 返回所有题目 ID |
| `setup` / `teardown` | 可选重写 | Dataset 级别生命周期，评测前后各调用一次 |
| `setup_episode` / `teardown_episode` | 可选重写 | Episode 级别生命周期，每道题前后各调用一次 |
| `_prepare_exec_input` | 必须实现 | 构建 Executor 输入的 `InputManifest` |
| `_prepare_eval_input` | 必须实现 | 构建 Evaluator 输入的 `InputManifest` |
| `_judge` | 必须实现 | 判定本轮是否通过，设置 `turn.passed` |

**注意**：不要直接重写 `prepare_exec_input` / `prepare_eval_input` / `make_judgement`，这些是模板方法，负责保存 manifest 和调用子类的 `_prepare_xxx` / `_judge`。

### 第二步：注册 Dataset 配置

在 `src/otter/config/dataset_settings.py` 中：

1. 新增一个继承 `DatasetSettings` 的配置类
2. 将其注册到 `DATASET_SETTINGS_REGISTRY`

```python
class LeetcodeSettings(DatasetSettings):
    cache_dir: Path = untracked_field(
        default=ROOT_DIR / "data" / "cache",
        description="Local cache directory"
    )
    difficulty: str = tracked_field(
        default="medium",
        description="Problem difficulty filter"
    )


DATASET_SETTINGS_REGISTRY: dict[str, type[DatasetSettings]] = {
    "evalplus": EvalplusSettings,
    "mbppplus": MbppplusSettings,
    "leetcode": LeetcodeSettings,         # ← 新增
}
```

注册后，框架会从 `DATASET__` 前缀读取配置。例如 `DATASET__difficulty=hard`。

### 第三步：注册 Dataset 工厂

在 `src/otter/pipeline.py` 的 `create_dataset` 函数中添加一个 `case` 分支：

```python
from otter import dataset

def create_dataset() -> dataset.BaseDataset:
    settings = get_settings()
    output_dir = settings.output_dir
    match settings.dataset_name:
        case "evalplus": return dataset.EvalPlusDataset(output_dir)
        case "mbppplus": return dataset.MBPPPlusDataset(output_dir)
        case "leetcode": return dataset.LeetCodeDataset(output_dir)  # ← 新增
        case _:
            raise ValueError(f"unknown dataset: {settings.dataset_name}")
```

### 第四步：导出类

在 `src/otter/dataset/__init__.py` 中导入并导出新类：

```python
from .leetcode import LeetCodeDataset

__all__ = [
    "BaseDataset",
    "APPSDataset",
    "EvalPlusDataset",
    "MBPPPlusDataset",
    "LeetCodeDataset",    # ← 新增
]
```

### 完成

在 `.env` 中设置 `DATASET_NAME=leetcode` 即可使用新数据集。

---

## 扩展清单速查

### 添加新 Backend

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `backend/my_api.py` | 新建 Backend 类 |
| 2 | `config/backend_settings.py` | 新增 Settings 类 + 注册到 `BACKEND_SETTINGS_REGISTRY` |
| 3 | `backend/__init__.py` | 在 `create_backend` 中添加 `case` 分支 |
| 4 | `role.py` | 添加 extract/pack 函数 + 注册到分发字典 |

### 添加新 Dataset

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `dataset/leetcode.py` | 新建 Dataset 子类 |
| 2 | `config/dataset_settings.py` | 新增 Settings 类 + 注册到 `DATASET_SETTINGS_REGISTRY` |
| 3 | `pipeline.py` | 在 `create_dataset` 中添加 `case` 分支 |
| 4 | `dataset/__init__.py` | 导入并导出新类 |
