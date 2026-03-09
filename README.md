# otter

LLM 代码生成能力评测框架。加载编程题数据集，异步并发调用 LLM 生成代码解答。

## 项目结构

```
otter/
├── pyproject.toml                  # 项目配置与依赖
├── .env.example                    # 环境变量模板（待填写）
├── .env                            # 实际环境变量（已 gitignore）
├── data/cache/                     # HuggingFace 数据集本地缓存
└── src/otter/
    ├── __init__.py
    ├── pipeline.py                 # 主流程编排（入口）
    ├── logger.py                   # 全局单例 logger
    ├── config/
    │   ├── __init__.py
    │   └── setting.py              # pydantic-settings 配置管理
    ├── datasets/
    │   ├── __init__.py             # 导出 BaseDataset, APPSDataset, HumanEvalDataset, MBPPPlusDataset
    │   ├── base.py                 # BaseDataset 抽象基类
    │   ├── mbppplus.py             # MBPP+ 数据集（完整实现）
    │   ├── humaneval.py            # HumanEval 数据集（继承 BaseDataset，仅实现 load）
    │   └── apps.py                 # APPS 数据集（继承 BaseDataset，仅实现 load）
    └── llm/
        ├── __init__.py             # 导出 BaseLLM, OpenAICompatibleLLM
        ├── base.py                 # BaseLLM 抽象基类（模板方法：重试、指数退避、日志追踪、ping 检测）
        └── openai_compatible.py    # OpenAI 兼容接口 LLM 实现
```

## 核心功能

### 配置管理

`config/setting.py` 使用 `pydantic-settings` 从 `.env` 文件加载配置，`Settings` 包含：

| 配置类 | 环境变量前缀 | 字段 |
|---|---|---|
| `LLMSettings` | `LLM__` | `api_key`, `base_url`, `model`, `concurrency`(=10), `samples_per_problem`(=1), `max_retries`(=3), `retry_base_delay`(=1.0) |
| `LoggerSettings` | `LOG__` | `level`(=INFO), `log_file`(=None) |
| `DatasetSettings` | — | `cache_dir`, `dataset_name`(=mbppplus) |
| `ExecutorSettings` | `EXECUTOR__` | `concurrency`(=5), `timeout`(=10)（已定义，尚未接入） |

### 全局日志

`logger.py` 基于 `LoggerSettings` 构建全局单例 logger，输出到 stderr，可选同时写入文件。格式：`时间 | 级别 | 模块名 | 消息`。

```python
from otter.logger import logger
```

### LLM 调用层

- **`BaseLLM`**（模板方法模式）：`generate` 为具体方法，内置重试与指数退避，每次尝试记录日志（任务 ID、第几次尝试、成功/失败及原因）。子类只需实现 `_generate`。另提供 `ping()` 方法快速检测 API 配置是否可用。
- **`OpenAICompatibleLLM`**：通过 `AsyncOpenAI` 客户端调用兼容 OpenAI 接口的服务（OpenAI / DeepSeek / vLLM / Ollama 等）。

### 数据集

- **`BaseDataset`** 抽象基类，规范 `load`、`__len__`、`__getitem__`、`make_prompt` 四个接口。
- **`MBPPPlusDataset`**：完整实现，加载 `evalplus/mbppplus`，将题目解析为 `MBPPPlusProblem` dataclass。
- **`HumanEvalDataset`**：继承 `BaseDataset`，仅实现 `load`，缺少 `__len__`、`__getitem__`、`make_prompt`。
- **`APPSDataset`**：继承 `BaseDataset`，仅实现 `load`，缺少 `__len__`、`__getitem__`、`make_prompt`。

### 主流程

`pipeline.py` 编排完整流程：
1. `create_dataset()` 根据 `dataset_name` 创建数据集实例并加载
2. `create_llm()` 创建 LLM 客户端
3. `ping()` 检测 API 可用性，失败直接退出
4. `run()` 异步并发调用 LLM 生成代码，通过 Semaphore 控制并发，返回结果列表

## 未实现

- **代码执行与评测**：`ExecutorSettings` 已定义，执行器尚未实现
- **HumanEval / APPS 数据集**：继承了 `BaseDataset` 但接口未完整实现
- **结果持久化**：生成结果目前仅在内存中，未写入文件
- **结果分析与统计**：无 pass@k 等指标计算

## 已知问题

1. **`create_llm()` 引用不存在的字段**：`settings.llm.llm_type` 在 `LLMSettings` 中未定义
2. **`APPSDataset.load()` 引用不存在的配置**：`settings.dataset.apps` 在 `DatasetSettings` 中未定义
3. **`HumanEvalDataset.load()` 使用 `dataset_name` 加载**：硬编码为 `settings.dataset.dataset_name` 而非固定的 HumanEval 数据集标识
4. **`.env.example` 为空**：未提供环境变量配置示例

## 依赖

- Python >= 3.11
- `datasets==4.6.1` — HuggingFace Datasets
- `pydantic-settings==2.13.1` — 配置管理
- `openai==2.26.0` — OpenAI API 客户端