# CodeEval

LLM 代码生成能力评测框架。通过加载编程题数据集、调用 LLM 生成代码解答，并将结果保存为 JSONL 文件，用于后续评测。

## 项目结构

```
CodeEval/
├── pyproject.toml                  # 项目配置与依赖
├── .env.example                    # 环境变量模板（待填写）
├── .env                            # 实际环境变量（已 gitignore）
├── data/cache/                     # HuggingFace 数据集本地缓存
└── src/code_eval/
    ├── __init__.py
    ├── pipeline.py                 # 主流程编排（入口）
    ├── logger.py                   # 全局单例 logger
    ├── config/
    │   ├── __init__.py
    │   └── setting.py              # pydantic-settings 配置管理
    ├── datasets/
    │   ├── __init__.py             # 导出 BaseDataset, APPSDataset, HumanEvalDataset, MBPPPlusDataset
    │   ├── base.py                 # BaseDataset 抽象基类
    │   ├── mbppplus.py             # MBPP+ 数据集实现（完整）
    │   ├── humaneval.py            # HumanEval 数据集（未完成，未继承 BaseDataset）
    │   └── apps.py                 # APPS 数据集（未完成，未继承 BaseDataset）
    └── llm/
        ├── __init__.py             # 导出 BaseLLM, OpenAICompatibleLLM
        ├── base.py                 # BaseLLM 抽象基类（模板方法模式，内置重试与指数退避）
        └── openai_compatible.py    # OpenAI 兼容接口 LLM 实现
```

## 核心功能

### 已实现

- **配置管理** (`config/setting.py`)：使用 `pydantic-settings` 从 `.env` 文件加载配置。`Settings` 包含：
  - `DatasetSettings` — 数据集名称、缓存路径
  - `LLMSettings`（`LLM__` 前缀）— API 密钥、base_url、模型名、并发数、每题采样数、重试次数、重试退避基础延迟
  - `LoggerSettings`（`LOG__` 前缀）— 日志级别、可选日志文件路径
  - `ExecutorSettings`（`EXECUTOR__` 前缀）— Docker 并发数、超时（已定义，尚未接入）
- **全局日志** (`logger.py`)：基于 `LoggerSettings` 构建的全局单例 logger，输出到 stderr，可选同时写入文件。使用方式：`from code_eval.logger import logger`。
- **数据集抽象** (`datasets/base.py`)：定义 `BaseDataset` 抽象基类，规范 `load`、`__len__`、`__getitem__`、`make_prompt` 四个接口。
- **MBPP+ 数据集** (`datasets/mbppplus.py`)：完整实现了 `evalplus/mbppplus` 数据集的加载与 prompt 构造，将每道题解析为 `MBPPPlusProblem` dataclass。是目前唯一完整实现 `BaseDataset` 的数据集。
- **LLM 抽象** (`llm/base.py`)：`BaseLLM` 采用模板方法模式，`generate` 为具体方法，内置重试与指数退避（参数从 `settings.llm` 读取），子类只需实现 `_generate`。
- **OpenAI 兼容 LLM** (`llm/openai_compatible.py`)：通过 `AsyncOpenAI` 客户端调用任意兼容 OpenAI 接口的服务（OpenAI / DeepSeek / vLLM / Ollama 等）。
- **主流程** (`pipeline.py`)：`create_dataset()` 根据配置名（`humaneval` / `apps` / `mbppplus`）创建数据集实例；`create_llm()` 创建 LLM 客户端；`run()` 异步并发生成代码解答，支持断点续跑（检查已有 JSONL 结果跳过已完成任务），通过 Semaphore 控制并发。

### 未实现

- **代码执行与评测**：`ExecutorSettings` 已定义（Docker 并发数、超时时间），但执行器本身尚未实现。生成的代码无法自动运行和判定正确性。
- **HumanEval 数据集**：`humaneval.py` 有 `load` 方法但未继承 `BaseDataset`，缺少 `__len__`、`__getitem__`、`make_prompt` 实现。
- **APPS 数据集**：`apps.py` 未继承 `BaseDataset`，`load()` 中引用了不存在的 `settings.dataset.apps`，缺少完整的数据集接口实现。
- **结果分析与统计**：无 pass@k 等指标计算功能。

## 已知问题

1. **`Settings` 缺少 `result` 属性**：`pipeline.py` 的 `main()` 中引用了 `settings.result.result_dir`，但 `Settings` 类中未定义该字段，运行时会报 `AttributeError`。
2. **`create_llm()` 引用不存在的字段**：`settings.llm.llm_type` 在 `LLMSettings` 中未定义。
3. **`HumanEvalDataset` / `APPSDataset` 未继承 `BaseDataset`**：已在 `datasets/__init__.py` 中导出并在 `pipeline.py` 的 `create_dataset()` 中注册，但实际不满足 `BaseDataset` 接口，运行时会出错。
4. **`APPSDataset.load()` 引用不存在的配置**：`settings.dataset.apps` 在 `DatasetSettings` 中未定义。
5. **`.env.example` 为空**：未提供环境变量配置示例，不利于其他开发者上手。

## 依赖

- Python >= 3.11
- `datasets==4.6.1` — HuggingFace Datasets
- `pydantic-settings==2.13.1` — 配置管理
- `openai==2.26.0` — OpenAI API 客户端
