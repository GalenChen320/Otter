# Otter

<div style="display: flex; align-items: center; gap: 20px;">
  
  <!-- 左边图片 30% -->
  <div style="flex: 0 0 20%;">
    <img src="assets/otter.jpg" alt="图片" style="width: 100%; border-radius: 8px;">
  </div>

  <!-- 右边文字 -->
  <div style="flex: 1;">
    <h1 style="text-align: left; margin: 0 0 8px 0; font-size: 2.5em;">
      Otter
    </h1>
    <p style="margin: 0; font-size: 0.9em; color: #666;">
      这里是副标题小字，简短描述一下内容
    </p>
  </div>

</div>


原生支持多轮对话反馈的 LLM 代码能力评测框架。

## 快速开始

```bash
# 安装
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM API 配置

# 运行评测
otter run

# 指定环境变量文件
otter run --env .env.local
```

## 项目结构

```
Otter/
├── pyproject.toml                  # 项目配置与依赖
├── .env.example                    # 环境变量模板
├── .env                            # 实际环境变量（已 gitignore）
├── data/cache/                     # HuggingFace 数据集本地缓存
└── src/otter/
    ├── __init__.py
    ├── cli.py                      # CLI 入口（typer）
    ├── pipeline.py                 # 主流程编排
    ├── episode.py                  # 核心数据类：Turn, Episode, ExecutionResult
    ├── logger.py                   # 全局日志（延迟初始化）
    ├── config/
    │   ├── __init__.py
    │   └── setting.py              # pydantic-settings 配置管理（延迟初始化）
    ├── dataset/
    │   ├── __init__.py
    │   ├── base.py                 # BaseDataset 抽象基类
    │   ├── mbppplus.py             # MBPP+ 数据集（完整实现）
    │   ├── humaneval.py            # HumanEval 数据集（仅实现 load）
    │   └── apps.py                 # APPS 数据集（仅实现 load）
    └── llm/
        ├── __init__.py
        ├── base.py                 # BaseLLM 抽象基类（重试、指数退避、ping 检测）
        └── openai_compatible.py    # OpenAI 兼容接口 LLM 实现
```

## 配置

通过 `.env` 文件管理配置，支持 `--env` 参数指定不同的配置文件。

| 配置类 | 环境变量前缀 | 字段 |
|---|---|---|
| `LLMSettings` | `LLM__` | `api_key`*, `base_url`*, `model`*, `response_format`(=openai_compatible), `concurrency`(=10), `samples_per_problem`(=1), `max_retries`(=3), `retry_base_delay`(=1.0) |
| `ExperimentSettings` | `EXPERIMENT__` | `max_turns`(=1), `feedback_strategy`(=error_message) |
| `DatasetSettings` | `DATASET__` | `cache_dir`, `dataset_name`(=mbppplus) |
| `LoggerSettings` | `LOG__` | `level`(=INFO), `log_file`(=None) |
| `ExecutorSettings` | `EXECUTOR__` | `concurrency`(=5), `timeout`(=10) |

*标 `*` 的为必填项，无默认值。

## 核心概念

| 概念 | 说明 |
|---|---|
| **Dataset** | 编程题目集合（MBPPPlus / HumanEval / APPS） |
| **Episode** | 一道题目的完整多轮对话过程 |
| **Turn** | 单轮交互：prompt → response → 执行 → feedback |
| **ExecutionResult** | 代码执行结果（通过/报错/超时） |

## 模块说明

### CLI (`cli.py`)

基于 typer 的命令行入口，负责解析参数、初始化配置和日志，调用 pipeline。

### 配置 (`config/setting.py`)

使用 pydantic-settings 从 `.env` 文件加载配置。采用延迟初始化模式，确保 CLI 参数（如 `--env`）能在配置构建前生效。

- `set_env_file(path)` — 设置 env 文件路径
- `init_settings()` — 构建 Settings 单例
- `get_settings()` — 获取 Settings 单例

### 日志 (`logger.py`)

延迟初始化的全局日志，输出到 stderr，可选写入文件。格式：`时间 | 级别 | 模块名 | 消息`。

- `init_logger()` — 构建 logger
- `get_logger()` — 获取 logger

### LLM 调用层 (`llm/`)

- **`BaseLLM`**（模板方法模式）：`generate()` 内置重试与指数退避，子类只需实现 `_generate()`。`ping()` 方法检测 API 可用性。
- **`OpenAICompatibleLLM`**：通过 `AsyncOpenAI` 调用兼容 OpenAI 接口的服务（OpenAI / DeepSeek / vLLM / Ollama 等）。

### 数据集 (`datasets/`)

- **`BaseDataset`** — 抽象基类，定义 `load`、`__len__`、`__getitem__`、`make_prompt` 接口。
- **`MBPPPlusDataset`** — 完整实现，加载 `evalplus/mbppplus`。
- **`HumanEvalDataset`** / **`APPSDataset`** — 仅实现 `load()`，其余接口待补全。

### 多轮评测数据类 (`episode.py`)

- **`ExecutionResult`** — 代码执行结果：`passed`, `stdout`, `stderr`, `timed_out`
- **`Turn`** — 单轮交互：`turn_number`, `prompt`, `response`, `code`, `execution_result`；`passed` 为推导属性
- **`Episode`** — 完整对话过程：`task_id`, `max_turns`, `turns`；`resolved`（最后一轮是否通过）、`exhausted`（是否达到最大轮次）为推导属性

### Pipeline (`pipeline.py`)

编排评测流程：
1. `create_dataset()` — 根据配置创建数据集
2. `create_llm()` — 根据 `response_format` 创建 LLM 客户端
3. `run()` — 异步并发调用 LLM 生成代码，Semaphore 控制并发
4. `main()` — 串联完整流程：加载数据集 → ping LLM → 并发生成

## 未实现

- **代码执行环境**：`ExecutorSettings` 已定义，执行器尚未实现
- **Feedback 生成**：将执行结果转化为下一轮 prompt 的机制
- **多轮 Pipeline**：当前为单轮生成，多轮 Episode 驱动的循环尚未实现
- **HumanEval / APPS 数据集**：仅实现 `load()`，缺少完整接口
- **结果持久化**：生成结果仅在内存中
- **评分与报告**：无 pass@k 等指标计算

## 依赖

- Python >= 3.11
- `datasets==4.6.1` — HuggingFace Datasets
- `pydantic-settings==2.13.1` — 配置管理
- `openai==2.26.0` — OpenAI API 客户端
- `typer>=0.15.0` — CLI 框架