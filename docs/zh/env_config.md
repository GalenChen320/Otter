# 环境变量配置说明

Otter 的所有运行参数通过 `.env` 文件配置。运行时通过 `otter run --env <file>` 指定使用哪个配置文件。

## tracked 与 untracked

每个字段被标记为 **tracked** 或 **untracked**：

- **tracked**：影响实验可复现性的参数。首次运行时会写入 `experiment.json`，续跑时会校验一致性。如果 tracked 参数发生变化，框架会提示用户确认是否覆盖。
- **untracked**：纯运行时参数，不影响实验结果的可复现性（如并发数、API Key、日志级别等）。

---

## 顶层参数

顶层参数直接写在 `.env` 中，无需任何前缀。

| 环境变量 | 类型 | 默认值 | tracked | 说明 |
|---------|------|--------|---------|------|
| `EXPERIMENT_ID` | str | `"default"` | tracked | 实验唯一标识符，决定输出目录 `experiments/{id}` |
| `MAX_TURNS` | int | `1` | tracked | 每个 Episode 的最大反馈迭代轮次 |
| `SAMPLES_PER_PROBLEM` | int | `1` | tracked | 每道题的独立采样次数 |
| `DATASET_NAME` | str | `"mbppplus"` | tracked | 数据集名称，决定加载哪个数据集（见下方数据集参数） |
| `PROPOSER_TYPE` | str \| None | `None` | tracked | Proposer 使用的 Backend 类型，留空表示禁用 |
| `EXECUTOR_TYPE` | str \| None | `None` | tracked | Executor 使用的 Backend 类型，留空表示禁用 |
| `EVALUATOR_TYPE` | str \| None | `None` | tracked | Evaluator 使用的 Backend 类型，留空表示禁用 |
| `PROPOSER_CONCURRENCY` | int | `1` | untracked | Proposer 最大并发数 |
| `EXECUTOR_CONCURRENCY` | int | `1` | untracked | Executor 最大并发数 |
| `EVALUATOR_CONCURRENCY` | int | `1` | untracked | Evaluator 最大并发数 |

---

## 角色与 Backend 的动态绑定

Proposer、Executor、Evaluator 三个角色各自通过 `XXX_TYPE` 选择使用哪种 Backend。**同一个 Backend 类型可以被不同角色使用**，参数通过角色前缀隔离。

环境变量的前缀规则为 `{ROLE}__`，其中 `{ROLE}` 是角色名的大写形式：

| 角色 | type 字段 | 环境变量前缀 |
|------|----------|-------------|
| Proposer | `PROPOSER_TYPE` | `PROPOSER__` |
| Executor | `EXECUTOR_TYPE` | `EXECUTOR__` |
| Evaluator | `EVALUATOR_TYPE` | `EVALUATOR__` |

例如：当 `EXECUTOR_TYPE=chat_llm` 时，Executor 的参数从 `EXECUTOR__` 前缀读取（如 `EXECUTOR__api_key`、`EXECUTOR__model`）。如果 `PROPOSER_TYPE=chat_llm`，则 Proposer 的参数从 `PROPOSER__` 前缀读取（如 `PROPOSER__api_key`、`PROPOSER__model`），两者互不干扰。

当 `XXX_TYPE` 留空时，对应角色被禁用，其前缀下的参数会被忽略。

---

## Backend 参数

### chat_llm

用于调用 OpenAI 兼容的 LLM API。

| 环境变量 | 类型 | 默认值 | 必填 | tracked | 说明 |
|---------|------|--------|------|---------|------|
| `{ROLE}__api_key` | str | — | 是 | untracked | LLM 服务的 API Key |
| `{ROLE}__base_url` | str | — | 是 | tracked | LLM API 的 Base URL |
| `{ROLE}__model` | str | — | 是 | tracked | 使用的模型名称 |
| `{ROLE}__max_retries` | int | `3` | 否 | tracked | API 调用失败时的最大重试次数（≥1） |
| `{ROLE}__retry_base_delay` | float | `1.0` | 否 | tracked | 指数退避的基础延迟（秒） |

其中 `{ROLE}` 根据实际绑定的角色替换为 `PROPOSER`、`EXECUTOR` 或 `EVALUATOR`。

**示例**（Executor 使用 chat_llm）：

```env
EXECUTOR_TYPE=chat_llm
EXECUTOR__api_key=sk-xxx
EXECUTOR__base_url=https://api.deepseek.com
EXECUTOR__model=deepseek-chat
EXECUTOR__max_retries=3
EXECUTOR__retry_base_delay=1.0
```

### docker

用于在 Docker 容器中执行代码。

| 环境变量 | 类型 | 默认值 | 必填 | tracked | 说明 |
|---------|------|--------|------|---------|------|
| `{ROLE}__cpus` | float \| None | `1.0` | 否 | tracked | 每个容器分配的 CPU 核心数 |
| `{ROLE}__memory` | str \| None | `"512m"` | 否 | tracked | 容器内存限制（如 `512m`、`1g`） |
| `{ROLE}__memory_swap` | str \| None | `"512m"` | 否 | tracked | 容器交换内存限制 |
| `{ROLE}__memory_reservation` | str \| None | `"256m"` | 否 | tracked | 容器软内存限制 |
| `{ROLE}__network_mode` | str \| None | `"none"` | 否 | tracked | 容器网络模式（`none` 表示禁用网络） |
| `{ROLE}__device_read_bps` | str \| None | `"128m"` | 否 | tracked | 磁盘读取速率限制 |
| `{ROLE}__device_write_bps` | str \| None | `"128m"` | 否 | tracked | 磁盘写入速率限制 |
| `{ROLE}__timeout` | int | `10` | 否 | tracked | 单条命令执行超时（秒） |

以上所有 `str | None` 类型的字段，留空即为 `None`（取消该项限制）。

**示例**（Evaluator 使用 docker）：

```env
EVALUATOR_TYPE=docker
EVALUATOR__cpus=1.0
EVALUATOR__memory=512m
EVALUATOR__timeout=10
EVALUATOR__network_mode=none
```

---

## 数据集参数

数据集参数统一使用 `DATASET__` 前缀。具体有哪些字段取决于 `DATASET_NAME` 的值。

### evalplus（HumanEval+）

| 环境变量 | 类型 | 默认值 | tracked | 说明 |
|---------|------|--------|---------|------|
| `DATASET__cache_dir` | Path | `data/cache` | untracked | HuggingFace 数据集的本地缓存目录 |

### mbppplus（MBPP+）

| 环境变量 | 类型 | 默认值 | tracked | 说明 |
|---------|------|--------|---------|------|
| `DATASET__cache_dir` | Path | `data/cache` | untracked | HuggingFace 数据集的本地缓存目录 |

### sweci（SWE-CI）

| 环境变量 | 类型 | 默认值 | tracked | 说明 |
|---------|------|--------|---------|------|
| `DATASET__splitting` | str | `"default"` | tracked | 使用的数据集划分（如 `default`、`mini`） |
| `DATASET__agent_name` | str | `"opencode"` | tracked | AI CLI 智能体（`claude`、`codex`、`opencode`、`openhands`） |
| `DATASET__agent_api_key` | str | `""` | untracked | AI CLI 智能体的 API Key |
| `DATASET__agent_model_name` | str | `""` | tracked | AI CLI 智能体使用的模型名称 |
| `DATASET__agent_base_url` | str | `""` | tracked | AI CLI 智能体的 API 基础 URL |
| `DATASET__cache_dir` | Path | `data/cache` | untracked | HuggingFace 数据集的本地缓存目录 |

---

## 日志参数

日志参数使用 `LOG__` 前缀。

| 环境变量 | 类型 | 默认值 | tracked | 说明 |
|---------|------|--------|---------|------|
| `LOG__level` | str | `"INFO"` | untracked | 日志级别，可选 `DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `LOG__log_file` | Path \| None | `None` | untracked | 日志文件路径，留空则仅输出到 stderr |

---

## 完整示例

以下是一个运行 MBPP+ 评测的完整配置：

```env
# ── Experiment ──
EXPERIMENT_ID=mbpp_exp_01
MAX_TURNS=3
SAMPLES_PER_PROBLEM=1

# ── Component Types ──
DATASET_NAME=mbppplus
PROPOSER_TYPE=
EXECUTOR_TYPE=chat_llm
EVALUATOR_TYPE=docker

# ── Concurrency ──
PROPOSER_CONCURRENCY=1
EXECUTOR_CONCURRENCY=5
EVALUATOR_CONCURRENCY=10

# ── Executor (chat_llm) ──
EXECUTOR__api_key=sk-your-api-key
EXECUTOR__base_url=https://api.deepseek.com
EXECUTOR__model=deepseek-chat
EXECUTOR__max_retries=3
EXECUTOR__retry_base_delay=1.0

# ── Evaluator (docker) ──
EVALUATOR__cpus=1.0
EVALUATOR__memory=512m
EVALUATOR__memory_swap=512m
EVALUATOR__memory_reservation=256m
EVALUATOR__network_mode=none
EVALUATOR__device_read_bps=128m
EVALUATOR__device_write_bps=128m
EVALUATOR__timeout=10

# ── Dataset ──
DATASET__cache_dir=data/cache

# ── Logger ──
LOG__level=INFO
LOG__log_file=
```

此配置的含义：使用 DeepSeek 模型生成代码（Executor），在 Docker 容器中运行测试（Evaluator），每道题最多迭代 3 轮，Proposer 未启用。
