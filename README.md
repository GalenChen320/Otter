<table>
  <tr>
    <td width="20%" align="center">
      <img src="assets/otter.jpg" width="100%" alt="logo">
    </td>
    <td valign="top" align="left">
      <h1>Otter</h1>
      <p>原生支持多轮反馈迭代的 LLM 代码能力评测框架。</p>
    </td>
  </tr>
</table>

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

## 设计理念

Otter 的核心循环：

```
Prompt/Feedback → LLM → Response → Environment → Observation → Reviewer → Feedback（循环）
```

**文件编排架构**：Pipeline 在 Episode 级别编排各组件，各阶段之间通过文件系统通信。Turn 不存储实际内容，只持有目录路径。每个阶段的输出写入对应目录，下一个阶段从目录中读取。这使得不同类型的数据集（函数级 / 仓库级 / Agent 级）可以用完全不同的文件格式，而 Pipeline 代码不需要任何改动。

```
experiments/{experiment_id}/{task_id}#{sample_id}/
└── turn_N/
    ├── input/          # Dataset 写入的输入
    ├── response/       # Dataset 写入的 LLM 响应
    ├── observation/    # Dataset 写入的环境执行结果
    └── meta.json       # passed 状态，标记 turn 完成
```

## 配置

通过 `.env` 文件管理所有配置，CLI 仅接受 `--env` 参数选择配置文件。

| 配置类 | 环境变量前缀 | 字段 |
|---|---|---|
| `LLMSettings` | `LLM__` | `api_key`\*, `base_url`\*, `model`\*, `response_format`(=openai_compatible), `concurrency`(=10), `samples_per_problem`(=1), `max_retries`(=3), `retry_base_delay`(=1.0) |
| `ExperimentSettings` | `EXPERIMENT__` | `experiment_id`(=default), `max_turns`(=1), `feedback_strategy`(=error_message) |
| `DatasetSettings` | `DATASET__` | `cache_dir`, `dataset_name`(=mbppplus) |
| `DockerSettings` | `DOCKER__` | `cpus`(=1.0), `memory`(=512m), `memory_swap`(=512m), `memory_reservation`(=256m), `device_read_bps`(=128m), `device_write_bps`(=128m), `timeout`(=10) |
| `LoggerSettings` | `LOG__` | `level`(=INFO), `log_file`(=None) |

\* 为必填项，无默认值。

## 核心概念

| 概念 | 说明 |
|---|---|
| **Dataset** | 编程题目集合，同时是适配中心：负责文件读写、构建 ExecSpec、判定 passed |
| **Episode** | 一道题目的完整多轮对话过程 |
| **Turn** | 单轮交互的索引：四个目录路径（input / response / observation）+ passed 判定 |
| **Environment** | 执行环境，接收 ExecSpec 返回 ExecutionObservation，无状态纯类方法 |
| **ExecSpec** | 执行规格，由 Dataset 构建，描述注入什么文件、执行什么命令 |
| **ExecutionObservation** | 环境执行的原始观测（stdout / stderr / returncode / timed_out） |
| **Store** | 管理目录结构，分配 Turn 路径，通过 meta.json 标记完成状态 |

## 模块说明

### Pipeline (`pipeline.py`)

纯编排层，不处理文件内容，不做适配。按顺序调用各组件：

```
allocate_turn → write_input → make_messages → generate → write_response
→ to_exec_spec → execute → write_observation → judge → save_meta
```

### Dataset (`dataset/`)

适配中心，每个数据集实现 BaseDataset 的全部接口：

- **生命周期**：`setup()` / `teardown()`（Dataset 级）、`setup_episode()` / `teardown_episode()`（Episode 级）
- **编排接口**：`write_input`、`make_messages`、`write_response`、`to_exec_spec`、`write_observation`、`judge`
- **已实现**：MBPPPlus（完整）、HumanEval / APPS（仅 load）

### Environment (`environment/`)

- **`DockerEnvironment`**：无状态类方法。`execute(spec)` 按规格创建容器 → 注入文件 → 执行命令 → 导出文件 → 销毁容器。资源限制从 DockerSettings 读取。
- **`build_image()` / `remove_image()`**：镜像管理工具方法，由 Dataset 按需调用。

### LLM (`llm/`)

- **`BaseLLM`**：模板方法模式，`generate()` 内置重试与指数退避，子类实现 `_generate()`。
- **`OpenAICompatibleLLM`**：通过 AsyncOpenAI 调用兼容接口（OpenAI / DeepSeek / vLLM / Ollama 等）。

### Store (`store.py`)

管理实验输出的目录结构。`allocate_turn` 创建目录并 append Turn 到 Episode，`save_meta` 写入 meta.json 标记完成，`load_episodes` 扫描目录重建状态。断点续跑：没有 meta.json 的 Turn 视为未完成，跳过。

### 配置 (`config/setting.py`)

pydantic-settings 从 `.env` 加载，延迟初始化。`set_env_file()` → `init_settings()` → `get_settings()`。

## 依赖

- Python >= 3.11
- `datasets` — HuggingFace Datasets
- `pydantic-settings` — 配置管理
- `openai` — OpenAI API 客户端
- `typer` — CLI 框架
- `docker` — Docker SDK
