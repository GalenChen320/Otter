# Environment Variable Configuration

All runtime parameters in Otter are configured via `.env` files. Use `otter run --env <file>` to specify which config file to use.

## Tracked vs Untracked

Each field is marked as either **tracked** or **untracked**:

- **tracked**: Affects experiment reproducibility. Written to `experiment.json` on the first run; validated for consistency on subsequent runs. If a tracked parameter changes, the framework will prompt the user for confirmation before overriding.
- **untracked**: Runtime-only parameters that do not affect reproducibility (e.g., concurrency, API keys, log level).

---

## Top-Level Parameters

Top-level parameters are written directly in the `.env` file without any prefix.

| Variable | Type | Default | Tracked | Description |
|---------|------|---------|---------|-------------|
| `EXPERIMENT_ID` | str | `"default"` | tracked | Unique experiment identifier; determines output directory `experiments/{id}` |
| `MAX_TURNS` | int | `1` | tracked | Maximum feedback iteration turns per episode |
| `SAMPLES_PER_PROBLEM` | int | `1` | tracked | Independent samples per problem |
| `DATASET_NAME` | str | `"mbppplus"` | tracked | Dataset name; determines which dataset to load (see Dataset Parameters below) |
| `PROPOSER_TYPE` | str \| None | `None` | tracked | Backend type for Proposer; leave empty to disable |
| `EXECUTOR_TYPE` | str \| None | `None` | tracked | Backend type for Executor; leave empty to disable |
| `EVALUATOR_TYPE` | str \| None | `None` | tracked | Backend type for Evaluator; leave empty to disable |
| `PROPOSER_CONCURRENCY` | int | `1` | untracked | Max concurrent Proposer executions |
| `EXECUTOR_CONCURRENCY` | int | `1` | untracked | Max concurrent Executor executions |
| `EVALUATOR_CONCURRENCY` | int | `1` | untracked | Max concurrent Evaluator executions |

---

## Dynamic Role–Backend Binding

Proposer, Executor, and Evaluator each select a backend type via their `XXX_TYPE` field. **The same backend type can be used by different roles**, with parameters isolated by role prefix.

The environment variable prefix follows the pattern `{ROLE}__`, where `{ROLE}` is the uppercase role name:

| Role | Type Field | Env Prefix |
|------|-----------|------------|
| Proposer | `PROPOSER_TYPE` | `PROPOSER__` |
| Executor | `EXECUTOR_TYPE` | `EXECUTOR__` |
| Evaluator | `EVALUATOR_TYPE` | `EVALUATOR__` |

For example: when `EXECUTOR_TYPE=chat_llm`, Executor parameters are read from the `EXECUTOR__` prefix (e.g., `EXECUTOR__api_key`, `EXECUTOR__model`). If `PROPOSER_TYPE=chat_llm`, Proposer parameters are read from the `PROPOSER__` prefix (e.g., `PROPOSER__api_key`, `PROPOSER__model`) — the two do not interfere.

When `XXX_TYPE` is left empty, the corresponding role is disabled and all parameters under its prefix are ignored.

---

## Backend Parameters

### chat_llm

For calling OpenAI-compatible LLM APIs.

| Variable | Type | Default | Required | Tracked | Description |
|---------|------|---------|----------|---------|-------------|
| `{ROLE}__api_key` | str | — | Yes | untracked | API key for the LLM provider |
| `{ROLE}__base_url` | str | — | Yes | tracked | Base URL of the LLM API endpoint |
| `{ROLE}__model` | str | — | Yes | tracked | Model name to use for generation |
| `{ROLE}__max_retries` | int | `3` | No | tracked | Max retry attempts on API failure (≥1) |
| `{ROLE}__retry_base_delay` | float | `1.0` | No | tracked | Base delay in seconds for exponential backoff |

Replace `{ROLE}` with `PROPOSER`, `EXECUTOR`, or `EVALUATOR` depending on which role uses this backend.

**Example** (Executor using chat_llm):

```env
EXECUTOR_TYPE=chat_llm
EXECUTOR__api_key=sk-xxx
EXECUTOR__base_url=https://api.deepseek.com
EXECUTOR__model=deepseek-chat
EXECUTOR__max_retries=3
EXECUTOR__retry_base_delay=1.0
```

### docker

For executing code inside Docker containers.

| Variable | Type | Default | Required | Tracked | Description |
|---------|------|---------|----------|---------|-------------|
| `{ROLE}__cpus` | float \| None | `1.0` | No | tracked | CPU cores allocated per container |
| `{ROLE}__memory` | str \| None | `"4096m"` | No | tracked | Container memory limit (e.g., `512m`, `1g`) |
| `{ROLE}__memory_swap` | str \| None | `"4096m"` | No | tracked | Container swap memory limit |
| `{ROLE}__memory_reservation` | str \| None | `"2048m"` | No | tracked | Container soft memory limit |
| `{ROLE}__network_mode` | str \| None | `"host"` | No | tracked | Container network mode (`none` disables networking) |
| `{ROLE}__device_read_bps` | str \| None | `"128m"` | No | tracked | Disk read rate limit |
| `{ROLE}__device_write_bps` | str \| None | `"128m"` | No | tracked | Disk write rate limit |
| `{ROLE}__timeout` | int | `10` | No | tracked | Per-command execution timeout in seconds |

All `str | None` fields above accept an empty value to set `None` (removing the corresponding limit).

**Example** (Evaluator using docker):

```env
EVALUATOR_TYPE=docker
EVALUATOR__cpus=1.0
EVALUATOR__memory=4096m
EVALUATOR__timeout=10
EVALUATOR__network_mode=host
```

---

## Dataset Parameters

Dataset parameters use the `DATASET__` prefix. The available fields depend on the value of `DATASET_NAME`.

### evalplus (HumanEval+)

| Variable | Type | Default | Tracked | Description |
|---------|------|---------|---------|-------------|
| `DATASET__cache_dir` | Path | `data/cache` | untracked | Local cache directory for HuggingFace datasets |

### mbppplus (MBPP+)

| Variable | Type | Default | Tracked | Description |
|---------|------|---------|---------|-------------|
| `DATASET__cache_dir` | Path | `data/cache` | untracked | Local cache directory for HuggingFace datasets |

### sweci (SWE-CI)

| Variable | Type | Default | Tracked | Description |
|---------|------|---------|---------|-------------|
| `DATASET__splitting` | str | `"default"` | tracked | Dataset splitting to use (e.g., `default`, `mini`) |
| `DATASET__agent_name` | str | `"opencode"` | tracked | AI CLI agent to use (`claude`, `codex`, `opencode`, `openhands`) |
| `DATASET__agent_api_key` | str | `""` | untracked | API key for the AI CLI agent |
| `DATASET__agent_model_name` | str | `""` | tracked | Model name for the AI CLI agent |
| `DATASET__agent_base_url` | str | `""` | tracked | Base URL for the AI CLI agent API |
| `DATASET__cache_dir` | Path | `data/cache` | untracked | Local cache directory for HuggingFace datasets |

---

## Logger Parameters

Logger parameters use the `LOG__` prefix.

| Variable | Type | Default | Tracked | Description |
|---------|------|---------|---------|-------------|
| `LOG__level` | str | `"INFO"` | untracked | Log verbosity; one of `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG__log_file` | Path \| None | `None` | untracked | Log file path; leave empty for stderr only |

---

## Full Example

A complete configuration for running MBPP+ evaluation:

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
EVALUATOR__memory=4096m
EVALUATOR__memory_swap=4096m
EVALUATOR__memory_reservation=2048m
EVALUATOR__network_mode=host
EVALUATOR__device_read_bps=128m
EVALUATOR__device_write_bps=128m
EVALUATOR__timeout=10

# ── Dataset ──
DATASET__cache_dir=data/cache

# ── Logger ──
LOG__level=INFO
LOG__log_file=
```

This configuration uses a DeepSeek model for code generation (Executor), runs tests in Docker containers (Evaluator), allows up to 3 feedback iterations per problem, and leaves Proposer disabled.
