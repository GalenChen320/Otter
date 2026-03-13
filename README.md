**English** | [中文](docs/README_zh.md)

<table>
  <tr>
    <td width="20%" align="center">
      <img src="assets/otter.jpg" width="100%" alt="logo">
    </td>
    <td valign="top" align="left">
      <h1>Otter</h1>
      <p>An LLM code evaluation framework with native multi-turn feedback iteration.</p>
    </td>
  </tr>
</table>

## Why Otter

Mainstream code benchmarks use snapshot-style evaluation — one input, one output. But real-world programming involves iterating based on compiler errors, test failures, and other feedback. **This feedback-driven iteration is the core of programming ability.**

Otter integrates environment feedback into the evaluation loop, letting LLMs work like real developers: write code → run → read errors → fix → run again, until the tests pass or the maximum number of turns is reached.

```
      ┌──────────── Feedback ────────────┐
      │                                  │
      ↓                                  │
   Prompt ──→ LLM ──→ Code ──→ Environment ──→ Pass? ──→ Done
                                                 │
                                                 └─→ Failed, loop continues
```

## Quick Start

**Prerequisites**: Python >= 3.11, Docker

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your LLM API credentials

# Run evaluation
otter run
```

## Configuration

All parameters are managed via `.env` files. The CLI only accepts `--env` to select a config file:

```bash
otter run                    # uses .env by default
otter run --env .env.local   # specify a config file
```

### Required

```ini
LLM__api_key=sk-xxx
LLM__base_url=https://api.openai.com/v1
LLM__model=gpt-4o
```

### Optional

| Variable | Default | Description |
|---|---|---|
| `LLM__llm_type` | `openai_compatible` | LLM interface type |
| `LLM__concurrency` | `1` | Max concurrent LLM requests |
| `LLM__max_retries` | `3` | API call retry attempts |
| `LLM__retry_base_delay` | `1.0` | Retry backoff base delay (seconds) |
| `EXPERIMENT__experiment_id` | `default` | Experiment ID, results saved to `experiments/{id}/` |
| `EXPERIMENT__max_turns` | `1` | Max feedback iteration turns |
| `EXPERIMENT__samples_per_problem` | `1` | Independent samples per problem |
| `EXPERIMENT__feedback_strategy` | `error_message` | Feedback strategy (`minimal` / `error_message` / `progressive`) |
| `DATASET__dataset_name` | `mbppplus` | Dataset (`mbppplus` / `humaneval` / `apps`) |
| `DATASET__cache_dir` | `data/cache` | Dataset cache directory |
| `DOCKER__cpus` | `1.0` | Container CPU limit |
| `DOCKER__memory` | `512m` | Container memory limit |
| `DOCKER__network_mode` | `none` | Container network mode |
| `DOCKER__timeout` | `10` | Per-command execution timeout (seconds) |
| `LOG__level` | `INFO` | Log level |
| `LOG__log_file` | — | Log file path |

## Output Structure

Results are saved under `experiments/` as a directory tree, with a full record for each turn of each problem:

```
experiments/{experiment_id}/
└── {task_id}#{sample_id}/
    ├── turn_1/
    │   ├── llm_input/    # Prompt sent to the LLM
    │   ├── llm_output/   # Raw LLM response
    │   ├── env_input/    # Code script to execute
    │   ├── env_output/   # Execution results (stdout/stderr/returncode)
    │   └── meta.json     # Turn verdict {"passed": true/false}
    ├── turn_2/           # Turn 2 (if turn 1 failed and max_turns > 1)
    │   └── ...
    └── ...
```

## Supported Datasets

| Dataset | Status | Description |
|---|---|---|
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | Fully supported | Function-level Python problems |
| HumanEval | In development | Function-level Python problems |
| APPS | In development | Competition-level programming problems |

## License

[Apache License 2.0](LICENSE)
