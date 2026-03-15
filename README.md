**English** | [中文](docs/README_zh.md)

<table>
  <tr>
    <td width="20%" align="center">
      <img src="assets/otter.jpg" width="100%" alt="logo">
    </td>
    <td valign="top" align="left">
      <h1>Otter</h1>
      <p>An Agent code evaluation framework with native multi-turn feedback iteration.</p>
    </td>
  </tr>
</table>

## Why Otter

Mainstream code benchmarks use snapshot-style evaluation — one input, one output. But real-world programming involves iterating based on compiler errors, test failures, and other feedback. **This feedback-driven iteration is the core of programming ability.**

Otter integrates evaluation feedback into the evaluation loop, letting agents work like real developers: write code → run → read errors → fix → run again, until the tests pass or the maximum number of turns is reached.

```
   ┌────────────────────────────┐
   ↓                            │
Proposer ───→ Executor ───→ Evaluator
                                │
                          Pass? │
                                ↓
                               End
```

## Quick Start

**Prerequisites**: Python >= 3.11, Docker

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API credentials

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
EXECUTOR__api_key=sk-xxx
EXECUTOR__base_url=https://api.openai.com/v1
EXECUTOR__model=gpt-4o
```

### Optional

| Variable | Default | Description |
|---|---|---|
| `PROPOSER_TYPE` | — | Proposer type, empty to disable |
| `EXECUTOR_TYPE` | — | Executor type (e.g. `chat_llm`), empty to disable |
| `EVALUATOR_TYPE` | — | Evaluator type (e.g. `docker`), empty to disable |
| `EXECUTOR__concurrency` | `1` | Max concurrent executor executions |
| `EXECUTOR__max_retries` | `3` | API call retry attempts |
| `EXECUTOR__retry_base_delay` | `1.0` | Retry backoff base delay (seconds) |
| `EXPERIMENT_ID` | `default` | Experiment ID, results saved to `experiments/{id}/` |
| `MAX_TURNS` | `1` | Max feedback iteration turns |
| `SAMPLES_PER_PROBLEM` | `1` | Independent samples per problem |
| `DATASET__dataset_name` | `mbppplus` | Dataset name |
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
    │   ├── prop_input/    # Proposer input (created if proposer enabled)
    │   ├── prop_output/   # Proposer output (created if proposer enabled)
    │   ├── exec_input/    # Executor input (created if executor enabled)
    │   ├── exec_output/   # Executor output (created if executor enabled)
    │   ├── eval_input/    # Evaluator input (created if evaluator enabled)
    │   ├── eval_output/   # Evaluator output (created if evaluator enabled)
    │   └── meta.json     # Turn verdict {"passed": true/false}
    ├── turn_2/           # Turn 2 (if turn 1 failed and max_turns > 1)
    │   └── ...
    └── ...
```

## Supported Datasets

| Dataset | Status | Description |
|---|---|---|
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | Fully supported | Function-level Python problems |
| [EvalPlus (HumanEval+)](https://github.com/evalplus/evalplus) | Fully supported | Rigorous LLM4Code benchmarks |
| [LiveCodeBench](https://livecodebench.github.io/) | Planned | Contamination-free live coding problems |
| [SWE-Bench](https://www.swebench.com/) | Planned | Real-world GitHub issue resolution |
| [Tau2Bench](https://github.com/sierra-research/tau2-bench) | Planned | Multi-turn agentic task evaluation |
| [TerminalBench](https://terminalbench.com/) | Planned | Terminal-based coding tasks |
| [SWE-CI](https://arxiv.org/abs/2603.03823) | Planned | CI-driven software engineering tasks |

## License

[Apache License 2.0](LICENSE)
