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

<div align="center">

**English** | [中文](docs/zh/README.md)

</div>

<div align="center">

![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

</div>

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

See [Environment Variable Configuration](docs/en/env_config.md) for the full parameter reference.

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
