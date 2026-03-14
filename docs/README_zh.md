[English](../README.md) | **中文**

<table>
  <tr>
    <td width="20%" align="center">
      <img src="../assets/otter.jpg" width="100%" alt="logo">
    </td>
    <td valign="top" align="left">
      <h1>Otter</h1>
      <p>原生支持多轮反馈迭代的智能体代码能力评测框架。</p>
    </td>
  </tr>
</table>

## 为什么需要 Otter

主流代码评测集采用快照式评估——给一次输入，出一次结果。但真实编程中，开发者会根据编译错误、测试失败等反馈反复修改代码。**这个反馈驱动的迭代过程才是编程能力的核心体现。**

Otter 将执行反馈集成进评测流程，让智能体像真实开发者一样：写代码 → 运行 → 看报错 → 修改 → 再运行，直到通过或达到最大轮次。

```
   ┌────────────────────────────┐
   ↓                            │
Proposer ───→ Executor ───→ Evaluator
                                │
          达到最大轮次/完全解决问题 │
                                ↓
                               结束
```

## 快速开始

**前置要求**：Python >= 3.11，Docker

```bash
# 安装
pip install -e .

# 配置
cp .env.example .env
# 编辑 .env，填入你的 API 信息

# 运行评测
otter run
```

## 配置

所有参数通过 `.env` 文件管理。CLI 仅接受 `--env` 参数选择配置文件：

```bash
otter run              # 默认使用 .env
otter run --env .env.local  # 指定配置文件
```

### 必填项

```ini
EXECUTOR__api_key=sk-xxx
EXECUTOR__base_url=https://api.openai.com/v1
EXECUTOR__model=gpt-4o
```

### 可选项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EXECUTOR__executor_type` | `chat_llm` | 执行器类型 |
| `EXECUTOR__concurrency` | `1` | 执行器并发请求数 |
| `EXECUTOR__max_retries` | `3` | API 调用失败重试次数 |
| `EXECUTOR__retry_base_delay` | `1.0` | 重试退避基础延迟（秒） |
| `EXPERIMENT__experiment_id` | `default` | 实验 ID，结果存入 `experiments/{id}/` |
| `EXPERIMENT__max_turns` | `1` | 最大反馈迭代轮次 |
| `EXPERIMENT__samples_per_problem` | `1` | 每道题独立采样次数 |
| `EXPERIMENT__feedback_strategy` | `error_message` | 反馈策略（`minimal` / `error_message` / `progressive`） |
| `DATASET__dataset_name` | `mbppplus` | 数据集名称 |
| `DATASET__cache_dir` | `data/cache` | 数据集缓存目录 |
| `DOCKER__cpus` | `1.0` | 容器 CPU 限制 |
| `DOCKER__memory` | `512m` | 容器内存限制 |
| `DOCKER__network_mode` | `none` | 容器网络模式 |
| `DOCKER__timeout` | `10` | 单条命令执行超时（秒） |
| `LOG__level` | `INFO` | 日志级别 |
| `LOG__log_file` | 无 | 日志文件路径 |

## 输出结构

每次运行的结果以目录结构保存在 `experiments/` 下，每道题的每轮尝试都有完整记录：

```
experiments/{experiment_id}/
└── {task_id}#{sample_id}/
    ├── turn_1/
    │   ├── exec_input/    # 发给执行器的输入
    │   ├── exec_output/   # 执行器产生的输出
    │   ├── eval_input/    # 发给评估器的输入
    │   ├── eval_output/   # 评估器产生的输出
    │   └── meta.json     # 本轮判定结果 {"passed": true/false}
    ├── turn_2/           # 第二轮（如果第一轮未通过且 max_turns > 1）
    │   └── ...
    └── ...
```

## 支持的数据集

| 数据集 | 状态 | 说明 |
|---|---|---|
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) | 完整支持 | 函数级 Python 编程题 |
| [LiveCodeBench](https://livecodebench.github.io/) | 计划中 | 无污染的实时编程题 |
| [EvalPlus](https://github.com/evalplus/evalplus) | 计划中 | 严格的 LLM4Code 评测集 |
| [SWE-Bench](https://www.swebench.com/) | 计划中 | 真实 GitHub Issue 修复 |
| [Tau2Bench](https://github.com/sierra-research/tau2-bench) | 计划中 | 多轮智能体任务评测 |
| [TerminalBench](https://terminalbench.com/) | 计划中 | 终端环境编程任务 |
| [SWE-CI](https://github.com/SWE-CI/SWE-CI) | 计划中 | CI 驱动的软件工程任务 |

## 许可证

[Apache License 2.0](LICENSE)