# Graph-as-Code: Arxiv 节点分类复现

复现论文《Actions Speak Louder Than Prompts》中 **Graph-as-Code (φ_code)** 交互模式在 **OGB Arxiv 数据集** 上的节点分类 pipeline。

---

## 项目概述

传统图神经网络（GNN）需要训练模型参数，而 Graph-as-Code 的核心思想是：**让大语言模型（LLM）通过生成并执行代码来直接与图数据交互，从而完成节点分类任务**。

本项目实现了一个完整的 end-to-end pipeline：
- 输入：目标节点 ID
- 过程：LLM 迭代生成 pandas 查询代码 → 沙箱执行 → 获取结果 → 继续推理
- 输出：预测类别标签

论文基准：74.40 ± 3.02%（Arxiv 测试集）

---

## 核心机制：Graph-as-Code

采用 **ReAct 风格**的多轮交互循环：

```
1. 向 LLM 提供任务描述 + DataFrame 结构 + 40个类别定义
2. LLM 进行 reasoning，生成一条 pandas/Python 查询代码
3. 在受限沙箱中执行代码，返回执行结果
4. 将结果追加到对话历史
5. 循环直到 LLM 输出 "Answer: [class_id]"
```

典型推理路径：
1. 查看目标节点的文本特征（title + abstract）
2. 查看邻居节点的标签分布（多数投票）
3. 查看邻居的文本特征验证假设
4. 综合判断，输出最终类别

---

## 数据集

**OGB Arxiv (ogbn-arxiv)**
- 169,343 个节点（学术论文）
- 1,166,243 条边（引用关系）
- 40 个类别（CS 子领域）
- 分割：Train / Valid / Test = 53.7% / 17.6% / 28.7%
- 平均度：13.64

数据来源：
- `Datas/Arxiv.csv`：节点文本特征（title, abstract）+ label_id + category
- OGB 下载：图结构（边列表）

---

## 项目结构

```
graph-inference/
├── run.py                    # 主入口：加载数据 → 采样测试节点 → 批量评估 → 输出准确率
├── test_pipeline.py          # 集成测试：Mock LLM 测试完整 pipeline（无需真实 API）
├── patch_ogb.py              # 辅助脚本：验证 OGB 数据加载
├── requirements.txt          # Python 依赖
├── .env                      # API Key 配置（不提交到仓库）
│
├── config/
│   └── config.yaml           # 模型参数、数据集路径、日志配置
│
├── src/
│   ├── data_loader.py        # 加载 CSV + OGB 边数据，构建 DataFrame，缓存为 pkl
│   ├── llm_client.py         # 统一 LLM 客户端（DeepSeek / Kimi / OpenAI 兼容）
│   ├── prompt_template.py    # Graph-as-Code 系统提示模板（含 Schema + 40类定义）
│   ├── graph_as_code.py      # ReAct 交互循环：LLM 推理 → 提取代码 → 执行 → 终止判断
│   ├── code_executor.py      # 受限 Python 沙箱：AST 白名单 + 超时控制，仅暴露 df/pd
│   └── evaluator.py          # 批量评估：计算准确率、生成 JSONL 日志
│
├── Datas/
│   ├── Arxiv.csv             # 节点特征（原始数据）
│   ├── arxiv_processed.pkl   # 预处理后的 DataFrame（缓存）
│   └── Feature/              # RoBERTa 编码特征（本次未使用）
│
├── logs/                     # 运行日志（JSONL 格式：node_id, true_label, predicted_label, correct, elapsed_sec）
└── results/                  # 评估结果输出
```

---

## 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：pandas, numpy, requests, openai, python-dotenv, pyyaml, ogb, torch

### 2. 配置 API Key

编辑 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
KIMI_API_KEY=your_kimi_key_here
```

### 3. 配置模型

编辑 `config/config.yaml`：

```yaml
llm:
  provider: "deepseek"          # 或 "kimi"
  model: "deepseek-chat"        # 或 "kimi-latest"
  base_url: "https://api.deepseek.com/v1"
  api_key_env: "DEEPSEEK_API_KEY"
  temperature: 0.1
  max_tokens: 4096
```

---

## 使用方式

### 批量运行评估

```bash
python run.py
```

可选参数：
```bash
python run.py --num-samples 50   # 评估 50 个测试节点（覆盖配置）
python run.py --seed 123         # 指定随机种子
python run.py --verbose          # 打印每步的 LLM reasoning 和代码执行结果
```

输出示例：
```
[Run] DataFrame loaded: 169343 nodes
[Run] Evaluating on 100 test nodes (seed=42)
[1/100] node=126870 true=31 pred=10 ✗ (55.7s) acc=0/1=0.00%
...
==================================================
Accuracy: 72.00% (72/100)
Log file: logs/run_20260420_164227.jsonl
==================================================
```

### 运行集成测试（无需 API）

```bash
python test_pipeline.py
```

使用 Mock LLM 模拟完整交互循环，验证 pipeline 逻辑正确性。

### 验证 OGB 数据

```bash
python patch_ogb.py
```

---

## 核心模块详解

### `data_loader.py`

- `build_arxiv_dataframe()`: 从 CSV 读取节点文本，从 OGB 下载边数据，构建无向邻居列表，生成 DataFrame（columns: `node_id`, `features`, `neighbors`, `label`），缓存为 `.pkl`
- `load_category_names()`: 从 CSV 提取 label_id → category 名称映射（40 类）
- `patch_torch_load()`: 兼容新版 PyTorch（解决 `weights_only` 问题）

### `llm_client.py`

- 统一封装 OpenAI 兼容 API（DeepSeek、Kimi）
- 支持自动重试（指数退避）
- 配置驱动：provider, model, temperature, max_tokens, retry, backoff

### `graph_as_code.py`

- `GraphAsCodeAgent.classify(node_id)`: 核心 ReAct 循环
  - `MAX_STEPS = 15`：单节点最大交互步数
  - `ANSWER_PATTERN`: 正则匹配 `Answer: [class_id]` 终止条件
  - `_extract_code()`: 从 LLM 输出中提取 pandas 表达式（自底向上扫描，优先 `df.`/`pd.` 开头）
  - 代码执行异常或格式错误时，将错误信息反馈给 LLM 让其修正

### `code_executor.py`

- `SecureExecutor`: 受限代码沙箱
  - AST 白名单：仅允许安全节点类型（Expression, BinOp, Call, Subscript, ListComp 等）
  - 名称白名单：仅暴露 `df`, `pd`, `len`, `sum`, `sorted` 等基础函数
  - 禁止：`eval`, `exec`, `compile`, `__import__`
  - 超时：单步执行不超过 5 秒
  - 结果格式化：DataFrame > 100 行时只返回前 5 行，避免 Token 爆炸

### `prompt_template.py`

- 生成系统提示，包含：
  - Task 描述
  - DataFrame Schema（index, features, neighbors, label）
  - 40 个类别定义
  - Response 格式约束（reasoning + 单行 pandas 代码，或 `Answer: [class_id]`）

### `evaluator.py`

- 遍历测试节点，调用 `GraphAsCodeAgent.classify()`
- 计算准确率，生成 JSONL 日志
- 实时输出进度：`[idx/total] node=X true=Y pred=Z ✓/✗ (elapsed)s acc=...`

---

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.10+ |
| 数据处理 | pandas, numpy |
| 图数据 | OGB (Open Graph Benchmark), torch |
| LLM API | OpenAI 兼容 SDK |
| 配置管理 | PyYAML + python-dotenv |
| 日志 | 结构化 JSONL |

---

## 注意事项

1. **Token 成本**：Graph-as-Code 每节点需多轮 API 调用（平均度 13.64，每节点通常 3-5 步）。建议先在小样本（10-20 节点）验证，再批量运行。
2. **API Key 安全**：使用 `.env` 存储，已加入 `.gitignore`。
3. **代码执行安全**：`SecureExecutor` 通过 AST 白名单严格限制，但请避免在生产环境暴露给不可信输入。
4. **运行时间**：单节点约 10-60 秒（取决于 API 响应速度和交互步数），100 节点约 15-30 分钟。
5. **OGB 数据首次加载**：会自动下载 `ogbn-arxiv` 数据集到 `dataset/` 目录。
