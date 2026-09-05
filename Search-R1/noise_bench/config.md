顶层统一挂在 `noise.*`，并分成**通用**与**类型专属**两层。

### 1) 通用（所有噪声共享）

- `noise.enable`：bool，是否启用加噪（默认 `true`）。
- `noise.type`：str，噪声类型（当前支持：`incomplete`；后续可扩：`error|redundancy|induce|failure`）。
- `noise.apply_always`：bool，是否**每个 step 都加噪**（默认 `true`）。如果为 `false`，配合 `prob_per_step` 使用。
- `noise.prob_per_step`：float，0~1，step 级触发概率（默认 `1.0`；当 `apply_always=false` 时生效）。
- `noise.start_step`：int，从第几次 **step/episode** 开始允许加噪（默认 `1`）。
- `noise.interval_steps`：int，加噪间隔的最小 step 数（默认 `1`；即每个 step 都可加噪）。
- `noise.max_times_per_trail`：int，该 **trail** 内最多加噪次数（默认 `9999`；你现在 val_batch=1，最小版本可不限制）。
- `noise.seed`：int，LLM 采样或随机选择的种子（默认 `0`）。
- `noise.temperature`：float，LLM 采样温度（默认 `0.7`）。
- `noise.api.model`：str，noiser 使用的 LLM 模型名（例如 `gpt-4o-mini` / 内网模型名等）。
- `noise.api.base_url`：str，noiser 的 API base（可为空）。
- `noise.api.key_env`：str，从哪个环境变量读取 API Key（默认 `OPENAI_API_KEY`；**不要**把 key 明文写脚本）。

> 说明：这层参数能支撑**调度**、**触发**、**可复现**、**安全兜底**；即使只做 `incomplete`，也建议把骨架搭好。

### 2) 类型专属：`noise.incomplete.*`

结合你给的中文 prompt（`zh_add_incomplete`），提供**模式选择+掩蔽细节**：

- `noise.incomplete.language`：`zh|en|`（默认 `zh`，便于 LLM 采样风格；将来英文数据可切 `en`）。

### 3) 类型专属：`noise.failure.*`

- `apply_always`: 表示从“第一步”开始“每一步”都“100%”加噪。若想自定义加噪的概率和频率，可以设置为 `false`，并调整下面的参数。
- `start_step`: 从第几步开始允许加噪（默认 `1`，即从第一步开始）。
- `prob_per_step`: 每一步加噪的概率（默认 `0.7`，表示每次请求有70%的概率成功，30%的概率失败）。
- `interval_steps`: 两次加噪之间的最小间隔步数（默认 `1`，表示每隔一步都可以加噪）。
- `max_times_per_trail`: 单条样本最多加噪次数（默认 `-1`，表示不限次数）。

我们测试的情况：

```
+    +noise.failure.apply_always=false \
+    +noise.failure.start_step=1 \
+    +noise.failure.prob_per_step=0.7 \ 
+    +noise.failure.interval_steps=1 \
+    +noise.failure.max_times_per_trail=-1 
```

即： 模拟现实中**网络不佳**的情况，比如每一次请求有70%的概率失败，30%的概率成功。
