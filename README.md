<div align="center"><h1>
    Agent Noise Benchmark V2<br>
    with Versatile Interactive Tasks
</h1></div>

## Quick Start

### Installation

1. Create two virtual environments (virtual environments 1):

```bash
git clone https://github.com/littlelittlenine/Agent_Noise.git
cd vitabench
pip install -e .
```

2. (virtual environments 2)

```bash
cd tau2-bench-main
pip install -e .
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

   This will enable you to run the `vita` and `tau2` command.

3. Dataset Preparation

   Due to permission restrictions, the datasets need to be downloaded separately:

   1. **TAU2-Bench**: Download from [tau2-bench](https://github.com/sierra-research/tau2-bench)
   2. **VitaBench**: Download from [vitabench](https://github.com/meituan-longcat/vitabench)

   After downloading, extract the domains files and place them in:

   - `data/tau2/` for TAU2-Bench data
   - `data/vita/` for VitaBench data

4. Refine information in `$REPO/tau2-bench-main/src/tau2/evaluator/.env` and

```bash
# $REPO is this checkout; every path below is relative to it.
REPO="$(pwd)"
export NOISE_ENV_PATH="$REPO/tau2-bench-main/src/tau2/evaluator/.env"
bash $REPO/tau2_run.sh
bash $REPO/vita_run.sh
```

5. Refine information in `$REPO/tau2-bench-main/src/tau2/models.yaml` and `$REPO/vitabench/src/vita/models.yaml`

   You just need to comment out all the models above except for gpt-4.1 and claude-3.7, and uncomment the models below.

   The model number does not need to be changed.

### Setup LLM Configurations

If you want to customize the location of the `models.yaml` file, you can specify the environment variable `VITA_MODEL_CONFIG_PATH` (default path from repository root is `src/vita/models.yaml`). For example:

```bash
export VITA_MODEL_CONFIG_PATH=/path/to/your/model/configuration
```

Example `models.yaml` file

```yaml
default:
  base_url: <base url>
  temperature: <temperature>
  max_input_tokens: <max input tokens>
  headers:
    Accept: "*/*"
    Accept-Encoding: "gzip, deflate, br"
    Content-Type: "application/json"
    Authorization: "Bearer <api key>"
    Connection: "keep-alive"
    Cookie: <cookie>
    User-Agent: <user agent>

models:
  - name: <model name>
    max_tokens: <max completion tokens (for some models, use max_completion_tokens)>
    max_input_tokens: <max input tokens>
    reasoning_effort: "high"
    thinking:
      type: "enabled"
      budget_tokens: <budget tokens>
    cost_1m_token_dollar:
      prompt_price: <dollars per 1 million tokens>
      completion_price: <dollars per 1 million tokens>
```

The default configuration can apply to all models, the custom model configuration can overwrite default values.

### Run evaluations

To run a vita test evaluation:

```bash
vita run \
  --domain <domain> \              # support single domain (delivery/instore/ota) and cross domain ([delivery,instore,ota])
  --user-llm <model name> \        # model name in models.yaml
  --agent-llm <model name> \       # model name in models.yaml
  --enable-think \                 # Enable think mode for the agent. Default is False.
  --evaluator-llm <model name> \   # The LLM to use for evaluation.
  --num-trials 1 \                 # (Optional) The number of times each task is run. Default is 1.
  --num-tasks 1 \                  # (Optional) The number of tasks to run. Default is the number of all tasks.
  --task-ids 1 \                   # (Optional) Run only the tasks with the given IDs. Default is run all tasks.
  --max-steps 300 \                # (Optional) The maximum number of steps to run the simulation. Default is 300.
  --max-concurrency 1 \            # (Optional) The maximum number of concurrent simulations to run. Default is 1.
  --csv-output <csv path> \        # (Optional) Path to CSV file to append results.
  --language <chinese/english> \   # (Optional) The language to use for prompts and tasks. Choices: chinese, english. Default is chinese.
  --noise_category others \        # (Optional) Selected in ['others','wrong','fault','induce','redundant','incomplete']. Default is 'others' (injects nothing); 'fault' simulates a failed tool call.
  --priority_level 1 \             # (Optional) It can be fixed at 1.
  --noise_nums 1000000 \           # (Optional) Total noise, set to 100,000.
  --max_single_tool_noise 1 \      # (Optional) The noise ceiling of each tool.
  --user_noise original \          # (Optional) Selected in ["original", "ambiguity","topic_shift","conflict","redundancy","boundary"]. Default is "original".
```

Results will be saved in `data/simulations/`.

To run a tau2 test evaluation:

```bash
tau2 run \
  --domain <domain> \              # airline / retail / telecom / telecom-workflow / mock
  --user-llm <model name> \        # model name in models.yaml
  --agent-llm <model name> \       # model name in models.yaml
  --enable-think \                 # Enable think mode for the agent. Default is False.
  --num-trials 1 \                 # (Optional) The number of times each task is run. Default is 1.
  --num-tasks 1 \                  # (Optional) The number of tasks to run. Default is the number of all tasks.
  --max-concurrency 1 \            # (Optional) The maximum number of concurrent simulations to run. Default is 1.
  --noise_category others \        # (Optional) Selected in ['others','wrong','fault','induce','redundant','incomplete']. Default is 'others' (injects nothing); 'fault' simulates a failed tool call.
  --noise_nums 1000000 \           # (Optional) Total noise, set to 100,000.
  --max_single_tool_noise 1 \      # (Optional) The noise ceiling of each tool.
  --user_noise original \          # (Optional) Selected in ["original", "ambiguity","topic_shift","conflict","redundancy","boundary","cognition"]. Default is "original".
```

Once you have the datasets prepared, you can quickly run evaluations:

- Add your model to `model.yaml`.
- Run the evaluation scripts: `./vita_run.sh` for VitaBench or `./tau2_run.sh` for TAU2-Bench.

### Reading the results

For the `induce` and `redundant` categories, every injection is audited individually and the
stability-gated score is written into `reward_info` of each simulation:

| Field | Meaning |
| --- | --- |
| `step_deviations` | One entry per injection: `turn_idx`, `deviated` and the judge's `justification`. |
| `traj_ok` | Conjunction of the per-step verdicts. `false` if any step deviated, `null` if the judge could not be reached. |
| `task_success` | Task completion on its own, ungated. |
| `sga` | `task_success * traj_ok` — the stability-gated success. `null` when `traj_ok` is `null`. |

The other categories are not audited, so `traj_ok` stays `true` and `sga` equals `task_success`.

Two things to keep in mind:

- The judge reads its credentials from the `.env` pointed to by `NOISE_ENV_PATH`. Without it the
  run still finishes and reports `reward`, but `traj_ok` and `sga` will be `null`.
- A simulation that stops at `--max-steps` is scored as a premature termination and skips
  evaluation entirely, so keep the default (`300` for VitaBench, `200` for TAU2-Bench) unless you
  only want the trajectories.

## Test Search-R1

1. Open a terminal and enter the Search-R1 directory.

```bash
cd Search-R1
```

2. Create the conda environment `searchr1`

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

3. Open a new terminal and enter the Search-R1 directory.

```bash
cd Search-R1
```

4. Create the conda environment `retriever`

```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

5. (In `retriever`) Download the indexing and corpus.

```bash
save_path=./corpus
mkdir -p $save_path
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

6. (In `searchr1`) Prepare data.

```bash
bash ./data_process/prepare_test_data.sh
```

7. (In `retriever`) Launch a local retrieval server.

```bash
bash retrieval_launch.sh
```

8. Modify `models.yaml` to set your LLM configurations.

```yaml
# models.yaml
defaults:
  max_completion_tokens: 512
  temperature: 0.7
  enable_thinking: null  # true | false | null. If you need it, just overwrite it yourself in the models.
  reasoning_effort: null # "low" | "medium" | "high" | null. If you need it, just overwrite it yourself in the models.

models:
  # -------- non-think ----------
  deepseek-v3-0324:  # Case 1: The model itself does not support thinking, so do not write enable_thinking.
    api_model: deepseek-v3-0324
  qwen3-32b-without-thinking: # Case 2：The model itself supports thinking, but here we are testing its performance under the condition of not thinking, so we write enable_thinking: false.
    api_model: qwen3-32b
    enable_thinking: false
  # ...

  # -------- think ----------
  qwen3-32b-with-thinking:  # Reasoning models：enable_thinking: true.
    api_model: qwen3-32b
    enable_thinking: true
  gpt-5:
    api_model: gpt-5
    reasoning_effort: high  # For the GPT family, set reasoning effort instead of enable thinking.
  # ...
```

9. (In `searchr1`) Run the evaluation.

```bash
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""
```

```bash
cd ../
bash search_r1_run.sh
```

10. (Optional) Modify `search_r1_run.sh`.
- Set the number of test examples.

```bash
main() {
  echo "Results dir: $RESULTS_DIR"

  local fifo_file="/tmp/$$.fifo"; mkfifo "$fifo_file"; exec 3<>"$fifo_file"; rm -f "$fifo_file"
  for ((i=0; i<MAX_CONCURRENT_MODELS; i++)); do echo >&3; done

  max_examples=1  # <- adjust as needed
```

- Noise related config.

```bash
# ===== Agent / Noise Default Parameters (adjust as needed) =====
MAX_TURNS=10
MAX_OUTPUT_TOKENS=""       # Leave empty → use max_completion_tokens from models.yaml
TEMPERATURE=""             # Leave empty → use temperature from models.yaml
NOISE_MODEL="gpt-4o-mini"
LANGUAGE="en"
APPLY_ALWAYS_FLAG="--apply-always"  # Enable this flag to apply noise at every step; otherwise, noise is applied probabilistically
```

- Override some config.

```bash
# If you want to temporarily override the YAML max_tokens / temperature, you can add them here
# [[ -n "$MAX_OUTPUT_TOKENS" ]] && cmd="$cmd --max-output-tokens $MAX_OUTPUT_TOKENS"  # If you want to modify the maximum output length per model round, you should first change MAX_OUTPUT_TOKENS in the "Noise related config." above and uncomment this line.
# [[ -n "$TEMPERATURE" ]] && cmd="$cmd --temperature $TEMPERATURE" # If you want to modify the model's output temperature, you should first change the TEMPERATURE in the 'Noise related config.' above and uncomment this line.
[[ -n "$APPLY_ALWAYS_FLAG" ]] && cmd="$cmd $APPLY_ALWAYS_FLAG"  # Uncomment this line to enable noise addition for every step.
[[ -n "$VERBOSE" ]] && cmd="$cmd $VERBOSE"                      # Uncomment this line to enable detailed logging.
```

11. (Optional) If the port is occupied, you can modify the port number in `/Search-R1/search_r1/search/retrieval_server.py`.

```python
uvicorn.run(app, host="0.0.0.0", port=8008)
```

12. (Optional) Test if the retrieval service is started successfully

```bash
curl -s -X POST http://127.0.0.1:8008/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["Who wrote Sapiens?"],"topk":3,"return_scores":true}'
```

13. View running results.
- Evaluation results are saved in `./eval_results/`.
- Log files are saved in `./eval_logs/`.

### 🤗 Acknowledgement
We built our evaluation framework by adapting parts of the codebases from [tau2-bench](https://github.com/sierra-research/tau2-bench), [vitabench](https://github.com/meituan-longcat/vitabench), and [Search-R1](https://github.com/petergriffinjin/search-r1), and we sincerely appreciate their contributions to the Agent community.
