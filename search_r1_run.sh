#!/bin/bash
# search_r1_run.sh
# - 按模型并行，内部（数据集×噪声）串行
# - 读取 config/models.yaml 中的每个模型配置；evaluate 会据此实例化 agent

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "CURRENT_DIR = $CURRENT_DIR"

cd "$CURRENT_DIR"

RESULTS_DIR="${CURRENT_DIR}/Search-R1/eval_results"
echo "RESULTS_DIR = $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

LOG_DIR="${CURRENT_DIR}/Search-R1/eval_logs"
echo "LOG_DIR = $LOG_DIR"
mkdir -p "$LOG_DIR"

EVAL_SCRIPT="${CURRENT_DIR}/Search-R1/evaluate/evaluate_hotpot_2wiki.py"
echo "EVAL_SCRIPT = $EVAL_SCRIPT"

MODELS_YAML="${CURRENT_DIR}/Search-R1/models.yaml"
echo "MODELS_YAML = $MODELS_YAML"

# # ===== 两组模型（键名需与 models.yaml 对应） =====
non_think_model=('deepseek-v3-0324' 'qwen3-32b-without-thinking' 'gpt-5-mini' 'gemini-2.5-flash-without-thinking' 'doubao-seed-1.6' 'gpt-4.1' 'qwen3-235b-a22b-instruct-2507' 'deepseek-v3.1-without-thinking' 'deepseek-v3.2-exp-without-thinking' 'qwen3-max' 'glm-4.5-without-thinking' 'longcat-flash-chat' 'claude-4-sonnet-without-thinking' 'claude-4.1-opus-without-thinking' 'gpt-3.5-turbo')

think_model=('qwen3-32b-with-thinking' 'gemini-2.5-flash-with-thinking' 'deepseek-r1-0528' 'gemini-2.5-pro' 'doubao-seed-1.6-Thinking' 'qwen3-235b-a22b-thinking-2507' 'o4-mini' 'glm-4.5-with-thinking' 'gpt-5' 'claude-4-sonnet-with-thinking' 'longcat-flash-thinking' 'grok-4')

# # ===== 两个数据集 & 六类噪声 =====
DATASETS=('hotpotqa' '2wikimultihopqa')
# echo "DATASETS = ${DATASETS[*]}"

HOTPOT_PATH="${CURRENT_DIR}/Search-R1/data/nq_hotpotqa_train/test_hotpotqa.json"
TWIKI_PATH="${CURRENT_DIR}/Search-R1/data/nq_hotpotqa_train/test_2wikimultihopqa.json"
echo "HOTPOT_PATH = $HOTPOT_PATH"
echo "TWIKI_PATH = $TWIKI_PATH"

NOISE_TYPES=('clean' 'failure' 'error' 'induce' 'incomplete' 'redundancy')

# # ===== Agent / 噪声缺省参数（可按需改） =====
MAX_TURNS=10
MAX_OUTPUT_TOKENS=""       # 留空→使用 models.yaml 的 max_completion_tokens
TEMPERATURE=""             # 留空→使用 models.yaml 的 temperature
NOISE_MODEL="gpt-4o-mini"
LANGUAGE="en"
APPLY_ALWAYS_FLAG="--apply-always"  # 若要每步都加噪，启用此标志；否则按概率加噪

# 当APPLY_ALWAYS_FLAG为空时，下面几个才生效
START_STEP=1             # 从第几步开始尝试加噪  
PROB_PER_STEP=0.7        # 每步尝试加噪的概率
INTERVAL_STEPS=1         # 每隔多少步尝试加噪，比如第1步加噪，则第3步会尝试加噪（如果PROB_PER_STEP不等于1的话）
MAX_TIMES_PER_TRAIL=4    # -1表示不限次数

MAX_CONCURRENT_MODELS=8   # 模型级并行度，最多同时跑多少个 model
VERBOSE="--verbose"                 # 是否输出详细日志

dataset_path() {
    case "$1" in
        "hotpotqa") echo "$HOTPOT_PATH" ;;
        "2wikimultihopqa") echo "$TWIKI_PATH" ;;
        *) echo "Unknown dataset: $1" && exit 1 ;;
    esac
}

# 返回日志目录（与 eval_results 一致的层次:max_examples / dataset / noise）
generate_log_dir() {
    local dataset="$1"; local noise="$2"; local max_examples="$3"
    local a=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local b=$(echo "$noise"   | tr '[:upper:]' '[:lower:]')
    echo "${LOG_DIR}/${max_examples}/${a}/${b}"
}

# 生成日志文件名
generate_log_file() {
    local model="$1"; local dataset="$2"; local noise="$3"; local max_examples="$4"
    local dir=$(generate_log_dir "$dataset" "$noise" "$max_examples")
    local m=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr '/().' '_' | tr '-' '_')
    echo "${dir}/${m}.log"
}


# 测试generate_log_file函数
# echo "$(generate_log_file "gpt-4.1" "hotpotqa" "failure" 10)"

# output_dir
generate_output_dir() {
    local dataset="$1"; local noise="$2"; local max_examples="$3"
    local a=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local b=$(echo "$noise" | tr '[:upper:]' '[:lower:]')
    echo "${RESULTS_DIR}/${max_examples}/${a}/${b}"
}

# 合并两组模型（去重）
get_all_models() {
  local seen=()
  for m in "${non_think_model[@]}" "${think_model[@]}"; do
    local key=$(echo "$m" | tr -d '\n')
    if [[ " ${seen[*]} " != *" $key "* ]]; then
      echo "$key"
      seen+=("$key")
    fi
  done
}

# # 测试get_all_models函数
# # mapfile -t models < <(get_all_models)
# # echo "All models:"
# # for m in "${models[@]}"; do
# #   echo "$m"
# # done 


main() {
  echo "Results dir: $RESULTS_DIR"

  # 用 FIFO（pipe）作为 token 桶，最多允许 MAX_CONCURRENT_MODELS 个并行模型
  local fifo_file="/tmp/$$.fifo"; mkfifo "$fifo_file"; exec 3<>"$fifo_file"; rm -f "$fifo_file"
  for ((i=0; i<MAX_CONCURRENT_MODELS; i++)); do echo >&3; done

  max_examples=1  # 可根据需要调整

  mapfile -t ALL_MODELS < <(get_all_models)

  for MODEL_KEY in "${ALL_MODELS[@]}"; do
    read -u 3
    {
      echo "[MODEL $MODEL_KEY] start: $(date)"
      for ds in "${DATASETS[@]}"; do
        local_path=$(dataset_path "$ds")
        for nz in "${NOISE_TYPES[@]}"; do
          out_dir=$(generate_output_dir "$ds" "$nz" "$max_examples")
          log_file=$(generate_log_file "$MODEL_KEY" "$ds" "$nz" "$max_examples")
          mkdir -p "$out_dir"
          mkdir -p "$(dirname "$log_file")"     

          cmd="python \"$EVAL_SCRIPT\" \
                  --dataset $ds \
                  --dataset-path \"$local_path\" \
                  --model \"$MODEL_KEY\" \
                  --models-yaml \"$MODELS_YAML\" \
                  --max-turns $MAX_TURNS \
                  --max-examples $max_examples \
                  --output-dir \"$out_dir\" \
                  --noise-type $nz \
                  --noise-model \"$NOISE_MODEL\" \
                  --language $LANGUAGE \
                  --start-step $START_STEP \
                  --prob-per-step $PROB_PER_STEP \
                  --interval-steps $INTERVAL_STEPS \
                  --max-times-per-trail $MAX_TIMES_PER_TRAIL"

          # 若你想临时覆盖 YAML 的 max_tokens / temperature，可在此追加
          # [[ -n "$MAX_OUTPUT_TOKENS" ]] && cmd="$cmd --max-output-tokens $MAX_OUTPUT_TOKENS"
          # [[ -n "$TEMPERATURE" ]] && cmd="$cmd --temperature $TEMPERATURE"
          [[ -n "$APPLY_ALWAYS_FLAG" ]] && cmd="$cmd $APPLY_ALWAYS_FLAG"  # 启用每步加噪
          [[ -n "$VERBOSE" ]] && cmd="$cmd $VERBOSE"                      # 启用详细日志

          echo "CMD: $cmd" > "$log_file"
          stdbuf -o0 -e0 bash -lc "$cmd" >> "$log_file" 2>&1 # 写日志

          sleep 3  # 轻微节流，防限流
        done
      done
      echo "[MODEL $MODEL_KEY] done: $(date)"
      echo >&3
    } &
  done

  wait
  exec 3>&-
  echo "== All done =="
}
main "$@"
