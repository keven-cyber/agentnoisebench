#!/bin/bash
# VITA实验专用脚本（串行执行版本）
# 要求：我需要修正一下这部分内容，首先non_think_model完全可以用这里面的配置
# 但是think_model需要改变model_yaml里面的配置，给gpt5 里面的reasoning_effort: "minimal"改成reasoning_effort: "high" ,然后给Gemini-2.5-Flash，Qwen3-32B，Claude-4-Sonnet，GLM-4.5，Claude-4.1-Opus对应的model的加上
# thinking: 
#   type: "enabled"
model_yaml='Agent_Noise_Bench/vitabench/src/vita/models.yaml'

echo "=== VITA实验开始 ==="
echo "请确保已激活环境: conda activate noise_vita"
echo "请确保当前目录: cd vitabench"
echo "=========================================="
cd vitabench
# 当前脚本所在目录
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 创建results目录
RESULTS_DIR="${CURRENT_DIR}/results_end"
CSV_DIR="${CURRENT_DIR}/results"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CSV_DIR"
# 模型列表定义
non_think_model=('deepseek-v3-0324', 'qwen3-32b', 'gpt-5', 'gpt-4.1', 'gemini-2.5-flash', 'doubao-seed-1.6', 'qwen3-235b-a22b-instruct-2507', 'kimi-k2-0905', 'deepseek-v3.1', 'deepseek-v3.2-exp', 'qwen3-max', 'glm-4.5', 'longcat-flash-chat', 'claude-4-sonnet', 'claude-4.1-opus')
think_model=('deepseek-r1-0528', 'gemini-2.5-pro','doubao-seed-1.6', 'qwen3-235b-a22b-thinking-2507', 'o4-mini', 'o3', 'longcat-flash-thinking', 'grok-4')
# non_think_model=()
# think_model=('qwen3-32b','gemini-2.5-flash','glm-4.5', 'claude-4-sonnet','claude-4.1-opus', 'gpt-5')
vita_domains=('delivery', 'instore', 'ota')
noise_categories=('others', 'fault', 'incomplete', 'induce', 'redundant', 'wrong')
user_noise="original"

# 要排除的模型列表
# EXCLUDE_MODELS=("gpt-4.1" "claude-3-7-sonnet-20250219")

# 检查模型是否在排除列表中
should_exclude() {
    local model="$1"
    for excluded in "${EXCLUDE_MODELS[@]}"; do
        if [[ "$model" == "$excluded" ]]; then
            return 0
        fi
    done
    return 1
}

# 检查模型是否需要启用think
need_enable_think() {
    local model="$1"
    for think_m in "${think_model[@]}"; do
        if [[ "$model" == "$think_m" ]]; then
            return 0
        fi
    done
    return 1
}

# 生成vita日志文件名
generate_vita_log_filename() {
    local model="$1"
    local domain="$2"
    local noise_category="$3"
    local user_noise="$4"
    local max_single_tool_noise="$5"
    
    local log_model_name=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.' | tr '-' '_')
    local log_domain=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    local log_noise_category=$(echo "$noise_category" | tr '[:upper:]' '[:lower:]')
    local log_user_noise=$(echo "$user_noise" | tr '[:upper:]' '[:lower:]')
    
    echo "${RESULTS_DIR}/vita_agent_llm_${log_model_name}_domain_${log_domain}_noise_category_${log_noise_category}_user_noise_${log_user_noise}_max_single_tool_noise_${max_single_tool_noise}.log"
}

# 生成vita CSV输出路径
generate_vita_csv_output() {
    local model="$1"
    local domain="$2"
    local noise_category="$3"
    local user_noise="$4"
    local max_single_tool_noise="$5"
    
    local csv_model_name=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.' | tr '-' '_')
    local csv_domain=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    local csv_noise_category=$(echo "$noise_category" | tr '[:upper:]' '[:lower:]')
    local csv_user_noise=$(echo "$user_noise" | tr '[:upper:]' '[:lower:]')
    
    echo "$CSV_DIR/agent_llm_${csv_model_name}_domain_${csv_domain}_noise_category_${csv_noise_category}_user_noise_${csv_user_noise}_max_single_tool_noise_${max_single_tool_noise}"
}

# 获取最大单工具噪声数量
get_max_single_tool_noise() {
    local noise_category="$1"
    case "$noise_category" in
        "others"|"fault")
            echo "1"
            ;;
        "wrong")
            echo "2"
            ;;
        "induce"|"redundant"|"incomplete")
            echo "10"
            ;;
        *)
            echo "1"
            ;;
    esac
}

# 合并所有模型并去重
get_all_models() {
    local all_models=()
    
    # 添加non_think_model
    for model in "${non_think_model[@]}"; do
        if ! should_exclude "$model"; then
            all_models+=("$model")
        fi
    done
    
    # 添加think_model并去重
    for model in "${think_model[@]}"; do
        if ! should_exclude "$model"; then
            # 检查是否已存在
            found=0
            for existing_model in "${all_models[@]}"; do
                if [[ "$model" == "$existing_model" ]]; then
                    found=1
                    break
                fi
            done
            if [[ $found -eq 0 ]]; then
                all_models+=("$model")
            fi
        fi
    done
    
    # 返回单个模型，而不是数组引用
    for model in "${all_models[@]}"; do
        echo "$model"
    done
}

# 运行vita实验
main() {
    echo "当前工作目录: $(pwd)"
    echo "结果目录: $RESULTS_DIR"

    # 获取所有模型
    mapfile -t all_models < <(get_all_models)
    total_models=${#all_models[@]}
    
    # 检查是否有模型可运行
    if [ "$total_models" -eq 0 ]; then
        echo "错误：没有可运行的模型！请检查 non_think_model 和 think_model 是否为空。"
        exit 1
    fi

    # 1. 设置最大并行模型数量（可根据CPU核心数和API限制调整）
    MAX_CONCURRENT_MODELS=30
    # 创建命名管道（FIFO）用于控制并发
    fifo_file="/tmp/$$.fifo"
    mkfifo "$fifo_file"
    # 将文件描述符3与FIFO关联
    exec 3<>"$fifo_file"
    rm -f "$fifo_file"
    
    # 2. 向管道中预先放入等于最大并发数的"令牌"
    for ((i=0; i<MAX_CONCURRENT_MODELS; i++)); do
        echo >&3
    done

    echo "VITA实验开始：总模型数=$total_models, 最大并行模型数=$MAX_CONCURRENT_MODELS"
    echo "每个模型内部（领域×噪声）保持串行执行"
    echo "=========================================="

    # 3. 为每个模型启动一个后台任务
    for model in "${all_models[@]}"; do
        # 读取一个令牌，若无令牌则阻塞等待，从而实现并发控制
        read -u 3
        {
            echo "[模型 $model] 实验开始于: $(date)"
            
            # 此模型内部：领域和噪声类别仍为串行循环
            for domain in "${vita_domains[@]}"; do
                for noise_category in "${noise_categories[@]}"; do
                    max_single_tool_noise=$(get_max_single_tool_noise "$noise_category")
                    log_file=$(generate_vita_log_filename "$model" "$domain" "$noise_category" "$user_noise" "$max_single_tool_noise")
                    csv_output=$(generate_vita_csv_output "$model" "$domain" "$noise_category" "$user_noise" "$max_single_tool_noise")
                    
                    echo "  [$model] 处理: $domain - $noise_category"
                    
                    # 构建并执行命令
                    base_cmd="vita run --domain $domain --user-llm gpt-4.1 --agent-llm $model --evaluator-llm gpt-4.1 --num-trials 1 --num-tasks 1 --max-steps 300 --max-concurrency 1 --csv-output $csv_output --language chinese --noise_category $noise_category --priority_level 5 --noise_nums 100000 --max_single_tool_noise $max_single_tool_noise --user_noise $user_noise"
                    
                    if need_enable_think "$model"; then
                        cmd="$base_cmd --enable-think"
                    else
                        cmd="$base_cmd"
                    fi
                    echo "  [执行命令] $cmd"
                    # 串行执行单个实验任务，并记录日志
                    stdbuf -o0 -e0 $cmd > "$log_file" 2>&1
                    # 单个任务完成后可短暂间隔（避免API限流）
                    sleep 10
                done
            done
            
            echo "[模型 $model] 所有任务完成于: $(date)"
            # 4. 该模型所有任务完成后，向管道归还一个令牌
            echo >&3
        } &  # 注意：将整个模型的大括号块放入后台执行
    done

    # 5. 等待所有后台模型任务完成
    wait
    # 关闭文件描述符
    exec 3>&-
    
    echo "========================================"
    echo "VITA实验: 所有模型任务已完成!"
    echo "日志文件位置: $RESULTS_DIR/"
    echo "CSV输出位置: $CSV_DIR/vitabench/results_dir/agent_llm_*"
}

# 运行主函数
main "$@"



