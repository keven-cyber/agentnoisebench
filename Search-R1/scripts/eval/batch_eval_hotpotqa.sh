#!/usr/bin/env bash
set -euxo pipefail

mkdir -p ./logs/eval_hotpotqa/incomplete ./logs/eval_hotpotqa/error ./logs/eval_hotpotqa/redundancy ./logs/eval_hotpotqa/induce ./logs/eval_hotpotqa/failure

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_MODEL="/NAS/ykwang/resources/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo"  # 在这里设置基础模型路径

export VAL_DATA_NUM=200 # 在这里设置验证数据集的样本数量,null表示使用全部样本
export VAL_BATCH_SIZE=8 # 在这里设置验证数据集的批量大小

export RETRIEVER_URL="http://127.0.0.1:8008/retrieve"  # 在这里设置检索器服务的URL
export API_BASE_URL="" # 这里填chatgpt的api base url

# clean(不加噪，作为baseline)
export EXP_NAME="evaluate_hotpotqa_clean"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_clean.sh > ./logs/eval_hotpotqa/clean/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID0=$!
echo $PID0 > ./logs/eval_hotpotqa/clean/last.pid

# incomplete
export EXP_NAME="evaluate_hotpotqa_incomplete_config1"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_incomplete.sh > ./logs/eval_hotpotqa/incomplete/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID1=$!
echo $PID1 > ./logs/eval_hotpotqa/incomplete/last.pid

# 等待第一个任务结束（保证顺序执行）
wait ${PID1}

# error
export EXP_NAME="evaluate_hotpotqa_error_config1"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_error.sh > ./logs/eval_hotpotqa/error/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID2=$!
echo $PID2 > ./logs/eval_hotpotqa/error/last.pid

# 等待第二个任务结束（保证顺序执行）
wait ${PID2}

# redundancy
export EXP_NAME="evaluate_hotpotqa_redundancy_config1"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_redundancy.sh > ./logs/eval_hotpotqa/redundancy/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID3=$!
echo $PID3 > ./logs/eval_hotpotqa/redundancy/last.pid

# 等待第三个任务结束（保证顺序执行）
wait ${PID3}

# induce
export EXP_NAME="evaluate_hotpotqa_induce_config1"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_induce.sh > ./logs/eval_hotpotqa/induce/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID4=$!
echo $PID4 > ./logs/eval_hotpotqa/induce/last.pid

# 等待第四个任务结束（保证顺序执行）
wait ${PID4}

# failure
export EXP_NAME="evaluate_hotpotqa_failure_config1"  # 在这里设置实验名称便于在保存的结果中区分

nohup bash scripts/eval/hotpotqa/evaluate_failure.sh > ./logs/eval_hotpotqa/failure/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID5=$!
echo $PID5 > ./logs/eval_hotpotqa/failure/last.pid

# 等待第五个任务结束（保证顺序执行）
wait ${PID5}

echo "All evaluation tasks completed."



