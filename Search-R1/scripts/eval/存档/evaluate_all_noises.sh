#!/usr/bin/env bash
set -euxo pipefail

mkdir -p ./logs/incomplete ./logs/error ./logs/redundancy ./logs/induce ./logs/failure

# 在这里设置两个个变量用来指定测试的模型和数据集路径，这些变量会被传入各个评估脚本中使用


nohup bash scripts/eval/evaluate_incomplete.sh > ./logs/incomplete/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID1=$!
echo $PID1 > ./logs/incomplete/last.pid

# 等待第一个任务结束（保证顺序执行）
wait ${PID1}

# 第一个任务完成后再启动第二个
nohup bash scripts/eval/evaluate_error.sh > ./logs/error/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID2=$!
echo $PID2 > ./logs/error/last.pid

# 等待第二个任务结束（保证顺序执行）
wait ${PID2}

第二个任务完成后再启动第三个
nohup bash scripts/eval/evaluate_redundancy.sh > ./logs/redundancy/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID3=$!
echo $PID3 > ./logs/redundancy/last.pid

# 等待第三个任务结束（保证顺序执行）
wait ${PID3}

第三个任务完成后再启动第四个(2wiki的最后一个)
nohup bash scripts/eval/evaluate_induce.sh > ./logs/induce/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID4=$!
echo $PID4 > ./logs/induce/last.pid

# 等待第四个任务结束（保证顺序执行）
wait ${PID4}

hotpotqa
nohup bash scripts/eval/evaluate_incomplete.sh > ./logs/incomplete/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
PID1=$!
echo $PID1 > ./logs/incomplete/last.pid

# # 等待第一个任务结束（保证顺序执行）
# wait ${PID1}

# nohup bash scripts/eval/evaluate_error.sh > ./logs/error/$(date +%Y%m%d_%H%M%S).log 2>&1 & 
# PID2=$!
# echo $PID2 > ./logs/error/last.pid


# wait ${PID2}

# 第二个任务完成后再启动第三个
nohup bash scripts/eval/evaluate_redundancy.sh > ./logs/redundancy/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID3=$!
echo $PID3 > ./logs/redundancy/last.pid

wait ${PID3}

# 第三个任务完成后再启动第四个
nohup bash scripts/eval/evaluate_induce_hotpot.sh > ./logs/induce/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID4=$!
echo $PID4 > ./logs/induce/last.pid

# 等待第四个任务结束（保证顺序执行）
wait ${PID4}

nohup bash scripts/eval/evaluate_failure.sh > ./logs/failure/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID4=$!
echo $PID4 > ./logs/failure/last.pid

# 等待第四个任务结束（保证顺序执行）
wait ${PID4}


