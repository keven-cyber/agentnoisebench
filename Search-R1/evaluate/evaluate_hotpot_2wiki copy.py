import os
import json
import argparse
import string
from collections import Counter
from typing import List, Dict, Any, Iterable, Optional

from tqdm import tqdm
from openai import OpenAI

from agent.llm_agent import SearchAgent, AgentConfig
from agent.searcher import Searcher
from noise_bench.config import NoiseAPIConfig, NoiseCommonConfig, IncompleteConfig, FailureConfig, RedundancyConfig, InduceConfig, ErrorConfig, NoiseConfig
from noise_bench.noiser.factory import NoiserFactory

# ============ noise config ============
def noise_config_preparation(args) -> NoiseConfig:
        api_cfg = NoiseAPIConfig(
            model=args.noise_model
        )
    
        noise_type = args.noise_type
        
        if noise_type != "clean":
            enable_noise = True
        else:
            enable_noise = False
    
        common_cfg = NoiseCommonConfig(
            api=api_cfg,
            enable=enable_noise,
            noise_type=noise_type,
        )
        
        if noise_type == "incomplete":
            return NoiseConfig(
                common=common_cfg,
                incomplete=IncompleteConfig(
                    language=args.language,
                    apply_always=args.apply_always,
                    start_step=args.start_step,
                    prob_per_step=args.prob_per_step,
                    interval_steps=args.interval_steps,
                    max_times_per_trail=args.max_times_per_trail,
                ),
            )
        elif noise_type == "failure":
            return NoiseConfig(
                common=common_cfg,
                failure=FailureConfig(
                    apply_always=args.apply_always,
                    start_step=args.start_step,
                    prob_per_step=args.prob_per_step,
                    interval_steps=args.interval_steps,
                    max_times_per_trail=args.max_times_per_trail,
                ),
            )
        elif noise_type == "redundancy":
            return NoiseConfig(
                common=common_cfg,
                redundancy=RedundancyConfig(
                    language=args.language,
                    apply_always=args.apply_always,
                    start_step=args.start_step,
                    prob_per_step=args.prob_per_step,
                    interval_steps=args.interval_steps,
                    max_times_per_trail=args.max_times_per_trail,
                ),
            )
        elif noise_type == "induce":
            return NoiseConfig(
                common=common_cfg,
                induce=InduceConfig(
                    language=args.language,
                    apply_always=args.apply_always,
                    start_step=args.start_step,
                    prob_per_step=args.prob_per_step,
                    interval_steps=args.interval_steps,
                    max_times_per_trail=args.max_times_per_trail,
                ),
            )
        elif noise_type == "error":
            return NoiseConfig(
                common=common_cfg,
                error=ErrorConfig(
                    language=args.language,
                    apply_always=args.apply_always,
                    start_step=args.start_step,
                    prob_per_step=args.prob_per_step,
                    interval_steps=args.interval_steps,
                    max_times_per_trail=args.max_times_per_trail,
                ),
            )
        else:
            return NoiseConfig(
                common=common_cfg,
            )   
        

        


# ============ 文本规整 & 指标 ============

def normalize_answer(s: str) -> str:
    """标准 Hotpot/SQuAD 风格的归一化."""
    def remove_articles(text):
        return " ".join(
            [w for w in text.split() if w.lower() not in ("a", "an", "the")]
        )

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: List[str]
) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ============ 数据加载 ============

def load_hotpotqa(path: str, max_examples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """
    适配官方 HotpotQA JSON 格式：
    [
      {"id": ..., "question": ..., "golden_answers": ...},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, example in enumerate(data):
        if max_examples is not None and i >= max_examples:
            break
        qid = example.get("id", str(i))
        question = example["question"]
        ans = example["golden_answers"]
        if isinstance(ans, list):
            gold_answers = [str(a) for a in ans]
        else:
            gold_answers = [str(ans)]
        yield {
            "id": qid,
            "question": question,
            "gold_answers": gold_answers,
        }


def load_2wikimultihop(path: str, max_examples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """
    适配常见的 2WikiMultiHopQA JSON 格式：
    [
      {"id": ..., "question": ..., "golden_answers": ...},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, example in enumerate(data):
        if max_examples is not None and i >= max_examples:
            break
        qid = example.get("id", str(i))
        question = example["question"]
        ans = example["golden_answers"]
        if isinstance(ans, list):
            gold_answers = [str(a) for a in ans]
        else:
            gold_answers = [str(ans)]
        yield {
            "id": qid,
            "question": question,
            "gold_answers": gold_answers,
        }


def get_dataset_loader(dataset_name: str):
    if dataset_name.lower() == "hotpotqa":
        return load_hotpotqa
    elif dataset_name.lower() in ("2wikimultihop", "2wikimultihopqa", "2wiki"):
        return load_2wikimultihop
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


# ============ 评估主逻辑 ============

def evaluate_agent_on_dataset(
    agent: SearchAgent,
    dataset_name: str,
    dataset_path: str,
    max_examples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    loader = get_dataset_loader(dataset_name)
    examples = list(loader(dataset_path, max_examples=max_examples))
    n = len(examples)

    total_em = 0.0
    total_f1 = 0.0
    predictions: List[Dict[str, Any]] = []

    for ex in tqdm(examples, desc=f"Evaluating on {dataset_name}"):
        qid = ex["id"]
        question = ex["question"]
        gold_answers = ex["gold_answers"]

        run_result = agent.run(question)
        pred = run_result.final_answer

        em = metric_max_over_ground_truths(exact_match_score, pred, gold_answers)
        f1 = metric_max_over_ground_truths(f1_score, pred, gold_answers)

        total_em += float(em)
        total_f1 += float(f1)

        predictions.append({
            "id": qid,
            "question": question,
            "prediction": pred,
            "gold_answers": gold_answers,
            "em": em,
            "f1": f1,
            "steps": run_result.steps,
            "finish_reason": run_result.finish_reason,
        })

    avg_em = 100.0 * total_em / n if n > 0 else 0.0
    avg_f1 = 100.0 * total_f1 / n if n > 0 else 0.0

    metrics = {
        "dataset": dataset_name,
        "num_examples": n,
        "EM": avg_em,
        "F1": avg_f1,
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "predictions": predictions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return metrics


# ============ CLI ============
"""
示例运行命令：
python evaluate/evaluate_hotpot_2wiki.py \
--dataset hotpotqa \
--dataset-path ./data/nq_hotpotqa_train/test_hotpotqa.json \
--model gpt-4o \
--max-turns 5 \
--max-output-tokens 512 \
--temperature 0.7 \
--max-examples 10 \
--output-dir ./eval_results \
--verbose \
--noise-type redundancy \
--noise-model gpt-4o-mini \
--language zh \
--apply-always \
--start-step 1 \
--prob-per-step 0.5 \
--interval-steps 1 \
--max-times-per-trail 3

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SearchAgent on HotpotQA / 2WikiMultiHopQA with noise."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "2wikimultihop"],
                        help="数据集名称")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="数据集 JSON 文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="商业 LLM 的模型名，例如 gpt-4o, deepseek-r1 等")
    parser.add_argument("--max-turns", type=int, default=4,
                        help="agent 最大工具调用步数")
    parser.add_argument("--max-output-tokens", type=int, default=512,
                        help="每步 LLM 生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="仅用前 N 条样本（调试用）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="保存预测和指标的目录")
    parser.add_argument("--verbose", action="store_true")
    # 噪声相关的参数
    parser.add_argument("--noise-type", type=str, required=True,
                        choices=["clean", "failure", "error", "induce", "incomplete", "redundancy"],
                        help="加噪类型，clean 表示不加噪")
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="随机种子")
    parser.add_argument("--noise-model", type=str, default="gpt-4o-mini",
                        help="用来加噪声的模型名称，默认为 gpt-4o-mini")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh"],
                        help="噪声生成使用的语言")
    parser.add_argument("--apply-always", action="store_true",
                        help="是否每一步都加噪")
    parser.add_argument("--start-step", type=int, default=1,
                        help="从第几步开始加噪（apply_always=false 时生效）")
    parser.add_argument("--prob-per-step", type=float, default=1.0,
                        help="每步加噪的概率（apply_always=false 时生效）")
    parser.add_argument("--interval-steps", type=int, default=1,        
                        help="两次加噪至少间隔步数（apply_always=false 时生效）")
    parser.add_argument("--max-times-per-trail", type=int, default=4,
                        help="单条样本最多加噪次数（apply_always=false 时生效）")   
    
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化 OpenAI 客户端（兼容各种 OpenAI 格式的代理 / 商业服务）
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    cfg = AgentConfig(
        model=args.model,
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # 这里接入噪声模块，根据 noise_type 构造 noiser
    
    if args.noise_type != "clean":
        noise_config = noise_config_preparation(args)
        noiser = NoiserFactory.get_noiser_instance(noise_config) # 根据type实例化对应的noiser
        
    else:
        noiser = None
    
    
    agent = SearchAgent(
        client=client,
        config=cfg,
        searcher=Searcher(),
        noiser=noiser #  一个agent注入一个noiser
    ) 

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"pred_{args.dataset}_model_{args.model.replace('/', '_')}_noise_{args.noise_type}.json",
    )

    metrics = evaluate_agent_on_dataset(
        agent=agent,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
        output_path=output_file,
    )

    print("\n=== Evaluation Finished ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
