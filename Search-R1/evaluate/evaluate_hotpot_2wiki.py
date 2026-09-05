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
import yaml

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
--model deepseek-r1-0528 \
--models-yaml ./models.yaml \
--max-turns 5 \
--max-output-tokens 512 \
--temperature 0.7 \
--max-examples 10 \
--output-dir ./eval_results/1000/hotpotqa/redundancy \
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
    parser = argparse.ArgumentParser(description="Evaluate SearchAgent on HotpotQA / 2WikiMultiHopQA with noise.")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="在 models.yaml 里的模型键名；若未提供 --models-yaml，则作为 API 模型名使用")
    parser.add_argument("--models-yaml", type=str, default=None, help="模型配置 YAML 路径（若提供，则用其中的 defaults + models[model]）")

    # 让 CLI 对生成参数的默认值为 None，这样才能优先采用 YAML 中的值
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=None, help="如不传则优先采用 models.yaml 的 max_completion_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="如不传则优先采用 models.yaml 的 temperature")

    # 思考相关（可覆盖 YAML）
    parser.add_argument("--enable-thinking", action="store_true", help="强制开启思考模式（覆盖 YAML）")
    parser.add_argument("--disable-thinking", action="store_true", help="强制关闭思考模式（覆盖 YAML）")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"],
                        help="为支持 reasoning 的代理设置 effort；如不传则优先采用 YAML")

    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")

    # 噪声相关
    parser.add_argument("--noise-type", type=str, required=True,
                        choices=["clean", "failure", "error", "induce", "incomplete", "redundancy"])
    parser.add_argument("--noise-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--apply-always", action="store_true")
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--prob-per-step", type=float, default=1.0)
    parser.add_argument("--interval-steps", type=int, default=1)
    parser.add_argument("--max-times-per-trail", type=int, default=4)
    return parser.parse_args()

def _merge_model_profile(args) -> Dict[str, Any]:
    """
    从 models.yaml 载入 defaults + 指定模型条目，最终再用 CLI 覆盖，返回合并配置。
    """
    profile = {
        "api_model": args.model,           # 若不提供 models.yaml，就直接把 --model 当 api_model 用
        "max_completion_tokens": 512,
        "temperature": 0.7,
    }
    if args.models_yaml:
        with open(args.models_yaml, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        defaults = raw.get("defaults", {})
        models = raw.get("models", {})
        entry = models.get(args.model)
        if entry is None:
            raise KeyError(f"models.yaml 中未找到模型键：{args.model}")
        # defaults -> entry
        for k, v in defaults.items():
            profile[k if k != "max_completion_tokens" else "max_completion_tokens"] = v
        for k, v in entry.items():
            # k: api_model / max_completion_tokens / temperature / enable_thinking / reasoning_effort
            profile[k if k != "max_completion_tokens" else "max_completion_tokens"] = v

    # CLI 覆盖
    if args.max_output_tokens is not None:
        profile["max_completion_tokens"] = args.max_output_tokens
    if args.temperature is not None:
        profile["temperature"] = args.temperature
    if args.enable_thinking:
        profile["enable_thinking"] = True
    if args.disable_thinking:
        profile["enable_thinking"] = False
    if args.reasoning_effort is not None:
        profile["reasoning_effort"] = args.reasoning_effort

    return profile

def main():
    args = parse_args()

    # OpenAI 兼容客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    # 读取模型配置（YAML + CLI 合并）
    m = _merge_model_profile(args)
    if args.verbose:
        print("[Model Profile]", json.dumps(m, ensure_ascii=False, indent=2))

    cfg = AgentConfig(
        model=m["api_model"],
        max_turns=args.max_turns,
        max_output_tokens=m["max_completion_tokens"],
        temperature=m["temperature"],
        verbose=args.verbose,
        enable_thinking=m["enable_thinking"],
        reasoning_effort=m["reasoning_effort"],
    )

    # 噪声模块
    noiser = None
    if args.noise_type != "clean":
        noise_config = noise_config_preparation(args)
        noiser = NoiserFactory.get_noiser_instance(noise_config)

    agent = SearchAgent(
        client=client,
        config=cfg,
        searcher=Searcher(),
        noiser=noiser
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.model.replace('/', '_')}.json",
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
