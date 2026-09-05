# 与噪声无关的配置
和evaluate_official.sh中的一致，包括prompt_length, max_turns等等

# 与噪声相关的配置
## 第一套
对五种噪声都采取:

从第一步开始，尝试给每一步都加噪，每一步被加噪的概率设为0.7，随机加噪。

```
+noise.enable=true \
    +noise.type="failure" \
    +noise.seed=42 \
    +noise.temperature=0.7 \
    +noise.api.model="gpt-4o-mini" \
    +noise.api.base_url="https://coultra.blueshirtmap.com/v1" \
    +noise.api.key_env="OPENAI_API_KEY" \
    +noise.failure.apply_always=false \
    +noise.failure.start_step=1 \
    +noise.failure.prob_per_step=0.7 \
    +noise.failure.interval_steps=0 \
    +noise.failure.max_times_per_trail=-1 
```
