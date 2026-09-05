想到什么记什么：

1. 用来加噪的LLM。这个需要作为common config的内容，用户指定api_key，base_url，model_name。使用OpenAI lib，实例化一个client model作为noiser的成员变量。
2. 仔细思考noise config的设计。这块怎么实现？分成common cfg和specific cfg吗？
3. 想一想怎么实现incomplete的加噪流程和配套的一些成员变量和方法

比如：
