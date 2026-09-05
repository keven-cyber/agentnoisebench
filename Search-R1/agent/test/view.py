Response(
    id='resp_0bce93d7c30aeabb00692e8bd26f408194848410a0cb3c01af', 
    created_at=1764658130.0, 
    error=None, 
    incomplete_details=None, 
    instructions='You are a helpful assistant.', 
    metadata={}, model='gpt-4o-2024-11-20', 
    object='response', 
    output=[
        ResponseOutputMessage(id='msg_0bce93d7c30aeabb00692e8bd3240c8194a285fc0a7b211779', 
            content=[ResponseOutputText(annotations=[], text='*Sapiens: A Brief History of Humankind* was written by **Yuval Noah Harari**, an Israeli historian and philosopher. The book, first published in 2011 in Hebrew, was later translated into English in 2014, gaining worldwide popularity.', type='output_text', logprobs=[])], role='assistant', status='completed', type='message')], parallel_tool_calls=True, temperature=1.0, tool_choice='auto', tools=[FunctionTool(name='search_documents', parameters={'additionalProperties': False, 'properties': {'query': {'description': '用户问题或查询语句，会直接用于在线检索。', 'type': 'string'}, 'topk': {'default': 3, 'description': '要返回的文档数量（1-10 条）。', 'maximum': 10, 'minimum': 1, 'type': 'integer'}}, 'required': ['query', 'topk'], 'type': 'object'}, strict=True, type='function', description='在线检索服务，根据用户问题检索相关文档片段。')], top_p=1.0, background=False, conversation=None, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, prompt_cache_key=None, reasoning=Reasoning(effort=None, generate_summary=None, summary=None), safety_identifier=None, service_tier='default', status='completed', text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'), top_logprobs=0, truncation='disabled', usage=ResponseUsage(input_tokens=109, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=55, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=164), user=None, content_filters=None, store=True)