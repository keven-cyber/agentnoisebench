# Defines a class that injects noise; the noise types are tool-call failure and
# blurred tool results.
# todo: Error 401 Unauthorized。Error 403 Forbidden.Error 408 Request Timeout​​ Error 429 Too Many Requests
# Introduce a separate variable to distinguish the total number of injected noises
# from the per-tool usage cap.
#
# from vita.noise.system_prompt import en_add_blurred
from tau2.domains.airline.tools_noise_schema import TOOL_DESCRIPTIONS_WRONG_EN
# The description is no longer needed here; import it directly instead.
# from tau2.domains.delivery.tools_all import TOOL_DESCRIPTIONS_ZH
from tau2.data_model.message import ToolCall
from tau2.environment.environment import Environment
import random
import json
import re
import time
from loguru import logger
import requests
from tau2.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
tool_calls = ['delivery_store_search_recommand', 'delivery_product_search_recommand', "search_direct_flight", "search_onestop_flight"]
def replace_placeholders(template_str, tool_msg, user_msg, priority_level):

    result = template_str.replace("%prompt1%", tool_msg)
    result = result.replace("%prompt2%", user_msg)
    result = result.replace("%prompt3%", str(priority_level))

    return result
def replace_placeholders_wrong(template_str, instruction):

    result = template_str.replace("%prompt1%", instruction)
    return result

class tool_noise_bench():
    def __init__(
        self,
        domain: str,
        messages: str,
        llm: str,
        API_key: str,
        sys_prompt: str,
        Base_url: str,
        user_msg: str,
        environment: Environment,
        tool_call: ToolCall,
        Category: str=None,
        priority_level: int=1,
        nums: int=1,
        instructions: str=None,
        tools_description: str=None,
        language: str=None,
    ):
        # All attribute initialisation below must stay indented (4 spaces).
        self.llm = llm
        self.domain = domain
        self.priority_level = priority_level
        self.user_msg = user_msg
        self.API_key = API_key
        self.Base_url = Base_url
        self.sys_prompt = sys_prompt
        self.Category = Category
        self.messages = messages
        self.nums = nums
        self.environment = environment
        self.tool_call = tool_call.name
        self.instructions = instructions
        self.tool_name_description = tools_description

    def ask_ai(self):
        from openai import OpenAI
        if self.Category=='blurred':
            input_messages = replace_placeholders(self.sys_prompt, self.messages, self.user_msg, self.priority_level)
        elif self.Category=='wrong':
            # input_messages = replace_placeholders_wrong(self.sys_prompt, self.user_msg)
            input_messages = self.sys_prompt.format(a=self.tool_name_description, b=TOOL_DESCRIPTIONS_WRONG_EN, c=self.tool_call, d=self.messages, e=self.instructions)

        # Redundancy uses a different set of placeholders here.
        elif self.Category=='redundant':
            input_messages = self.sys_prompt.format(a=self.messages, b=self.instructions)
        elif self.Category=='induce':
            input_messages = self.sys_prompt.format(a=self.messages, c=self.instructions)
        elif self.Category=='incomplete':
            input_messages = self.sys_prompt.format(a=self.messages, b=self.instructions)
        print('self.llm:',self.llm)
        print('self.API_key:',self.API_key)
        client = OpenAI(
            api_key=self.API_key,
            base_url=self.Base_url,
        )
        print('self.llm:',self.llm)

        response = client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "user", "content": input_messages},
            ]
        )
        # messages_formatted = []
        # messages_formatted.append({"role": "user", "content": input_messages})
        # print('messages_formatted:',messages_formatted)
        # try:
        #     data = {
        #         "model": 'gpt-3.5-turbo',
        #         "messages": messages_formatted,
        #         "stream": False,
        #         "temperature": 0.7,
        #     }
        #     data.update(models['gpt-3.5-turbo'])
        #     # data = kwargs_adapter(data, enable_think, messages)
        #     headers = models['gpt-3.5-turbo']["headers"]
        #     max_retries = 3
        #     retry_delay = 1
        #     for attempt in range(max_retries + 1):
        #         try:
        #             response = requests.post(data["base_url"], json=data, headers=headers, timeout=(10, 600))
        #             # print('response:',response)
        #             if response.status_code != 500:
        #                 response = response.json()
        #                 break
        #             if attempt < max_retries:
        #                 logger.warning(f"API returned 500 error, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
        #                 time.sleep(retry_delay)
        #                 retry_delay *= 2
        #             else:
        #                 response.raise_for_status()

        #         except requests.exceptions.RequestException as e:
        #             if attempt < max_retries:
        #                 logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
        #                 time.sleep(retry_delay)
        #                 retry_delay *= 2
        #             else:
        #                 raise e
        # except Exception as e:
        #     logger.error(e)
        #     raise e
        # print('response:',response)
        evaluation_response = response.choices[0].message.content
        return evaluation_response

    def Add_fault(self):
        error_list = [
            'Error 401 Unauthorized',
            'Error 403 Forbidden',
            'Error 408 Request Timeout',
            'Error 429 Too Many Requests'
        ]
        random_index = random.randint(0, len(error_list) - 1)
        if self.tool_call == 'repair_default':
            # self.tools_dict[self.tool_call] = self.max_single_tool_noise
            return False, self.messages
        else:
            return True, error_list[random_index]
    def Add_blurred(self):
        return True, self.ask_ai()
    def Add_wrong(self):
        key_list = [key for key in TOOL_DESCRIPTIONS_WRONG_EN]
        if self.tool_call in key_list:
            return True, self.ask_ai()
        else:
            return False, self.messages
    def Add_Redundant(self):
        return True, self.ask_ai()
    def Add_induce(self):
        # Modify the tags by prefixing them with a recommendation.
        if self.domain == 'airline' or self.domain == 'telecom' or self.domain == 'retail':
            return True, self.ask_ai()
        else:
            return False, self.messages
    def Add_incomplete(self):
        return True, self.ask_ai()
    def Add_Noise(self):
        if self.Category == "wrong":
            return self.Add_wrong()
        elif self.Category == "blurred":
            return self.Add_blurred()
        elif self.Category == "induce":
            return self.Add_induce()
        elif self.Category == "redundant":
            return self.Add_Redundant()
        elif self.Category == "fault":
            return self.Add_fault()
        elif self.Category == "incomplete":
            return self.Add_incomplete()
        else:
            return False, self.messages
