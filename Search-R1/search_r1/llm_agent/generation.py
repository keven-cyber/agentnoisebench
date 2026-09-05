import torch
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import requests

from noise_bench.noiser.factory import NoiserFactory
from noise_bench.noiser.base import NoiserContext
from noise_bench.config import NoiseConfig

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str : list[str] = self.tokenizer.batch_decode(  # 一个batch的res_str[[res1],[res2],...[resn]]
            responses, 
            skip_special_tokens=True
        )

        # 模型响应中`</search>`或`</answer>`标签以首个出现为准，因生成过程被约束为单次操作输出
        # 多次工具调用不在同一响应中处理，首个操作标签后内容视为无效(防止模型一次性输出多个<search>，强制要求step-by-step)，把后面可能存在的<search>截掉不要了
        
        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
            
        responses = self._batch_tokenize(responses_str)  # 将处理好的res重新tokenize,返回token_ids和str。NOTE：这里的batch_tokenize会根据最长的数据的长度进行padding。
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        print(f"[DEBUG] num_gpus: {num_gpus}, active_batch size: {active_batch.batch['input_ids'].shape[0]}")
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long() # 这里为什么要转成long类型？
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():  # k: ("input_ids", "attention_mask", "position_ids"), v: tensor(batch_size, seq_len)
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            # 把第一个样本复制 padding_size 次拼接到批次末尾，以便批次大小变成能被 num_gpus 整除。生成完成后再把多出的 padding 输出去掉（trim）。
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, noise_config:NoiseConfig) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)  # 一个batch大小的list，表示哪些样本还在active状态(还没有结束)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)   # 一个batch大小的list，记录每个样本已经进行了多少turn。NOTE：初始化为1
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # 一个batch大小的list，记录每个样本valid action的次数
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # 一个batch大小的list，记录每个样本valid search的次数
        active_num_list = [active_mask.sum().item()] # 记录每一步active的样本数量，方便debug观察。比如第1个turn(也就是现在)，所有的样本都是active的，所以是batch_size；后面每一步可能有一些样本结束了，就会减少。
        
        rollings = gen_batch

        # NOTE：现在相当于是对每一个batch都会有实例化一个noiser，当这个batch结束后，这个noiser就会被销毁
        noiser = NoiserFactory.get_noiser_instance(noise_config) # 根据type实例化对应的noiser

        # Main generation loop
        for step in range(self.config.max_turns): # 比如max_turns=4,则这里就是0,1,2,3共4个step。
            if not active_mask.sum(): # 所有样本都结束了，跳出循环
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(  # 左截断（设置的max_prompt_length是2048，左侧padding，所以右侧是对齐的，可能长度是1800，600，300，那就取最长的1800，把左边的248截断，短一点计算量就少一点）
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()  # 取出batch中还active的样本
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active) # 只包含新生成的部分，不含输入的prompt部分

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses']) # 主要目的是：可能responses中含有多个<search>或者<answer>标签，只保留第一个标签及其内容，后面的都不要了。
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, contents = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            ############ 示例加噪代码 #############
            """
            for i, tool_call in enumerate(self.message.tool_calls):
                ConsoleDisplay.console.print(f"\n💾 [bold green]Start[/bold green]")
                ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {tool_call}[/bold green]")
                tool_msg = self.environment.get_response(tool_call)
                tool_args = tool_call.arguments
                ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {type(tool_args)}[/bold green]")
                # 在这里产生噪声,将里面的%prompt1%替换为tool_msg，%prompt2%替换为user_msg, %prompt3%替换为1
                # ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {tool_call}[/bold green]")
                
                # ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {type(tool_call)}[/bold green]")
                # ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: {self.environment}[/bold red]")
                # ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: {type(self.environment)}[/bold red]")
                # ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {tool_call.name}[/bold green]")
                # ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: {type(tool_call.name)}[/bold red]")
                # 首先就是这里的tool_call里面的参数是什么
                if tool_call.name == 'repair_default':
                    pre_tool_call = tool_call.arguments['pre_tool_name']
                    self.tools_dict[pre_tool_call] = self.max_single_tool_noise + 1
                if_add_noise = False # 初始化
                if tool_call.name == 'repair_default' or (self.index < self.noise_nums and self.tools_dict[tool_call.name] < self.max_single_tool_noise):  # 只在指定的次数内添加噪声
                    # 新建一个prompt对应的字典
                    # from vita.noise.system_prompt import zh_add_wrong, zh_add_induce, zh_add_redundant, zh_add_incomplete
                    
                    noise_category_sysprompt_dict = {'wrong':zh_add_wrong, 'redundant':zh_add_redundant, 'induce':zh_add_induce, 'incomplete':zh_add_incomplete, 'default':'', 'others':''}
                    ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: 1[/bold red]")
                    noise_machine = tool_noise_bench(
                        messages=tool_msg.content, 
                        llm='o4-mini', 
                        API_key="<api key>",  # 替换为实际的API key
                        Base_url="https://www.blueshirtmap.com/v1",  # 替换为实际的Base URL
                        sys_prompt=noise_category_sysprompt_dict[self.noise_category],
                        user_msg=self.latest_user_msg.content,
                        Category=self.noise_category,  # 噪声类别
                        priority_level=self.priority_level,
                        nums=self.noise_nums,
                        environment=self.environment,
                        tool_call=tool_call,
                        instructions=self.task.instructions,
                    )
                    ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: 2[/bold red]")
                    if_add_noise, noisy_tool_msg = noise_machine.Add_Noise()
                    # ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: {en_add_blurred if self.noise_category=='blurred' else zh_add_wrong}[/bold red]")
                    ConsoleDisplay.console.print(f"\n💾 [bold red]Results appended to CSV: {noisy_tool_msg}[/bold red]")
                    tool_msg.content = noisy_tool_msg  # 替换为添加噪声后的内容
                    ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {tool_msg.content}[/bold green]")
                    # 目前这里是就算调用多个也只出现一次错误
                    if if_add_noise == True:
                        self.tools_dict[tool_call.name] += 1
                        self.index += 1
                ConsoleDisplay.console.print(f"\n💾 [bold green]Results appended to CSV: {tool_msg.content}[/bold green]")
                tool_msgs.append(tool_msg)
            """
            
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # 【位置2】TODO：在这里进行加噪处理
            noisy_next_obs_list = []
            
            # 注意它的奇特的变量命名吧。ob代表单条observation，obs表示复数
            for i, next_ob in enumerate(next_obs):
                ## 引入加噪器
                if noiser is None:
                    print("[WARNING] Noiser is None, skip adding noise.")
                    noisy_next_obs_list.append(next_ob)
                    continue
                
                ctx = NoiserContext(
                    step_idx=step,
                    is_search=bool(is_search[i]), # 只有是is_search才加噪声
                    max_turns=self.config.max_turns,
                    trail_id=i,  # 用批内索引作为 trail_id；同一条样本跨 step 索引一致
                )
                
                content = contents[i] # 搜索的关键词/答案内容/“”
                
                added_noise, noisy_next_ob = noiser.maybe_add_noise(content, next_ob, ctx) # 留出来一个标志位，表示是否加噪成功。
                
                print(f"【王昱凯debug】加噪成功?{added_noise}, 原始ob:{next_ob}, 加噪后ob:{noisy_next_ob}")
                
                noisy_next_obs_list.append(noisy_next_ob)
            
            next_obs_ids = self._process_next_obs(noisy_next_obs_list)
            # next_obs_ids = self._process_next_obs(next_obs)  # 这一步就是把next_obs（str）转成token_ids了
            
            # Update states.把模型生成的response和环境返回的observation拼接到rolling和right_side中，作为下一步的输入。
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            # 单独维护一个只包含右边侧的状态，方便最后输出
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout.达到最大turns后，再进行一次生成，看看能不能直接给出答案。
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, contents = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # 最后一轮了，rollings不需要更新了，只需要把responses拼接到right_side中即可
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """
        Compose final generation output.
        final_output includes:
            - prompts: left_side input_ids
            - input_ids: concatenated left_side input_ids and right_side responses
            - attention_mask: combined attention mask
            - info_mask: combined info mask
            - position_ids: generated position ids
        """
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> Tuple[List[str], List[int], List[int], List[int], List[str]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: example_level_padded responses_str.
            pad_token: Token to use for padding
            active_mask: Mask indicating active examples
            
        Returns:
            List of observation strings
            List of done flags
            List of valid action flags
            List of is_search flags
            List of contents extracted from predictions(including search query or final answer)
        """
        cur_actions, contents = self.postprocess_predictions(predictions) # 两种action: search or answer
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            # 测试的时候先把val_batch设为1吧
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append('\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, contents

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
