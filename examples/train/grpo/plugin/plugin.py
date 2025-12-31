import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union, Tuple, Optional, Any ,Set
import json
import torch
import math

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

import numpy as np
from openai import OpenAI
import requests

import subprocess
import uuid
from pathlib import Path



logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards

orms['external_r1v_acc'] = MultiModalAccuracyORM


# R1 格式奖励
class EvidenceDrivenFormatORM(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion has a evidence-driven format.
        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores
        """

        pattern = (
            r'^<thinking>\s*'
            r'<perception>.*?</perception>\s*'
            r'<reasoning>.*?</reasoning>\s*'
            r'<review>.*?</review>\s*'
            r'</thinking>\s*'
            r'<answer>.*?</answer>\s*$'
        )
        matches = [re.match(pattern, content.strip(), re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


orms['evidence_driven_format'] = EvidenceDrivenFormatORM

# R2 答案+review奖励
class MCQAnswerExactORM(ORM):
    """
    整合型奖励函数：
    1. 计算 R_ans (Exact Match)。
    2. 如果 R_ans > 0 (答对了)，则调用 vLLM 计算 R_rev (Review Quality)。
    3. 最终分数 = R_ans * (1.0 + alpha * R_rev)。
    """

    def __init__(self, review_alpha: float = 0.5):
        self._ans_block_re = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
        self.review_alpha = review_alpha  # 乘法系数，默认 0.5
        
        # 初始化 Review 打分器
        self.review_scorer = ReviewScorer(
            base_url="http://127.0.0.1:8000/v1", # 根据你的实际配置修改
            served_model="Qwen3-32B"
        )

    # --- Answer Match 相关的辅助方法 ---
    def _extract_answer_block(self, text: str) -> Optional[str]:
        if not text: return None
        m = self._ans_block_re.search(text)
        return m.group(1).strip() if m else None

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()
    
    # --- 辅助方法：提取问题文本 (复用你原来的逻辑) ---
    @staticmethod
    def _concat_one_conversation(conv) -> Optional[str]:
        # ... (此处复用你原来的代码: _concat_one_conversation) ...
        if conv is None: return None
        try:
            if isinstance(conv, str): return conv.strip() or None
            if isinstance(conv, list):
                all_parts = []
                for msg in conv:
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if c: all_parts.append(str(c))
                    elif isinstance(msg, str):
                        all_parts.append(msg)
                return "\n".join(all_parts).strip() or None
        except: return None
        return None

    def _get_question_text(self, messages, idx: int) -> str:
        # ... (此处复用你原来的代码: _get_question_text) ...
        # 简化版示例：
        if messages and isinstance(messages, list) and idx < len(messages):
             raw = self._concat_one_conversation(messages[idx])
             return raw.replace("<audio>", "").strip() if raw else "Unknown Question"
        return "Unknown Question"

    # --- Call ---
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        
        # 获取 Batch 数据
        messages = kwargs.get("messages", [])
        batch_time_intervals = kwargs.get("time_interval", [])
        batch_events = kwargs.get("event", [])
        
        # 填充默认值防止越界
        if not batch_time_intervals: batch_time_intervals = [[]] * len(completions)
        if not batch_events: batch_events = [[]] * len(completions)

        for i, (pred_text, gt_ans_text) in enumerate(zip(completions, solution)):
            try:
                # --- Step 1: 计算 R_ans (Answer Exact Match) ---
                r_ans = 0.0
                pred_block = self._extract_answer_block(pred_text)
                # gt_ans_text 本身就是答案，不需要再提取
                
                if pred_block and gt_ans_text:
                    if self._normalize(pred_block) == self._normalize(gt_ans_text):
                        r_ans = 1.0
                
                # --- Step 2: 计算 R_rev (仅当 R_ans > 0 时才计算，节省资源) ---
                r_rev = 0.0
                if r_ans > 0.0:
                    # 获取该样本的元数据
                    question = self._get_question_text(messages, i)
                    t_intervals = batch_time_intervals[i] if i < len(batch_time_intervals) else []
                    events = batch_events[i] if i < len(batch_events) else []
                    
                    # 调用 ReviewScorer
                    # 注意：full_cot_text 就是 pred_text
                    r_rev = self.review_scorer.calculate_review_score(
                        full_cot_text=str(pred_text),
                        question_text=question,
                        answer_text=gt_ans_text,
                        gt_time_intervals=t_intervals,
                        gt_events=events
                    )

                # --- Step 3: 组合公式 R = R_ans * (1 + alpha * R_rev) ---
                # 如果答错：1.0 * 0.0 * (...) = 0.0
                # 如果答对且没review：1.0 * (1 + 0.5 * 0) = 1.0
                # 如果答对且review完美：1.0 * (1 + 0.5 * 1.0) = 1.5
                
                final_score = r_ans * (1.0 + self.review_alpha * r_rev)
                rewards.append(final_score)

            except Exception as e:
                # print(f"WeightedAnswerReviewORM Error: {e}")
                rewards.append(0.0)
        
        return rewards

# 注册
orms['answer_exact_match'] = MCQAnswerExactORM

# R3 perception 感知奖励
class PerceptionQualityJudgeORM(ORM):
    """
    使用外部 vLLM 评审 CoT 推理质量，返回模型给出的 [0,1] 分数作为奖励。
    结合了音频事件事实性检查（Audio Fidelity）与逻辑一致性检查。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        served_model: str = "Qwen3-32B",
        max_retries: int = 5
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.served_model = served_model
        self.max_retries = max_retries

    # --- 辅助方法：提取问题文本 ---
    @staticmethod
    def _concat_one_conversation(conv) -> Optional[str]:
        if conv is None:
            return None
        try:
            if isinstance(conv, str):
                return conv.strip() or None
            if isinstance(conv, list):
                user_parts, all_parts = [], []
                for msg in conv:
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if not c: continue
                        all_parts.append(str(c))
                        if msg.get("role") == "user":
                            user_parts.append(str(c))
                    elif isinstance(msg, str):
                        all_parts.append(msg)
                text = "\n".join(user_parts).strip() if user_parts else "\n".join(all_parts).strip()
                return text or None
        except Exception:
            return None
        return None

    @staticmethod
    def _get_question_text(messages, idx: int) -> Optional[str]:
        if messages is None: return None
        if isinstance(messages, list) and messages and isinstance(messages[0], (list, dict, str)):
            if idx < len(messages) and isinstance(messages[0], (list, dict, str)):
                return PerceptionQualityJudgeORM._concat_one_conversation(messages[idx])
        return PerceptionQualityJudgeORM._concat_one_conversation(messages)

    # --- 辅助方法：新增加的数据解析逻辑 ---

    @staticmethod
    def _extract_perception_content(output_text: str) -> str:
        """从 solution 中提取 <perception>...</perception> 的内容"""
        if not output_text:
            return "Unknown"
        match = re.search(r"<perception>(.*?)</perception>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Unknown"

    @staticmethod
    def _build_caption(time_intervals: list, events: list) -> str:
        """将时间戳和事件描述拼接成 Caption"""
        if not time_intervals or not events or len(time_intervals) != len(events):
            return "No audio events detected."
        
        # 按照你的要求： "[{s}, {e}]: {ev}" 格式拼接
        try:
            lines = []
            for (s, e), ev in zip(time_intervals, events):
                # 确保转成字符串时保留一定的精度，或者直接转str
                lines.append(f"[{s}, {e}]: {ev}")
            return " ".join(lines)
        except Exception:
            return "Error parsing audio events."

    # --- 核心：构建新的 Strict Prompt ---
    @staticmethod
    def _build_prompt(caption_text: str, question_text: str, answer_text: str, cot_text: str) -> str:
        prompt_text = (
            "You are an expert audio perception evaluator. I will give you a record containing: \n"
            "1. A Detailed Audio Caption (Ground Truth): A comprehensive, factual text description of the audio events.\n"
            "2. A Question and its Correct Answer: To determine which audio events are 'critical' for the task.\n"
            "3. A Model Perception Output: The content within the <perception> tags generated by the model, describing events with timestamps.\n"

            "Your task is to evaluate the fidelity, precision, and completeness of the Model Perception against the Ground Truth, and output a single numeric score from {0, 0.1, 0.2, ..., 1.0}. "
            "You must output only the score with no explanation or extra text.\n\n"

            "Evaluate based on the following CRITICAL principles:\n\n"

            "1. Audio Hallucination (Strict) — The model must NOT report events that do not exist in the 'Detailed Audio Caption'. "
            "Reporting a sound that is completely absent (e.g., hearing a siren when the description only mentions birds) is a fatal failure.\n"

            "2. Content Accuracy & Sequential Logic — While evaluating the model's generated timestamps, focus on:\n"
            "   - Event Identity: Does the model correctly identify the sound sources described in the Ground Truth? (e.g., distinguishing 'footsteps' from 'knocking').\n"
            "   - Chronological Flow: Does the sequence of events in the model's output match the narrative order of the Ground Truth? (e.g., if the description says 'a door opens then slams', the model must not place the slam before the opening).\n"

            "3. Critical Event Coverage (Relevance) — The model must capture all 'Key Events' necessary to answer the provided 'Question'. "
            "Compare with the Question/Answer pairs: if the answer depends on a specific sound cue, omitting this specific event in the <perception> phase is a critical failure.\n"

            "4. Consistency & Identity — The model should describe the same audio source consistently across different timestamps (unless the sound evolves). "
            "Avoid contradictory descriptions for the same ongoing event.\n"

            "5. Redundancy & Conciseness — The perception output should be dense and informative. "
            "Penalize distinct 'loops' (repeating the exact same phrase for adjacent timestamps) or extreme verbosity that adds no new details.\n\n"

            "Scoring guideline: \n"
            "1.0 = Flawless. Perfectly matches the Ground Truth description. Events are correctly identified and listed in the correct logical order. Concise.\n"
            "0.8-0.9 = Excellent. Accurate detection of all key events described. The sequence is logical. Maybe minor verbosity.\n"
            "0.5-0.7 = Mediocre. The KEY event was detected, but the description is vague, or information irrelevant to the question has been omitted.\n"
            "0.2-0.4 = Poor. Misses a KEY event needed for the Answer, or misidentifies a sound source. Sequence is disorderly compared to the description.\n"
            "0.0-0.1 = Severe. HALLUCINATION (inventing sounds not in the description), or total failure to identify the main audio event.\n\n"

            "Penalty guideline (Apply these cumulatively to reduce the score):\n"
            "*** [CRITICAL PENALTY] (Set Score to 0.0 - 0.2) ***: \n"
            "   - Hallucinating an event not present in the Ground Truth description.\n"
            "   - Misidentifying the main sound source (e.g., 'gunshot' vs 'drum').\n"
            "   - Missing the specific audio cue required to answer the Question.\n\n"
            "*** [MODERATE PENALTY] (-0.3 ~ -0.5) ***: \n"
            "   - Sequential Logic Error (Events are listed in an order contradicting the description).\n"
            "   - Significant omission of details mentioned in the description.\n\n"
            "*** [MINOR PENALTY] (-0.1 ~ -0.2) ***: \n"
            "   - Excessive wordiness or repetitive phrasing without new information.\n"
            "   - Vague descriptions (e.g., 'noise' instead of 'dog barking') if the Ground Truth is specific.\n\n"

            "Operational rule: Always output only one score (0-1 in 0.1 increments)."
            "Now evaluate the following record and output only the score.\n\n"

            f"The Detailed Audio Caption (Ground Truth) is:\n{caption_text}\n\n"
            f"The Question is:\n{question_text}\n\n"
            f"The Correct Answer is:\n{answer_text}\n\n"
            f"The Model perception to evaluate is:\n{cot_text}\n"
        )
        return prompt_text

    # --- API 调用相关 ---
    def _call_judge_once(self, prompt: str) -> Optional[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.served_model,
                messages=[
                    {"role": "system", "content": "You are an expert reasoning evaluator."},
                    {"role": "user", "content": prompt},
                ],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if not resp or not resp.choices:
                return None
            out = resp.choices[0].message.content
            return out if out is not None else None
        except Exception:
            return None

    def _call_judge_with_retry(self, prompt: str) -> Optional[str]:
        for _ in range(max(1, self.max_retries)):
            out = self._call_judge_once(prompt)
            if out is not None and out.strip() != "":
                return out
        return None

    @staticmethod
    def _parse_score(text: Optional[str]) -> Optional[float]:
        if not text: return None
        s = text.strip()
        m = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
        if not m: return None
        try:
            val = float(m.group(1))
        except Exception: return None
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        return float(val)

    # --- 主调用入口 ---
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        completions: 模型的预测输出 list[str]
        solution: Ground Truth 列表 (即 Ref-CoT) list[str]
        kwargs: 包含 messages, time_interval, event 等
        """
        rewards: List[float] = []
        messages = kwargs.get("messages")
        
        # 从 kwargs 获取 batch 数据
        # 注意：这里假设 kwargs 中的 value 都是与 completions 长度对齐的列表
        batch_time_intervals = kwargs.get("time_interval", [])
        batch_events = kwargs.get("event", [])
        
        # 安全性检查：如果没有传递这些列，设为空列表
        if not batch_time_intervals: batch_time_intervals = [[]] * len(completions)
        if not batch_events: batch_events = [[]] * len(completions)

        for i, (pred_text, ref_cot_text) in enumerate(zip(completions, solution)):
            try:
                # 1. 获取并清洗 Question
                question_text = self._get_question_text(messages, i)
                if question_text:
                    # 去掉 <audio> 标记
                    question_text = question_text.replace("<audio>", "").strip()
                else:
                    rewards.append(0.0)
                    continue

                # 2. 获取 perception
                cot_text = (str(pred_text) if pred_text is not None else "").strip()
                if not cot_text:
                    rewards.append(0.0)
                    continue
                cot_perception = self._extract_perception_content(cot_text)
                
                # Correct Answer 
                answer_text = ref_cot_text
                
                # Audio Caption (拼接 time_interval 和 event)
                current_time_intervals = batch_time_intervals[i] if i < len(batch_time_intervals) else []
                current_events = batch_events[i] if i < len(batch_events) else []
                caption_text = self._build_caption(current_time_intervals, current_events)

                # 4. 构建 Prompt
                prompt = self._build_prompt(
                    caption_text=caption_text,
                    question_text=question_text,
                    answer_text=answer_text,
                    cot_text=cot_perception
                )

                # 5. 调用打分模型
                # judge_out = self._call_judge_with_retry(prompt)
                
                # 为了防止 retry 耗时过长，可以先调一次，如果返回空再走 retry (原逻辑保留)
                # 这里直接复用你原有的逻辑
                judge_out = self._call_judge_once(prompt)
                if judge_out is None or judge_out.strip() == "":
                    judge_out = self._call_judge_with_retry(prompt)

                # 6. 解析分数
                score = self._parse_score(judge_out)
                rewards.append(score if score is not None else 0.0)

            except Exception as e:
                # 调试时可以 print(e)
                rewards.append(0.0)

        return rewards

# 注册
orms['perception_judge_vllm'] = PerceptionQualityJudgeORM

# R4 CoT 过程奖励
class CoTQualityJudgeORM: 
    """
    使用外部 vLLM 评审 CoT 推理质量。
    采用 [Step-level Geometric Mean] + [Holistic Score] 的混合奖励机制。
    此版本使用模型自身的 <perception> 输出作为事实依据进行打分。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        served_model: str = "Qwen3-32B",
        max_retries: int = 5,
        alpha: float = 0.7  # 子问题分数的权重
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.served_model = served_model
        self.max_retries = max_retries
        self.alpha = alpha

    # --- 辅助方法：文本提取与解析 ---
    
    @staticmethod
    def _extract_reasoning_content(output_text: str) -> str:
        """从 solution 中提取 <reasoning>...</reasoning>"""
        if not output_text: return "Unknown"
        match = re.search(r"<reasoning>(.*?)</reasoning>", output_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_perception_content(output_text: str) -> str:
        """从 solution 中提取 <perception>...</perception> 的内容"""
        if not output_text:
            return "Unknown"
        match = re.search(r"<perception>(.*?)</perception>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Unknown"

    @staticmethod
    def _extract_steps(reasoning_text: str) -> List[str]:
        """
        解析 '1. Sub-question: ... Answer: ...' 格式的步骤。
        """
        pattern = r"(\d+\.\s*Sub-question:.*?)(?=\n\d+\.\s*Sub-question:|$)"
        matches = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    # --- 核心：Prompt 构建 ---

    @staticmethod
    def _build_step_prompt(caption_text: str, question_text: str, history_text: str, current_step_text: str) -> str:
        """构建子问题级 Prompt (Step-Level)"""
        return (
            "You are an expert logic and reasoning evaluator for Audio-LLMs. I will give you a record containing: \n"
            "1. A Detailed Audio Caption (Model Perception): A comprehensive text description of the audio events.\n"
            "2. A User Question and Context: The goal of the reasoning.\n"
            "3. A Reasoning History: The steps taken so far.\n"
            "4. The CURRENT STEP: The specific sub-question or reasoning step to evaluate now.\n\n"

            "Your task is to evaluate the **validity, necessity, and audio-grounding** of the CURRENT STEP only, and output a single numeric score from {0, 0.1, 0.2, ..., 1.0}. "
            "You must output only the score with no explanation or extra text.\n\n"

            "Evaluate based on the following Micro-Level dimensions:\n\n"

            "--- CRITERIA: Local Quality Check ---\n"
            "1. Usefulness: Is this specific reasoning step useful for answering the main question?\n"
            "2. Evidence-Based Conclusion: Is the content of this step supported by the provided 'Audio Caption'?\n"
            "   - Every claim must align with specific events, sound sources, or acoustic details described in the caption.\n"
            "3. Criticality & Efficiency: Is this step a logical next move based on the [Reasoning History]?\n"
            "   - Penalize 'tangential reasoning' (analyzing irrelevant noise) or redundant repetition of previous steps.\n\n"

            "Scoring guideline: \n"
            "1.0 = Perfect. The step is firmly grounded in the caption, necessary, and logically follows the history.\n"
            "0.8-0.9 = Strong. Good step, but maybe slightly inefficient or the evidence citation is slightly vague.\n"
            "0.5-0.7 = Mediocre. Relevant, but weak grounding (making assumptions not explicitly in the caption). Logic holds but is messy.\n"
            "0.2-0.4 = Weak. The step makes a claim not supported by the caption, or merely repeats previous steps without adding value.\n"
            "0.0-0.1 = Failed. Completely incoherent, visual hallucination, or factual contradiction with the caption (e.g., claiming a sound exists when caption implies silence).\n\n"

            "Penalty guideline (Apply these cumulatively to reduce score):\n"
            "*** [CRITICAL PENALTY] (Set Score to 0.0 - 0.1) ***: \n"
            "   - 'Factual Contradiction': The step claims a specific sound or event occurs which is explicitly absent or contradicted by the Caption.\n\n"

            "Operational rule: Always output only one score (0-1). If [Current Step] is empty, return 0.0.\n\n"

            f"The Detailed Audio Caption is:\n{caption_text}\n\n"
            f"The User Question is:\n{question_text}\n\n"
            f"The Reasoning History is:\n{history_text}\n\n"
            f"The CURRENT STEP to evaluate is:\n{current_step_text}\n"
        )

    @staticmethod
    def _build_holistic_prompt(caption_text: str, question_text: str, answer_text: str, full_reasoning: str) -> str:
        """构建整体级 Prompt (Holistic)"""
        return (
        "You are an expert logic and reasoning evaluator for Audio-LLMs. I will give you a record containing: \n"
        "1. A Detailed Audio Caption (Model Perception): A comprehensive text description of the audio events.\n"
        "2. A Question and its Correct Answer.\n"
        "3. The COMPLETE Model Reasoning: The entire chain of thought generated by the model.\n\n"

        "Your task is to evaluate the **logical architecture, coherence, efficiency, and final derivability** of the entire process, and output a single numeric score from {0, 0.1, 0.2, ..., 1.0}. "
        "You must output only the score with no explanation or extra text.\n\n"

        "Evaluate based on the following Macro-Level dimensions:\n\n"

        "--- CRITERIA: Holistic Logical Architecture ---\n"
        "1. Goal-Orientation: Is the reasoning path linear and directed towards the [Correct Answer]? \n"
        "   - Penalize circular logic.\n"
        "2. Causal Dependency: Does Step B legitimately follow Step A? \n"
        "   - Penalize 'Logic Jumps' where a conclusion appears out of nowhere without a preceding premise defined in the audio caption.\n"
        "3. Error Propagation Check: Does an early error render the rest of the chain invalid?\n"
        "   - If Step 1 is wrong (e.g., misidentifying a gender or sound source compared to the caption), and subsequent steps rely on it, the whole chain collapses.\n"
        "4. Final Derivability: Does the reasoning naturally flow to the [Correct Answer]?\n"
        "   - The conclusion must be the inevitable result of the reasoning steps, not a sudden guess.\n"
        "5. Efficiency & Conciseness: Is the length of the reasoning proportional to the complexity of the question?\n"
        "   - **Penalize 'Over-Analysis'**: If the question is simple (e.g., 'Is there a dog?'), the reasoning should be short. Writing a 500-word essay for a simple question is a failure.\n"
        "   - **Penalize Repetition**: Check if the model repeats the same analysis in different words just to make the chain longer.\n\n"

        "Scoring guideline: \n"
        "1.0 = Perfect. Every step is necessary, concise, and the logic flows flawlessly from the caption evidence to the correct conclusion.\n"
        "0.8-0.9 = Strong. Good logic, but maybe slightly verbose or includes one unnecessary step, yet the path is valid.\n"
        "0.5-0.7 = Mediocre. Logic holds but is **bloated** or unfocused. Contains repetitive analysis or over-explains simple facts found in the caption.\n"
        "0.2-0.4 = Weak. Major Logic Jumps, or the reasoning is **excessively long** and tedious without adding value (Filibustering).\n"
        "0.0-0.1 = Failed. The reasoning contradicts the final answer, relies on 'Fatal Error Propagation', or is complete nonsense.\n\n"

        "Penalty guideline (Apply these cumulatively):\n"
        "*** [CRITICAL PENALTY] (Set Score to 0.0 - 0.2) ***: \n"
        "   - 'Fatal Error Propagation': Early false premise (contradicting the Audio Caption) corrupts the entire remaining chain.\n"
        "   - 'Contradiction': The reasoning concludes something different from the actual Correct Answer provided.\n\n"
        "*** [MODERATE PENALTY] (-0.2 ~ -0.4) ***: \n"
        "   - 'Bloated Reasoning': The reasoning is too long for the problem's difficulty (e.g., 10 steps for a Yes/No question).\n"
        "   - 'Irrelevance': Wasting steps on analyzing audio events that are present in the caption but do not help answer the specific Question.\n\n"

        "Operational rule: Always output only one score (0-1). \n\n"

        f"The Detailed Audio Caption is:\n{caption_text}\n\n"
        f"The Question is:\n{question_text}\n\n"
        f"The Correct Answer is:\n{answer_text}\n\n"
        f"The COMPLETE Model Reasoning is:\n{full_reasoning}\n"
    )

    # --- API 调用与解析 ---

    def _call_judge(self, prompt: str) -> float:
        for _ in range(max(1, self.max_retries)):
            try:
                resp = self.client.chat.completions.create(
                    model=self.served_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    temperature=0.1, 
                    max_tokens=10 
                )
                if resp.choices:
                    content = resp.choices[0].message.content
                    m = re.search(r"([+-]?\d+(?:\.\d+)?)", content)
                    if m:
                        val = float(m.group(1))
                        return max(0.0, min(1.0, val)) 
            except Exception:
                continue
        return 0.0 

    # --- 主调用入口 ---

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        batch_messages = kwargs.get("messages", [])
        
        # 注意：不再需要 batch_time_intervals 和 batch_events，因为我们用模型输出
        
        for i, (pred_text, ref_cot_text) in enumerate(zip(completions, solution)):
            try:
                # 1. 提取模型输出的完整文本
                full_output = str(pred_text) if pred_text is not None else ""
                
                # 2. 提取 Perception (作为新的 Fact source)
                # --- 修改点：这里改用提取模型输出 ---
                cot_perception = self._extract_perception_content(full_output)
                if not cot_perception or cot_perception == "Unknown":
                    # 如果没有感知内容，Reasoning 无法 Grounding，直接给 0 分
                    rewards.append(0.0)
                    continue

                # 3. 提取 Reasoning
                cot_reasoning = self._extract_reasoning_content(full_output)
                if not cot_reasoning:
                    rewards.append(0.0)
                    continue
                
                # 4. 获取问题文本 (简化逻辑，实际按需适配)
                question_text = "Unknown Question"
                if batch_messages and i < len(batch_messages):
                     question_text = str(batch_messages[i]) 
                
                # 5. 解析步骤 (Step Parsing)
                steps = self._extract_steps(cot_reasoning)
                if not steps:
                    rewards.append(0.0)
                    continue

                # --- 6. 局部奖励计算 (Step-level Reward) ---
                step_scores = []
                history_accum = ""
                
                for step in steps:
                    # 这里的 caption_text 传入的是 model perception
                    step_prompt = self._build_step_prompt(
                        caption_text=cot_perception,
                        question_text=question_text,
                        history_text=history_accum,
                        current_step_text=step
                    )
                    
                    s = self._call_judge(step_prompt)
                    
                    # 平滑处理
                    s_smooth = max(s, 0.01)
                    step_scores.append(s_smooth)
                    
                    history_accum += step + "\n"

                # 计算几何平均数
                geo_mean_score = 0.0
                if step_scores:
                    log_sum = sum(math.log(s) for s in step_scores)
                    geo_mean_score = math.exp(log_sum / len(step_scores))

                # --- 7. 全局奖励计算 (Holistic Reward) ---
                holistic_prompt = self._build_holistic_prompt(
                    caption_text=cot_perception, # 使用 model perception
                    question_text=question_text,
                    answer_text=ref_cot_text,
                    full_reasoning=cot_reasoning
                )
                holistic_score = self._call_judge(holistic_prompt)

                # --- 8. 最终加权融合 ---
                final_reward = self.alpha * geo_mean_score + (1.0 - self.alpha) * holistic_score
                rewards.append(final_reward)

            except Exception as e:
                # print(f"Error in reward calculation: {e}")
                rewards.append(0.0)

        return rewards

# 注册
orms['cot_quality_judge_vllm'] = CoTQualityJudgeORM

# 内置review奖励 与答案奖励叠加
class ReviewScorer:
    """
    辅助工具类：负责计算 Review 部分的质量分数 (0.0 - 1.0)。
    不直接作为 ORM 使用，而是被其他 ORM 调用。
    """
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        served_model: str = "Qwen3-32B",
        max_retries: int = 3  # 这里可以稍微减少重试次数以加快速度
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.served_model = served_model
        self.max_retries = max_retries

    # --- 静态解析方法 (直接复用原逻辑) ---
    @staticmethod
    def _extract_perception_content(output_text: str) -> str:
        if not output_text: return "Unknown"
        match = re.search(r"<perception>(.*?)</perception>", output_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"

    @staticmethod
    def _extract_reasoning_content(output_text: str) -> str:
        if not output_text: return "Unknown"
        match = re.search(r"<reasoning>(.*?)</reasoning>", output_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"

    @staticmethod
    def _extract_review_content(output_text: str) -> str:
        if not output_text: return "" # 如果没有review，返回空字符串
        match = re.search(r"<review>(.*?)</review>", output_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _build_caption(time_intervals: list, events: list) -> str:
        if not time_intervals or not events or len(time_intervals) != len(events):
            return "No audio events detected."
        try:
            lines = []
            for t, ev in zip(time_intervals, events):
                # 兼容 list 或 tuple 的时间戳
                if isinstance(t, (list, tuple)) and len(t) >= 2:
                    s, e = t[0], t[1]
                else:
                    s, e = 0.0, 0.0
                lines.append(f"[{s}, {e}]: {ev}")
            return " ".join(lines)
        except Exception:
            return "Error parsing audio events."

    @staticmethod
    def _parse_score(text: Optional[str]) -> float:
        if not text: return 0.0
        m = re.search(r"([+-]?\d+(?:\.\d+)?)", text.strip())
        if not m: return 0.0
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except: return 0.0

    # --- Prompt 构建 ---
    def _build_prompt(self, caption_text, ground_truth_text, question_text, answer_text, reasoning_text, review_text) -> str:
        # (完全复用你原来的 Prompt，这里省略具体长文本以节省空间)
        return  f"""
            You are a critical meta-evaluator for the "Self-Correction" (Review) phase of an Audio-LLM. 
            Your task is to judge whether the [Review Content] effectively audits, verifies, and corrects the [Model Reasoning].

            Input Data:
            1. [Detailed Audio Caption] (Model Perception): {caption_text}
            2. [Ground Truth Annotations] (Fact Reference): {ground_truth_text}
            3. [User Question]: {question_text}
            4. [Correct Answer] (Ground Truth): {answer_text}
            5. [Model Reasoning] (Target to Audit): {reasoning_text}
            6. [Review Content] (The Audit Output): {review_text}

            Task: Output a single score {{0.0, 0.1, ..., 1.0}} for the [Review Content].

            Evaluation Criteria (Review Quality):

            1. Evidence Verification (Content Alignment):
            - Does the Review explicitly verify that every event cited in the [Model Reasoning] actually exists in the [Detailed Audio Caption]?
            - Did it catch "Hallucinations" where the Reasoning cites a sound (e.g., "dog barking") that is completely absent from the Caption?
            - **CRITERIA**: The Review must confirm that the *evidence* used in reasoning is physically present in the text description.

            2. Temporal & Causal Logic Audit:
            - Does the Review check the narrative sequence? (e.g., if Reasoning says "A causes B", did the Review check if the Caption describes A happening before or leading into B?)
            - Did it catch chronological errors where the Reasoning flips the order of events described in the text?

            3. Logical Integrity Check:
            - Does the Review ensure the conclusion is strictly derived from the perceived evidence?
            - Did it flag any "Logic Jumps" (conclusions without premises) in the Reasoning?

            4. Genuine Error Correction & Answer Verification (Anti-Rubber-Stamping):
            - **ANSWER CHECK (CRITICAL)**: Compare the conclusion/answer in [Model Reasoning] with the [Correct Answer]. 
                - If Reasoning leads to a WRONG answer, the Review **MUST** detect and flag this failure.
                - If Reasoning leads to a CORRECT answer, the Review must validate the logic flow.
            - **HALLUCINATION CHECK**: If [Model Reasoning] contains factual errors (contradictions with Caption), the Review **MUST** point them out.

            - **PENALTY RULES**:
                - **Score 0.0 IMMEDIATELY** if the [Model Reasoning] concludes with a wrong answer (mismatch with [Correct Answer]) but the Review states "The reasoning is correct" or "The answer is valid".
                - **Score 0.0 IMMEDIATELY** if the Review approves reasoning that contradicts the Audio Caption (e.g., approving a sound that isn't in the description).

            Scoring Guide:
            - 1.0: Perfect Audit. The review rigorously checked evidence presence, sequential logic, AND accurately validated the final answer against Ground Truth.
            - 0.8-0.9: Strong. Good check, caught major issues, but maybe missed a minor detail in the description.
            - 0.5-0.7: Generic Validation. Says "Logic is good" without citing specific text evidence. (Rubber-stamping).
            - 0.2-0.4: Weak. Fails to catch obvious hallucinations or logic jumps.
            - 0.0-0.1: [FATAL] The Review acts as a "Yes-Man" (Rubber-Stamp) for an INCORRECT Answer. It approves a reasoning path that leads to a result different from the [Correct Answer].

            Output Rule: Return ONLY the numeric score.
            """

    # --- 核心打分逻辑 ---
    def calculate_review_score(
        self, 
        full_cot_text: str, 
        question_text: str, 
        answer_text: str, 
        gt_time_intervals: list, 
        gt_events: list
    ) -> float:
        """
        计算单个样本的 Review 分数。
        如果缺少必要组件（如没有review标签），返回 0.0。
        """
        try:
            # 1. 提取各个部分
            perception = self._extract_perception_content(full_cot_text)
            reasoning = self._extract_reasoning_content(full_cot_text)
            review = self._extract_review_content(full_cot_text)
            
            # 如果没有 review 内容，直接给 0 分
            if not review:
                return 0.0

            # 2. 构建 GT Caption
            gt_caption = self._build_caption(gt_time_intervals, gt_events)

            # 3. 构建 Prompt
            prompt = self._build_prompt(
                caption_text=perception,
                ground_truth_text=gt_caption,
                question_text=question_text,
                answer_text=answer_text,
                reasoning_text=reasoning,
                review_text=review
            )

            # 4. 调用 API (带简单重试)
            judge_out = None
            for _ in range(self.max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.served_model,
                        messages=[
                            {"role": "system", "content": "You are an expert reasoning evaluator."},
                            {"role": "user", "content": prompt},
                        ],
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    if resp and resp.choices:
                        judge_out = resp.choices[0].message.content
                        break
                except Exception:
                    continue
            
            return self._parse_score(judge_out)

        except Exception as e:
            print(f"ReviewScorer Error: {e}")
            return 0.0