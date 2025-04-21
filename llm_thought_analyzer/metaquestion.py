import json
import re
from typing import Any, Callable, Dict, List, Tuple, Union
from llm_thought_analyzer.utils import call_llm
from llm_thought_analyzer.prompt import (
    TEMPLATIZED_QUESTION_FROM_QUESTION_SYSTEM_PROMPT,
    TEMPLATIZED_QUESTION_FROM_QUESTION_USER_PROMPT,
    GROUND_TRUTH_FUNCTION_FROM_REASONING_SYSTEM_PROMPT,
    GROUND_TRUTH_FUNCTION_FROM_REASONING_USER_PROMPT,
    NEW_FUNCTION_FROM_REASONING_SYSTEM_PROMPT,
    NEW_FUNCTION_FROM_REASONING_USER_PROMPT,
)


class FunctionReasoning:
    def __init__(self, function_str: str, source: dict[str, Any]):
        self.function_str = function_str
        self.source = source
        self.function = self._compile_function()

    def _compile_function(self) -> Callable:
        namespace = {}
        exec(self.function_str, namespace)
        return namespace["solution"]

    def evaluate(self, **kwargs) -> float:
        return self.function(**kwargs)

    def __repr__(self) -> str:
        return self.function_str

    def to_dict(self):
        return {
            "function_str": self.function_str,
            "source": self.source,
        }


def extract_answer(reasoning_process: str) -> float:
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    FLOAT_RE = re.compile(r"(\-?[0-9]+\.?[0-9]*)")
    INVALID_ANS = "[invalid]"

    reasoning_process = reasoning_process.replace("$", "")
    match = ANS_RE.search(reasoning_process)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return float(match_str)

    float_matches = list(FLOAT_RE.finditer(reasoning_process))
    if float_matches:
        last_match = float_matches[-1]
        return float(last_match.group(1).strip())

    return INVALID_ANS


class GroundTruthQuestionAnswer:
    """
    A GroundTruthQuestionAnswer is conceptualized as a templated class for a question and its ground truth answer. For
    any math word problem, we can build the question template by replacing the numerical
    values with placeholders. Also, we can generate the python code of the ground truth
    answer.

    In particular, suppose we have a question

    <question>A train travels 60 kilometers in 2 hours. What is its average speed?</question>

    First, we build the templatized question using LLM, which will return results in the following format:

    {
        "templatized_question": "A train travels {{distance}} kilometers in {{time}} hours. What is its average speed?",
        "factual_assignment": {
            "distance": 60,
            "time": 2,
        },
    }

    Next, to get the ground truth python code of the reasoning process, we can use LLM to
    generate the python code from the templatized question and the factual assignment.

    <templatized_question>A train travels {{distance}} kilometers in {{time}} hours. What is its average speed?</templatized_question>
    <factual_assignment>
        distance: 60,
        time: 2,
    </factual_assignment>
    <reasoning_process>
        The average speed is the total distance divided by the total time.
    </reasoning_process>

    The returned result is

    def solution(distance, time): return distance / time
    """

    def __init__(
        self,
        templatized_question: str,
        function_reasoning: FunctionReasoning,
    ) -> None:
        self.templatized_question = templatized_question
        self.function_reasoning = function_reasoning

    @classmethod
    def from_question_answer(
        cls,
        question: str,
        reasoning_process: str,
        model_provider: str,
        return_factual_assignment: bool = True,
    ) -> Union[
        "GroundTruthQuestionAnswer",
        Tuple["GroundTruthQuestionAnswer", Dict[str, float]],
    ]:
        # first, build the templatized question
        response = call_llm(
            system_prompt=TEMPLATIZED_QUESTION_FROM_QUESTION_SYSTEM_PROMPT,
            user_prompt=TEMPLATIZED_QUESTION_FROM_QUESTION_USER_PROMPT.format(
                question=question
            ),
            model_provider=model_provider,
        )
        response = json.loads(response)
        templatized_question = response["templatized_question"]
        factual_assignment = response["factual_assignment"]

        # second, build the function reasoning
        response = call_llm(
            system_prompt=GROUND_TRUTH_FUNCTION_FROM_REASONING_SYSTEM_PROMPT,
            user_prompt=GROUND_TRUTH_FUNCTION_FROM_REASONING_USER_PROMPT.format(
                question=templatized_question,
                factual_assignment=factual_assignment,
                reasoning_process=reasoning_process,
            ),
            response_format="str",
            model_provider=model_provider,
        )
        function_reasoning = FunctionReasoning(
            response,
            source={
                "factual_assignment": factual_assignment,
                "reasoning_process": reasoning_process,
                "function_results_match_reasoning_process": True,
            },
        )
        expected_result = extract_answer(reasoning_process)
        if expected_result is not None:
            assert factual_assignment is not None, "Factual assignment is required"
            eval_result = function_reasoning.evaluate(**factual_assignment)
            assert (
                eval_result == expected_result
            ), "Expected result does not match the function output"

        if return_factual_assignment:
            return (
                GroundTruthQuestionAnswer(templatized_question, function_reasoning),
                factual_assignment,
            )

        return GroundTruthQuestionAnswer(templatized_question, function_reasoning)

    def evaluate(self, **kwargs) -> float:
        return self.function_reasoning.evaluate(**kwargs)

    def generate_new_function_reasoning(
        self,
        reasoning_process: str,
        factual_assignment: Dict[str, float],
        model_provider: str,
    ) -> FunctionReasoning:
        expected_result = extract_answer(reasoning_process)
        assert factual_assignment is not None, "Factual assignment is required"
        assert (
            self.evaluate(**factual_assignment) != expected_result
        ), "Answer matches the ground truth"

        response = call_llm(
            system_prompt=NEW_FUNCTION_FROM_REASONING_SYSTEM_PROMPT,
            user_prompt=NEW_FUNCTION_FROM_REASONING_USER_PROMPT.format(
                original_function=self.function_reasoning.function_str,
                question=self.templatized_question,
                reasoning_process=reasoning_process,
                factual_assignment=factual_assignment,
            ),
            response_format="str",
            model_provider=model_provider,
        )
        function_reasoning = FunctionReasoning(
            function_str=response,
            source={
                "factual_assignment": factual_assignment,
                "reasoning_process": reasoning_process,
            },
        )
        if expected_result is not None:
            eval_result = function_reasoning.evaluate(**factual_assignment)
            function_reasoning.source["function_results_match_reasoning_process"] = (
                eval_result == expected_result
            )

        return function_reasoning


class FunctionReasoningPool:
    """
    Various instances of a MetaQuestion can be generated by varying the numerical
    values, and model can give rather different reasoning processes.

    FunctionReasoningPool is a collection of these reasoning processes, parsed in
    python code.
    """

    def __init__(
        self,
        ground_truth_question_answer: GroundTruthQuestionAnswer,
        model_provider: str,
    ):
        self.ground_truth_question_answer = ground_truth_question_answer
        self.model_provider = model_provider
        self.function_reasoning_pool: List[FunctionReasoning] = [
            ground_truth_question_answer.function_reasoning
        ]

    def add(
        self,
        factual_assignment: Dict[str, float],
        reasoning_process: str,
    ) -> Tuple[FunctionReasoning, bool]:
        match = self.find_match(factual_assignment, reasoning_process)
        if match is not None:
            return match, False

        new_function = (
            self.ground_truth_question_answer.generate_new_function_reasoning(
                reasoning_process=reasoning_process,
                factual_assignment=factual_assignment,
                model_provider=self.model_provider,
            )
        )
        self.function_reasoning_pool.append(new_function)
        return new_function, True

    def find_match(
        self,
        factual_assignment: Dict[str, float],
        reasoning_process: str,
        tolerance: float = 1e-6,
    ) -> FunctionReasoning:
        expected_result = extract_answer(reasoning_process)
        for function_reasoning in self.function_reasoning_pool:
            try:
                if factual_assignment == function_reasoning.source[
                    "factual_assignment"
                ] and expected_result == extract_answer(
                    function_reasoning.source["reasoning_process"]
                ):
                    return function_reasoning

                result = function_reasoning.evaluate(**factual_assignment)
                if abs(result - expected_result) < tolerance:
                    return function_reasoning

            except Exception:
                continue
        return None
