"""
Convert the output of function reasoning to reasoning steps.
"""

import json
from llm_thought_analyzer.parser import ReasoningPath
import os

os.makedirs("data/test_output/converted", exist_ok=True)

TASKS = ["cleaner", "dice", "shrimp"]

for task in TASKS:
    with open(f"data/test_output/function_reasoning_pool_{task}.json", "r") as f:
        data = json.load(f)

    config = data["config"]
    ground_truth_function = data["ground_truth_function"]
    data["ground_truth_function"] = {
        "function_str": ground_truth_function,
        "reasoning_path_topological_levels": ReasoningPath.from_function_str(
            ground_truth_function
        )
        .simplify()
        .get_topological_levels(),
    }

    results = data["results"]

    new_results = []
    for result in results:
        new_result = result.copy()
        function_str = result["function"]["function_str"]
        reasoning_path = ReasoningPath.from_function_str(function_str)
        new_result["reasoning_path_topological_levels"] = (
            reasoning_path.simplify().get_topological_levels()
        )
        del new_result["reasoning_path"]
        new_results.append(new_result)

    data["results"] = new_results
    with open(
        f"data/test_output/converted/function_reasoning_{task}_steps.json", "w"
    ) as f:
        json.dump(data, f, indent=4)
