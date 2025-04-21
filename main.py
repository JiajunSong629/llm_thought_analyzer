import json
import argparse
from datetime import datetime
from llm_thought_analyzer import ReasoningPath
from llm_thought_analyzer import (
    GroundTruthQuestionAnswer,
    FunctionReasoningPool,
)
from llm_thought_analyzer.logger import FunctionLogger
from llm_thought_analyzer.config import (
    LLM_MODEL_PROVIDER,
    SAMPLING_MODEL_ID,
    N_SAMPLES,
    MODELS,
)


def run(
    question,
    reasoning_process,
    responses_under_randomizations=None,
    model_provider=LLM_MODEL_PROVIDER,
    output_file=None,
):
    parser = argparse.ArgumentParser(description="Run the thought entropy visualizer")
    parser.add_argument(
        "--log-file", type=str, help="Custom log file name (e.g., my_log.log)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to store log files"
    )
    args = parser.parse_args()

    function_logger = FunctionLogger(log_dir=args.log_dir, log_file=args.log_file)
    function_logger.log_step("MAIN_START", "Starting main execution")

    ground_truth_question_answer, factual_assignment = (
        GroundTruthQuestionAnswer.from_question_answer(
            question=question,
            reasoning_process=reasoning_process,
            model_provider=model_provider,
        )
    )

    function_logger.log_step("GROUND_TRUTH", "Created ground truth question answer")
    function_logger.log_step(
        "GROUND_TRUTH_TEMPLATIZED_QUESTION",
        ground_truth_question_answer.templatized_question,
    )
    function_logger.log_step("GROUND_TRUTH_FACTUAL_ASSIGNMENT", factual_assignment)

    pool = FunctionReasoningPool(
        ground_truth_question_answer=ground_truth_question_answer,
        model_provider=model_provider,
    )
    function_logger.log_step("POOL_CREATION", "Created function reasoning pool")

    if responses_under_randomizations is not None:
        function_logger.log_step(
            "LOAD_DATA",
            f"Loaded {len(responses_under_randomizations)} samples from sampling_responses.json",
        )
    else:
        function_logger.log_step(
            "LOAD_DATA",
            "Data is not provided. Generating responses under randomizations",
        )

        from llm_thought_analyzer.model import (
            generate_multiple_sequences,
            format_prompt,
        )

        responses_under_randomizations = generate_multiple_sequences(
            model_name=MODELS[SAMPLING_MODEL_ID],
            prompt=format_prompt(
                shot_examples=json.load(open("data/gsm_few_shot.json", "r")),
                new_question=question,
            ),
            num_return_sequences=N_SAMPLES,
            strategy="sampling",
        )
        with open(
            f"data/test_data/sampling_responses_{datetime.now().strftime('%d_%H%M%S')}.json",
            "w",
        ) as f:
            json.dump(responses_under_randomizations, f, indent=4)

        function_logger.log_step(
            "GENERATED_RESPONSES",
            f"Generated {len(responses_under_randomizations)} samples",
        )

    result = []
    for i, response in enumerate(responses_under_randomizations, 1):
        function_logger.log_step(
            "ADD_SAMPLE",
            f"Adding sample {i}/{len(responses_under_randomizations)} to pool",
        )
        function_logger.log_step("CURRENT_SAMPLE", response)
        function, added = pool.add(
            factual_assignment=factual_assignment,
            reasoning_process=response,
        )

        function_logger.log_step(
            "ADD_SAMPLE",
            f"Sample {i}/{len(responses_under_randomizations)} {'find a match' if added else 'is new'}",
        )

        function_logger.log_step("FUNCTION", f"{function}")

        path = ReasoningPath.from_function_str(function.function_str)
        result.append(
            {
                "sample_id": i,
                "added": added,
                "sample": response,
                "function": function.to_dict(),
                "reasoning_path": path.to_dict(),
            }
        )

    function_logger.log_step(
        "POOL_COMPLETE", f"Final pool size: {len(pool.function_reasoning_pool)}"
    )
    function_logger.log_step("MAIN_END", "Main execution completed")

    if output_file is None:
        output_file = f"data/test_data/function_reasoning_pool_{datetime.now().strftime('%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "config": {
                    "question": question,
                    "reasoning_process": (
                        reasoning_process if reasoning_process else "none"
                    ),
                    "model_provider": model_provider,
                    "sampling_model_id": SAMPLING_MODEL_ID,
                    "n_samples": N_SAMPLES,
                },
                "templatized_question": ground_truth_question_answer.templatized_question,
                "factual_assignment": factual_assignment,
                "ground_truth_function": ground_truth_question_answer.function_reasoning.function_str,
                "results": result,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def main():
    import os

    os.makedirs("data/test_output", exist_ok=True)
    question_reasoning = json.load(open("data/test_data/question_reasoning.json", "r"))
    for question_reasoning_process in question_reasoning:
        run(
            question=question_reasoning_process["question"],
            reasoning_process=question_reasoning_process["reasoning_process"],
            output_file=f"data/test_output/function_reasoning_pool_{question_reasoning_process['sample_id']}.json",
        )


if __name__ == "__main__":
    main()
