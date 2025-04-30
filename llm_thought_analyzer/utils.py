import json
import time
from . import config
from .logger import LLMLogger, FunctionLogger

llm_logger = LLMLogger()
function_logger = FunctionLogger()


def call_llm(
    user_prompt: str,
    system_prompt: str = None,
    response_format: str = "json_object",
    model_provider: str = "deepseek",
) -> str:
    start_time = time.time()
    function_logger.log_step(
        "LLM_CALL",
        f"Calling {model_provider} with prompt: {user_prompt}",
        "INFO",
    )

    if model_provider == "deepseek":
        from openai import OpenAI

        client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
        )

        response = client.chat.completions.create(
            model=config.API_BASE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": response_format},
        )

        response_text = response.choices[0].message.content
        # Log the call
        duration_ms = int((time.time() - start_time) * 1000)
        llm_logger.log_call(
            model_provider=model_provider,
            model_name=config.API_BASE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text,
            duration_ms=duration_ms,
        )

        function_logger.log_step(
            "LLM_RESPONSE",
            f"Received response from {model_provider} in {duration_ms}ms. Response: {response_text}",
            "INFO",
        )

        # Validate JSON response if required
        if response_format == "json_object":
            try:
                json.loads(response_text)
            except Exception as e:
                # raise ValueError(f"Invalid JSON response: {str(e)}")
                response_text = (
                    response_text.strip("```json").strip("```python").strip("```")
                )

        return response_text

    elif model_provider == "gemini":
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                system_prompt,
                user_prompt,
            ],
        )

        response_text = response.text

        # Log the call
        duration_ms = int((time.time() - start_time) * 1000)
        llm_logger.log_call(
            model_provider=model_provider,
            model_name=config.GEMINI_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response_text,
            duration_ms=duration_ms,
        )

        function_logger.log_step(
            "LLM_RESPONSE",
            f"Received response from {model_provider} in {duration_ms}ms. Response: {response_text}",
            "INFO",
        )

        # Validate JSON response if required
        if response_format == "json_object":
            try:
                json.loads(response_text)
            except json.JSONDecodeError as e:
                # raise ValueError(f"Invalid JSON response: {str(e)}")
                response_text = (
                    response_text.strip("```json").strip("```python").strip("```")
                )

        return response_text


def load_few_shot_examples(file_path="data/gsm_few_shot.json"):
    """Load few-shot examples from the JSON file."""
    function_logger.log_step(
        "LOAD_EXAMPLES", f"Loading few-shot examples from {file_path}", "INFO"
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        function_logger.log_step(
            "LOAD_EXAMPLES", f"Successfully loaded {len(examples)} examples", "INFO"
        )
        return examples
    except Exception as e:
        function_logger.log_step(
            "LOAD_EXAMPLES", f"Error loading examples: {str(e)}", "ERROR"
        )
        raise


def format_prompt(new_question, few_shot_examples=None):
    """Format the prompt with few-shot examples and the new question."""
    function_logger.log_step("FORMAT_PROMPT", "Starting prompt formatting", "INFO")

    if few_shot_examples is None:
        few_shot_examples = load_few_shot_examples()

    prompt = ""
    # Add shot examples
    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"

    # Add the new question
    prompt += f"Question: {new_question}\n"
    prompt += "Answer: Let's think step by step."

    function_logger.log_step("FORMAT_PROMPT", "Successfully formatted prompt", "INFO")

    return prompt
