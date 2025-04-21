import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, prompt_length):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        if (
            input_ids.shape[1] <= self.prompt_length
        ):  # Skip if we haven't generated beyond prompt
            return False

        # Only decode the newly generated text (everything after the prompt)
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length :])
        return any(keyword in generated_text for keyword in self.keywords)


def format_prompt(shot_examples, new_question):
    """Format the prompt with few-shot examples and the new question."""
    prompt = ""
    # Add shot examples
    for example in shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"

    # Add the new question
    prompt += f"Question: {new_question}\n"
    prompt += "Answer: Let's think step by step."
    return prompt


@torch.no_grad()
def generate_multiple_sequences(
    model_name,
    prompt,
    num_return_sequences,
    strategy="sampling",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add attention mask to inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_ids = inputs["input_ids"].to(model.device)

    if strategy == "beam":
        # Memory-efficient beam search
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=3,  # Reduced from 5 to save memory
            num_return_sequences=num_return_sequences,  # Reduced from 3 to save memory
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,  # Enable KV-caching for memory efficiency
            stopping_criteria=[
                StoppingCriteriaList(
                    [
                        KeywordStoppingCriteria(
                            ["Question:"],
                            tokenizer,
                            prompt_length=input_ids.shape[1],
                        )
                    ]
                )
            ],
        )
    elif strategy == "sampling":
        # Memory-efficient sampling
        answers = []
        for i in range(num_return_sequences):
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,  # Generate one sequence at a time
                temperature=0.7,
                top_p=0.9,
                use_cache=True,  # Enable KV-caching for memory efficiency
                stopping_criteria=[
                    StoppingCriteriaList(
                        [
                            KeywordStoppingCriteria(
                                ["Question:"],
                                tokenizer,
                                prompt_length=input_ids.shape[1],
                            )
                        ]
                    )
                ],
            )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = response.split("Answer:")[-1].strip()
            if answer.endswith("Question:"):
                answer = answer[: -len("Question:")].strip()
            answers.append(answer)

            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return answers
