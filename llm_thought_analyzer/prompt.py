TEMPLATIZED_QUESTION_FROM_QUESTION_SYSTEM_PROMPT = """
You are a helpful assistant that converts math word problems into templatized questions and factual assignments. NO ```json in the beginning and end of the JSON object.
"""

TEMPLATIZED_QUESTION_FROM_QUESTION_USER_PROMPT = """
Your task is to:

1. Identify AS MANY numerical values AS POSSIBLE in the question
2. Replace EACH numerical value with a descriptive placeholder in the format {{variable_name}}
3. Create a factual assignment dictionary that maps each placeholder to its original value
4. Ensure the templatized question maintains the same structure and meaning as the original
5. Use clear, descriptive variable names that reflect the meaning of the numbers

The output should be a valid JSON object with two fields:
- "templatized_question": The question with numerical values replaced by placeholders
- "factual_assignment": A dictionary mapping placeholder names to their original values

Example:
Input: "A train travels 60 kilometers in 2 hours. What is its average speed?"
Output: {{
    "templatized_question": "A train travels {{distance}} kilometers in {{time}} hours. What is its average speed?",
    "factual_assignment": {{
        "distance": 60,
        "time": 2
    }}
}}

Please convert the following math word problem into a templatized question and factual assignment:

<question>
{question}
</question>

Use this JSON schema:

TemplatizedQuestion = {{'templatized_question': str, 'factual_assignment': dict[str, float]}}
Return: TemplatizedQuestion

Return only a valid JSON object with "templatized_question" and "factual_assignment" fields, NO markdown formatting, NO ```json in the beginning and end of the JSON object.
"""


GROUND_TRUTH_FUNCTION_FROM_REASONING_SYSTEM_PROMPT = """
You are a helpful assistant that converts mathematical reasoning processes into Python functions. NO markdown formatting or ```python in the beginning and end of the function.
"""

GROUND_TRUTH_FUNCTION_FROM_REASONING_USER_PROMPT = """
Convert the following reasoning process into a Python function. The function should implement the exact same calculations as described in the reasoning.

Templatized Question:
{question}

Factual Assignment:
{factual_assignment}

Reasoning Process:
{reasoning_process}

REQUIREMENTS:

1. Please DIRECTLY provide only a valid Python function. Your function should faithfully follow the reasoning process. You MUST NOT solve the problem or introduce new deductions in the function.
2. The input parameters should be the keys in the factual assignment
3. Create the intermediate variables as demonstrated in the reasoning process. The amount of calculation in each step should follow the reasoning process.
4. You do NOT need to write comments or docstrings.

Make sure the function is valid Python code. The function MUST be named solution. NO ```python in the beginning and end of the function.

def solution(arg1, arg2, arg3):
    intermediate_variable1 = arg1 + arg3
    intermediate_variable2 = intermediate_variable1 * arg2
    answer = intermediate_variable2 * arg1 +intermediate_variable1
    return answer
"""


NEW_FUNCTION_FROM_REASONING_SYSTEM_PROMPT = """
You are a helpful assistant that creates a new Python function based on the reasoning process, considering the original function, the templatized question and the factual assignment. NO markdown formatting or ```python in the beginning and end of the function.
"""

NEW_FUNCTION_FROM_REASONING_USER_PROMPT = """
Given the following Python function and a new set of arguments, create a variation of this function that produces the result for the new reasoning process.

REQUIREMENTS:
1. Your function MUST faithfully follow the new reasoning process.
2. Keep the same input parameters as the original function.
3. Keep the same intermediate variables names. For the calculation, you should follow the new reasoning process.
4. You MUST NOT solve the problem or introduce new deductions in the function.

Original function:
{original_function}

Templatized question:
{question}

Factual assignment:
{factual_assignment}

New reasoning process:
{reasoning_process}

Please provide only the modified Python function, no explanations, no markdown formatting. NO ```python in the beginning and end of the function.
"""
