import os
import dotenv

dotenv.load_dotenv()

MODELS = {
    "gemma_9B_it": "google/gemma-2-9b-it",
    "deepseek_qwen_1_5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen_1_5B_it": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "gemma_9B": "google/gemma-2-9b",
    "llama_3_8B": "meta-llama/Meta-Llama-3-8B",
    "llama_3_8B_it": "meta-llama/Meta-Llama-3-8B-Instruct",
}

GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"  # "gemini-2.5-pro-exp-03-25"  # "gemini-2.0-flash-thinking-exp-01-21"

API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"
API_BASE_MODEL = "deepseek-v3-241226"

N_SAMPLES = 20
SAMPLING_MODEL_ID = "gemma_9B_it"

LLM_MODEL_PROVIDER = "deepseek"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
