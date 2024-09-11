from .aoai import AOAI
from .base import LanguageModel
from .deepseek import DeepSeek
from .llama import LlamaServer

MODEL_DICT = {
    "gpt35": "gpt-35-turbo-1106",
    "gpt4": "gpt-4-0125-preview",
    "llama": "Meta-Llama-3-70B-Instruct",
    "llama-8B": "Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-chat",
}


def get_model(model_name: str, **kwargs) -> LanguageModel:
    if model_name in MODEL_DICT.keys():
        model_name = MODEL_DICT[model_name]

    if "gpt" in model_name.lower():
        return AOAI(model=model_name, **kwargs)
    elif "deepseek" in model_name.lower():
        return DeepSeek(model=model_name, **kwargs)
    elif "llama" in model_name.lower():
        return LlamaServer(model=model_name, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
