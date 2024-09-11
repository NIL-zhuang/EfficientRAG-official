import datetime
from typing import Literal


def get_openai_token(token_cache_file: str = "cloudgpt-apim-token-cache.bin") -> str:
    # REPLACE THIS WITH YOUR OWN CODE OR API TOKEN
    pass


cloudGPT_available_models = Literal[
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-35-turbo-1106",
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-35-turbo-0301",
    "gpt-35-turbo-0613",
    "gpt-35-turbo-16k-0613",
]


def auto_refresh_token(
    token_cache_file: str = "cloudgpt-apim-token-cache.bin",
    interval: datetime.timedelta = datetime.timedelta(minutes=15),
    on_token_update: callable = None,
) -> callable:
    # REPLACE THIS WITH YOUR OWN CODE OR API TOKEN
    pass
