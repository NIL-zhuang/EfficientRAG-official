from typing import Union


class LanguageModel:
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def chat(
        self,
        messages: Union[str, list[str]],
        system_msg: str,
        json_mode: bool,
        **kwargs
    ):
        raise NotImplementedError

    def complete(self, prompts: str):
        raise NotImplementedError
