from openai import OpenAI

from .base import LanguageModel

LLAMA_ENDPOINT = "USE YOUR OWN VLLM ENDPOINT"
LLAMA_API_KEY = "USE YOUR OWN VLLM API KEY"


class LlamaServer(LanguageModel):
    def __init__(self, model: str = "Meta-Llama-3-70B-Instruct", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.client = OpenAI(
            base_url=LLAMA_ENDPOINT,
            api_key=LLAMA_API_KEY,
        )

    def chat(self, message: str, system_msg: str = None, json_mode: bool = False):
        if system_msg is None:
            system_msg = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        response = response.choices[0].message.content
        return response

    def complete(self, prompts: str):
        response = self.client.completions.create(
            model=self.model, prompt=prompts, echo=False, max_tokens=100
        )
        response = response.choices[0].text
        return response


if __name__ == "__main__":
    llama = LlamaServer("Meta-Llama-3-8B-Instruct")
    response = llama.complete(
        "The reason of human landing on moon is that, some one found it strange behind the moon."
    )
    print(response)
