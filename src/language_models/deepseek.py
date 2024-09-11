import json
import os
from time import sleep

import openai
from openai import OpenAI
from openai._types import NotGiven

from .base import LanguageModel

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SLEEP_SEC = 3


class DeepSeek(LanguageModel):
    def __init__(
        self, model: str = "deepseek_chat", api_key: str = None, *args, **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        if api_key is None:
            api_key = open("deepseek_api_key.txt").read().strip()
        self.client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=api_key)

    def chat(self, messages: str, system_msg: str = None, **kwargs):
        try:
            response = self._chat(messages, system_msg, **kwargs)
            return response
        except openai.BadRequestError as e:
            err = json.loads(e.response.text)
            if err["error"]["code"] == "content_filter":
                print("Content filter triggered!")
                return None
            print(f"The OpenAI API request was invalid: {e}")
            return None
        except openai.APIConnectionError as e:
            print(f"The OpenAI API connection failed: {e}")
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except openai.RateLimitError as e:
            print(f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second...")
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except openai.AuthenticationError as e:
            print(f"Invalid API token: {e}")
            self.update_api_key()
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                print("The OpenAI API request timed out. Please try again later.")
                sleep(SLEEP_SEC)
                return self.chat(messages, system_msg, **kwargs)
            elif "DeploymentNotFound" in str(e):
                print("The API deployment for this resource does not exist")
                return None
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                sleep(SLEEP_SEC)
                return self.chat(messages, system_msg, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")

    def _chat(
        self,
        messages: str,
        system_msg="",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        json_mode: bool = False,
    ):
        if system_msg is None or system_msg == "":
            system_msg = "You are a helpful assistant."
        msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": messages},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"} if json_mode else NotGiven(),
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    aoai = DeepSeek(model="deepseek-chat")
    print(aoai.chat("Hello, who are you?", json_mode=True))
