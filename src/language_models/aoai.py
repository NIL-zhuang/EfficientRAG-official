import json
import os
from time import sleep
from typing import Optional, Union

import numpy as np
import openai
from openai import AzureOpenAI
from openai._types import NotGiven

from .base import LanguageModel
from .cloudgpt import auto_refresh_token, cloudGPT_available_models, get_openai_token

AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT", None)
if AOAI_ENDPOINT is None:
    AOAI_ENDPOINT = "https://cloudgpt-openai.azure-api.net/"

SLEEP_SEC = 3


class AOAI(LanguageModel):
    def __init__(
        self,
        model: Union[str, cloudGPT_available_models] = "gpt-4-0125-preview",
        embedding_model: Optional[str] = "text-embedding-ada-002",
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(model)
        self.api_version = api_version
        self.client = AzureOpenAI(
            azure_endpoint=AOAI_ENDPOINT,
            api_key=get_openai_token(),
            api_version=self.api_version,
        )
        self.embedding_model = embedding_model
        auto_refresh_token()

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

    def update_api_key(self):
        self.client = AzureOpenAI(
            azure_endpoint=AOAI_ENDPOINT,
            api_key=get_openai_token(),
            api_version=self.api_version,
        )

    def _embed(self, query: Union[str, list[str]]):
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_model,
        )
        result = np.array([response.data[idx].embedding for idx in range(len(response.data))])
        if isinstance(query, str):
            return result[0]
        return result

    def embed(self, query: Union[str, list[str]], **kwargs):
        try:
            response = self._embed(query, **kwargs)
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
            return self.embed(query, **kwargs)
        except openai.RateLimitError as e:
            print(f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second...")
            sleep(SLEEP_SEC)
            return self.embed(query, **kwargs)
        except openai.AuthenticationError as e:
            print(f"Invalid API token: {e}")
            self.update_api_key()
            sleep(SLEEP_SEC)
            return self.embed(query, **kwargs)
        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                print("The OpenAI API request timed out. Please try again later.")
                sleep(SLEEP_SEC)
                return self.embed(query, **kwargs)
            elif "DeploymentNotFound" in str(e):
                print("The API deployment for this resource does not exist")
                return None
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                sleep(SLEEP_SEC)
                return self.embed(query, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    aoai = AOAI(model="gpt-4-1106-preview")
    print(aoai.chat("Hello, how are you?", json_mode=False))
