import json
import os
import re
import sys
from typing import Literal

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from language_models import AOAI

QUERY_TYPE = Literal["extremely long-tail", "long tail", "common"]
# QUERY_LENGTH = Literal['less than 5 words', '5 to 15 words', 'at least 10 words']
DIFFICULTY = Literal["children", "high school", "college", "PhD"]
CLARITY = Literal["clear", "understandable with some effort", "ambiguous"]
DOC_WORDS = Literal["50", "100", "200", "300", "400", "500"]
HOP = Literal["single-hop", "2-hop", "3-hop", "4-hop"]
HOP_TYPE = Literal["compositional", "comparison", "bridge comparison", "inference"]

# Task generation system prompt
TASK_GENERATION_SYSTEM_PROMPT = """Brainstorm a list of Wikipedia QnA text retrieval tasks. Here are a few examples for your reference:
- Make a comparison between two different types of animals.
- Retrieve documents that bridge two people from different time periods.

Please adhere to the following guidelines:
- Specify what the query is, and what the desired documents are.
- Each retrieval task should cover a wide range of queries, and should not be too specific.

Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!"""

# Query and chunk generation system prompt
QUERY_CHUNK_SYSTEM_PROMPT = """You have been assigned a retrieval task: {task}
Your mission is to write one multi-hop Wikipedia QnA text retrieval example for this task in JSON format. The JSON object must contain the following keys:
- "user_query": a string, a random user search query specified by the retrieval task.
- "reasoning_chain": a string, a reasoning chain that describes the information and retrieve process of the user query. #! TODO: reasoning chain
- "chunks": a list of search path. Each element in the list is a JSON object with the following keys
    - "positive_document": a string, a relevant document for the user query.
    - "hard_negative_document": a string, a hard negative document that only appears relevant to the query.

Please adhere to the following guidelines:
- The "user_query" should be {query_type}, {hop}, wikipedia focused,and diverse in topic.
- The "user_query" should be a {hop_type} query.
- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the query.
- All documents should be at least {num_words} words long.
- The "hard_negative_document" contains some useful information, but it should be less useful or comprehensive compared to the "positive_document".
- Both the query and documents should be in English.
- Do not provide any explanation in any document on why it is relevant or not relevant to the query.
- Both the query and documents require {difficulty} level education to understand.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


class QueryChunkSynthesizer(object):
    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
    ):
        if "gpt" in model:
            self.model = AOAI(model, api_version="2023-12-01-preview")
        else:
            raise ValueError("Model not supported")
        self.max_retry = 3

    def task_generation(self):
        trials = 0
        while trials < self.max_retry:
            try:
                tasks = self.model.chat(TASK_GENERATION_SYSTEM_PROMPT)
                matches = re.findall(r"```python(.*?)```", tasks, re.DOTALL)
                if matches:
                    tasks = matches[0]
                tasks = tasks.strip("\n")
                tasks = eval(tasks)
                if isinstance(tasks, list):
                    return tasks
                elif isinstance(tasks, str):
                    return [tasks]
            except Exception as e:
                trials += 1
        return []

    def query_chunk_generation(
        self,
        task: str,
        query_type: QUERY_TYPE,
        # query_length: QUERY_LENGTH,
        # clarity: CLARITY,
        num_words: DOC_WORDS,
        difficulty: DIFFICULTY,
        hop: HOP,
        hop_type: HOP_TYPE,
    ):
        trials = 0
        query = QUERY_CHUNK_SYSTEM_PROMPT.format(
            task=task,
            query_type=query_type,
            # query_length=query_length, clarity=clarity,
            num_words=num_words,
            difficulty=difficulty,
            hop=hop,
            hop_type=hop_type,
        )
        while trials < self.max_retry:
            try:
                query_chunk = self.model.chat(query)
                matches = re.findall(r"```json(.*?)```", query_chunk, re.DOTALL)
                if matches:
                    query_chunk = matches[0]
                query_chunk = query_chunk.strip("\n")
                data = json.loads(query_chunk)
                # for key in ("user_query", "positive_document", "hard_negative_document"):
                #     if key not in data:
                #         raise ValueError(f"Key {key} not found in the response")
                return data
            except Exception as e:
                trials += 1
        return None


def main():
    synthesizer = QueryChunkSynthesizer()
    tasks = synthesizer.task_generation()
    print(f"Tasks: {tasks}")
    data_list = []
    for task in tqdm(tasks[:1]):
        data = synthesizer.query_chunk_generation(task, "common", "50", "children", "2-hop", "compositional")
        data_list.append(data)
    print(data_list)


if __name__ == "__main__":
    main()
