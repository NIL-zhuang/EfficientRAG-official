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
HOP_TYPE = Literal["combination", "extension"]

# Topology generation system prompt
# TODO: the examples can be seed examples from KGQA cases

TOPOLOGY_GENERATION_SYSTEM_PROMPT = """Brainstorm a list of tuples of query, multi-hop reasoning path, and the answer. The reasoning path consists of a series of hops, and one hop is a relation between two entities, represented as ["entity", "relation", "entity"].

There are several basic types of reasoning paths, here are a few basic examples for your reference:
- Composition:
"user_query": "which country is the birthplace of the father Walter Chetwync?",
"reasoning_path": ["Walter Chetwynd, 1st Viscount Chetwynd", "father", "John Chetwynd"], ["John Chetwynd", "country of citizenship", "United Kingdom of Great Britain and Ireland"]
"answer": "United Kingdom of Great Britain and Ireland"

- Comparison:
"user_query": "which movie has a earlier publication date, Moonstruck or One step to Eternity?",
"reasoning_path": ["Moonstruck", "publication date", "1987"], ["One step to Eternity", "publication date", "1954"]
"answer": "One step to Eternity"

- Intersection:
"user_query": "Who is the only person to win an olympic medal and a Nobel prize?",
"reasoning_path": ["Philip John Noel-Baker", "won", "An olympic medal"], ["Philip John Noel-Baker", "award received", "Nobel Prize in Peace"]
"answer": "Philip John Noel-Baker"

There can be more types of multi-hop reasoning paths that could be combination of the above types, or extension of the above types.

Here are a few complex examples for your reference:
- Extension of Composition:
"user_query": "What influenced the architectural style of the Hagia Sophia?",
"reasoning_path": ["Hagia Sophia", "construction period", "6th century"], ["6th century", "Byzantine Empire", "influence"], ["Byzantine Empire", "Roman architecture", "influence"]
"answer": "Roman architecture"

- Combination of Composition and Comparison:
"user_query": "The director of The Millionaire's Double and the director of The Groundstar Conspiracy, who was born later?",
"reasoning_path": ["The Millionaire's Double", "director", "Harry Davenport"], ["The Groundstar Conspiracy", "director", "Lamont Johnson"], ["Harry Davenport", "date of birth", "1866"], ["Lamont Johnson", "date of birth", "1922"]
"answer": "Lamont Johnson"

The tuple is a JSON object which must contain the following keys:
- "user_query": a string, a multi-hop query.
- "reasoning_chain": a string, a reasoning chain that describes the information and retrieve process of the user query.
- "answer": a string, the answer to the user query.

Please adhere to the following guidelines:
- The "user_query" should be {query_type},and diverse in topic.
- The "reasoning_chain" should contain {hop} hops.
- Both the query and reasoning path should be in English.
- Both the query and reasoning path require {difficulty} level education to understand.
- the reasoning path should be {hop_type} of basic types of reasoning paths.

Your output must always be a python list of JSON object only, with about 5 elements. Do not explain yourself or output anything else. Be creative!"""


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

    def query_reasoning_path_generation(
        self,
        query_type: QUERY_TYPE,
        difficulty: DIFFICULTY,
        hop: HOP,
        hop_type: HOP_TYPE,
    ):
        trials = 0
        query = TOPOLOGY_GENERATION_SYSTEM_PROMPT.format(
            query_type=query_type, difficulty=difficulty, hop=hop, hop_type=hop_type
        )
        while trials < self.max_retry:
            try:
                query_reasoning_path = self.model.chat(query)
                print(query_reasoning_path)
                matches = re.findall(r"```python(.*?)```", query_reasoning_path, re.DOTALL)
                if matches:
                    query_reasoning_path = matches[0]
                query_reasoning_path = query_reasoning_path.strip("\n")
                data = json.loads(query_reasoning_path)
                return data
            except Exception as e:
                trials += 1
        return None


def main():
    synthesizer = QueryChunkSynthesizer()
    query_reasoning_path = synthesizer.query_reasoning_path_generation("common", "PhD", "5-hop", "combination")


if __name__ == "__main__":
    main()
