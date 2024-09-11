import json
import os
import re
import sys
from typing import Literal
import random
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from language_models import AOAI


QUERY_TOPIC = Literal["extremely long-tail", "long tail", "common"]
QUERY_TYPE = Literal["multi-hop", "implicit"]
# a dictionary to store the query definition
QUERY_DEF = {
    "multi-hop": "a query that requires multi-hop reasoning path to the answer",
    "implicit": "a query that appears to be one-hop, but the reasoning path requires multiple steps to the answer",
}

# QUERY_DEF = Literal['a query that requires multi-hop reasoning path to the answer', 'a query that appears to be one-hop, but the reasoning path requires multiple steps to the answer']
# QUERY_LENGTH = Literal['less than 5 words', '5 to 15 words', 'at least 10 words']
DIFFICULTY = Literal["children", "high school", "college", "PhD"]
CLARITY = Literal["clear", "understandable with some effort", "ambiguous"]
DOC_WORDS = Literal["50", "100", "200", "300", "400", "500"]
HOP = Literal["single-hop", "2-hop", "3-hop", "4-hop"]
HOP_TYPE = Literal["combination", "extension"]

# This prompt seems to have poor performance on reasoning path as each hop does not necessarily have a meanning, for example, the 4th reasoning path below:
# "user_query": "If a person has a PhD in Astrophysics and another in Philosophy, which academic discipline's doctorate typically takes longer to complete?",
# "reasoning_path": [
#     "Determine the average duration of completing a PhD in Astrophysics.",
#     "Determine the average duration of completing a PhD in Philosophy.",
#     "Compare the typical durations of the two PhD programs.",
#     "Conclude which PhD typically takes longer to complete."
# ],
# "answer": "A PhD in Philosophy typically takes longer to complete."

QUERY_GENERATION_SYSTEM_PROMPT_v0 = """Brainstorm a list of queries, where each query possibly needs a multi-hop reasoning path to the answer.

Here are a few examples for your reference:
- Which moive has a earlier publication date, Moonstruck or One step to Eternity?
- what are some tourist attractions in the biggest city in Japan?
- Who is the only person to win an olympic medal and a Nobel prize?
- The director of The Millionaire's Double and the director of The Groundstar Conspiracy, who was born later?

Please adhere to the following guidelines:
- Specify what the query is, and what is the answer to the query.
- Specify the reasoning path to the answer, and the reasoning path should be a list of one-hop reasoning path.
- Both the query and reasoning path should be in English.
- Both the query and reasoning path require {difficulty} level education to understand.

Your output must always be a python list of JSON object only, with about 5 elements. Each JSON object should contain the following keys:
- "user_query": a string, a multi-hop query. The query should be {query_type}, and diverse in topic.
- "reasoning_path": a list of string. 
- "answer": a string, the answer to the user query.

Do not explain yourself or output anything else. Be creative!"""

MULTI_HOP_EXAMPLES = []

IMPLICIT_EXAMPLES = [
    {
        "question": "Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?",
        "answer": "Charles Armand René de La Trémoille",
        "decomposed_question": {
            "1": {
                "sub_question": "Who is Charles Bretagne Marie De La Trémoille's father?",
                "answer": "Jean Bretagne Charles de La Trémoille",
                "dependency": [],
            },
            "2": {
                "sub_question": "Who is Jean Bretagne Charles de La Trémoille's father?",
                "answer": "Charles Armand René de La Trémoille",
                "dependency": ["1"],
            },
        },
    },
    {
        "question": "Is shrimp scampi definitely free of plastic?",
        "answer": "No",
        "decomposed_question": {
            "1": {
                "sub_question": "What protein is Shrimp scampi made out of?",
                "answer": "Shrimp scampi is a dish made with shrimp.",
                "dependency": [],
            },
            "2": {
                "sub_question": "What have shrimp been found to contain?",
                "answer": "Shrimp have been found to contain microplastics.",
                "dependency": ["1"],
            },
            "3": {
                "sub_question": "Are microplastics free from plastic?",
                "answer": "Microplastics are plastic material.",
                "dependency": ["2"],
            },
        },
    },
]

# # Comparison case
# {
#     'question': "Can Arnold Schwarzenegger lift a rhino?",
#     'answer': 'No',
#     'decomposed_question':{
#         "1": {
#             "sub_question": "What is the largest weight Arnold Schwarzenegger can lift?",
#             "answer": "710 pounds",
#             "dependency": []
#         },
#         "2":{
#             "sub_question": "What is the weight of a rhino?",
#             "answer": "average between 1700 and 3000 pounds",
#             "dependency": []
#         },
#         "3":{
#             "sub_question": "Is 710 pounds larger than the weight of a rhino?",
#             "answer": "No",
#             "dependency": ["1", "2"]
#         }
#     }
# }

FULL_QUERY_PATH_GENERATION_SYSTEM_PROMPT = """Brainstorm a list of queries, where each query possibly needs a multi-hop reasoning path to the answer.

Here are a few examples for your reference:
- Which moive has a earlier publication date, Moonstruck or One step to Eternity?
- what are some tourist attractions in the biggest city in Japan?
- Who is the only person to win an olympic medal and a Nobel prize?
- The director of The Millionaire's Double and the director of The Groundstar Conspiracy, who was born later?

Please adhere to the following guidelines:
- Specify what the query is, and what is the answer to the query.
- Specify the reasoning path to the answer, and the reasoning path should be a list of one-hop reasoning path.
- Both the query and reasoning path should be in English.
- Both the query and reasoning path require {difficulty} level education to understand.

Your output must always be a python list of JSON object only, with about 5 elements. Each JSON object should contain the following keys:
- "user_query": a string, a {query_type} query, {query_def}. The query should be {query_topic}, and diverse in topic.
- "reasoning_path": a string, a reasoning path that indicates the reasoning process from the query to the answer.
- "answer": a string, the answer to the user query.

Do not explain yourself or output anything else. Be creative!"""


# load examples from json file
with open("examples/query_few_shot_example.json", "r") as f:
    QUERY_EXAMPLES = json.load(f)

QUERY_GENERATION_SYSTEM_PROMPT = """Brainstorm a list of questions, where each question possibly needs a multi-hop reasoning path to the answer.

Here are a few question examples for your reference:
{reference_examples}

Please adhere to the following guidelines:
- The question should be in English.
- The question requires {difficulty} level education to understand.

Your output must always be a python list of JSON object only, with about {query_num} elements. Each JSON object should contain the following keys:
- "question": a string. The question should be {query_topic}, and diverse in topic. The question should be also diverse in types, such as open-ended, wh-questions, binary questions, etc. Check the above question example for reference.
- "answer": a string, the answer to the question.
- "reasoning": a string, a think step-by-step reasoning path that indicates the reasoning process from the question to the answer.

Do not explain yourself or output anything else. Be creative!"""

REASONING_PATH_GENERATION_SYSTEM_PROMPT = """You have been assigned a reasoning task. You mission is to decompose the given question into multiple sub_questions and these sub_questions may have dependencies on each other. 

Write reasoning path for the given question, answer and reasoning:
{question}

Please adhere to the following guidelines:
- The reasoning path should be a list of JSON objects. 
- Use the number of the sub_question as the key and the "sub_question", "answer", "supporting_document" and "dependency" as the value. 
- The "supporting_document" must be created independent of the sub_question. Avoid copying the sub_question verbatim. It's acceptable if some parts of the "supporting_document" are not topically related to the query.
- The "supporting_document" should be at least {doc_word_num} words long.
-The dependency is a list of the number of the sub_questions that the current sub_question depends on. If the sub_question does not depend on any other sub_questions, the dependency should be an empty list.

Here is the format for reference:
{{
    "1": {{
        "sub_question": "question",
        "answer": "answer to sub_question",
        "supporting_document": "a relevant document for the sub_question"
        "dependency": []
    }},
    "2": {{
        "sub_question": "question",
        "answer": "answer to sub_question",
        "supporting_document": "a relevant document for the sub_question"
        "dependency": ["1"]
    }}
}}

Do not explain yourself or output anything else."""


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

    def pick_examples(self, example_set=QUERY_EXAMPLES, example_num=3, random_seed=0):
        random.seed(random_seed)
        examples = random.sample(example_set, example_num)
        examples_text = "\n".join([f"- {example}" for example in examples])
        return examples_text

    def query_generation(
        self,
        query_topic: QUERY_TOPIC,
        difficulty: DIFFICULTY,
        query_num=5,
        random_seed=0,
    ):
        trials = 0
        reference_examples = self.pick_examples(random_seed=random_seed)
        query = QUERY_GENERATION_SYSTEM_PROMPT.format(
            reference_examples=reference_examples,
            query_num=query_num,
            query_topic=query_topic,
            difficulty=difficulty,
        )
        while trials < self.max_retry:
            try:
                output_query = self.model.chat(query, json_mode=False)
                print(output_query)
                match_python = re.findall(r"```python(.*?)```", output_query, re.DOTALL)
                match_json = re.findall(r"```json(.*?)```", output_query, re.DOTALL)
                if match_python:
                    output_query = match_python[0]
                elif match_json:
                    output_query = match_json[0]
                output_query = output_query.strip("\n")
                data = json.loads(output_query)
                return data
            except Exception as e:
                trials += 1
        return None

    # append the list of query json to a file
    def save_query(self, save_path, query_json):
        if not os.path.exists(save_path):
            with open(save_path, "w") as file:
                json.dump([], file)
        with open(save_path, "r") as f:
            existing_data = json.load(f)
        existing_data.extend(query_json)
        with open(save_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    def query_reasoning_path_generation(self, question, doc_word_num: DOC_WORDS):
        trials = 0
        query = REASONING_PATH_GENERATION_SYSTEM_PROMPT.format(question=question, doc_word_num=doc_word_num)
        while trials < self.max_retry:
            try:
                output_query = self.model.chat(query, json_mode=False)
                print(output_query)
                match_python = re.findall(r"```python(.*?)```", output_query, re.DOTALL)
                match_json = re.findall(r"```json(.*?)```", output_query, re.DOTALL)
                if match_python:
                    output_query = match_python[0]
                elif match_json:
                    output_query = match_json[0]
                output_query = output_query.strip("\n")
                data = json.loads(output_query)
                return data
            except Exception as e:
                trials += 1
        return None


def main():
    save_path = "examples/query_generation_example.json"
    synthesizer = QueryChunkSynthesizer()
    # query_reasoning_path = synthesizer.query_reasoning_path_generation('implicit', 'common', 'college')
    # queries = synthesizer.query_generation(query_topic='common', difficulty='high school', random_seed=2)
    # synthesizer.save_query(save_path, queries)
    ## load json list from save path
    with open(save_path, "r") as f:
        l_query = json.load(f)
    question = l_query[0]
    print(question)
    query_reasoning_path = synthesizer.query_reasoning_path_generation(question=question, doc_word_num="100")
    print(query_reasoning_path)


if __name__ == "__main__":
    main()
