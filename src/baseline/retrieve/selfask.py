import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from direct import (
    DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
    DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
)

from conf import (
    CORPUS_DATA_PATH,
    MODEL_DICT,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from language_models import LanguageModel, get_model
from retrievers import Retriever
from utils import ask_model, load_jsonl, write_jsonl

SELF_ASK_PROMPT = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "The final answer is:", or "Are follow up questions needed here:".

Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
The final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
The final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
The final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
The final answer is: No

Question: {question}
Are follow up questions needed here:
""".strip()

SELF_ASK_PROMPT_MUSIEUQ = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
#
Question: In which year did the publisher of In Cold Blood form?
Are follow up questions needed here: Yes.
Follow up: What business published In Cold Blood?
Intermediate answer: In Cold Blood was published in book form by Random House.
Follow up: Which year witnessed the formation of Random House?
Intermediate answer: Random House was form in 2001.
So the final answer is: 2001
#
Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Are follow up questions needed here: Yes.
Follow up: In which city was The Killing of a Sacred Deer filmed
Intermediate answer: The Killing of a Sacred Deer was filmed in Cincinnati.
Follow up: Who was in charge of Cincinnati?
Intermediate answer: The present Mayor of Cincinnati is John Cranley, so John Cranley is in charge.
So the final answer is: John Cranley
#
Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Are follow up questions needed here: Yes.
Follow up: What city does Signal Hill overlook?
Intermediate answer: Signal Hill is a hill which overlooks the city of St. John's.
Follow up: Where on the Avalon Peninsula is St. John's located?
Intermediate answer: St. John's is located on the eastern tip of the Avalon Peninsula.
So the final answer is: eastern tip
#
Question: {question}
Are follow up questions needed here:
""".strip()

SELF_ASK_PROMPT_WIKIMQA = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
Follow the examples below to answer the questions with natural language.
#
Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Are follow up questions needed here: Yes.
Follow up: When did Blind Shaft come out?
Intermediate answer: Blind Shaft came out in 2003.
Follow up: When did The Mask Of Fu Manchu come out?
Intermediate answer: The Mask Of Fu Manchu came out in 1932.
So the final answer is: The Mask Of Fu Manchu
#
Question: When did John V, Prince Of Anhalt-Zerbst's father die?
Are follow up questions needed here: Yes.
Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?
Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.
Follow up: When did Ernest I, Prince of Anhalt-Dessau die?
Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.
So the final answer is: 12 June 1516
#
Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Are follow up questions needed here: Yes.
Follow up: Who is the director of El Extrano Viaje?
Intermediate answer: The director of El Extrano Viaje is Fernando Fernan Gomez.
Follow up: Who is the director of Love in Pawn?
Intermediate answer: The director of Love in Pawn is Charles Saunders.
Follow up: When was Fernando Fernan Gomez born?
Intermediate answer: Fernando Fernan Gomez was born on 28 August 1921.
Follow up: When was Charles Saunders (director) born?
Intermediate answer: Charles Saunders was born on 8 April 1904.
So the final answer is: El Extrano Viaje
#
Question: {question}
Are follow up questions needed here:
""".strip()

SELF_ASK_PROMPT_HOTPOTQA = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
#
Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Are follow up questions needed here: Yes.
Follow up: Who worked with Modern Records?
Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.
Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?
Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no.
Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?
Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.
So the final answer is: Little Richard
#
Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
Are follow up questions needed here: Yes.
Follow up: What jobs did Chinua Achebe have?
Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.
Follow up: What jobs did Rachel Carson have?
Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.
Follow up: Did Chinua Achebe have more jobs than Rachel Carson?
Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs.
So the final answer is: Chinua Achebe
#
Question: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
Are follow up questions needed here: Yes.
Follow up: Which American rapper is featured by Remember Me Ballin', a CD single by Indo G?
Intermediate answer: Gangsta Boo
Follow up: In which year was Gangsta Boo born?
Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979.
So the final answer is: 1979
#
Question: {question}
Are follow up questions needed here:
""".strip()

GET_ANSWER_PROMPT_TEMPLATE_MAPPING = {
    "hotpotQA": DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    "2WikiMQA": DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
    "musique": DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
}

SELF_ASK_PROMPT_TEMPLATE_MAPPING = {
    "hotpotQA": SELF_ASK_PROMPT_HOTPOTQA,
    "2WikiMQA": SELF_ASK_PROMPT_WIKIMQA,
    "musique": SELF_ASK_PROMPT_MUSIEUQ,
}


def extract_question(generated):
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]

    if "Follow up:" not in last_line:
        print("Follow up not in last line: \n" + generated)

    if ":" not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(":")[-1]

    if after_colon == "":
        return ""
    if " " == after_colon[0]:
        after_colon = after_colon[1:]
    if "?" != after_colon[-1]:
        print("Question not end with ?: " + generated)

    return after_colon


def extract_answer(generated):
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]

    if ":" not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(":")[-1]

    if after_colon == "":
        return ""
    if " " == after_colon[0]:
        after_colon = after_colon[1:]
    if "." == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon


def get_last_line(generated):
    if "\n" not in generated:
        last_line = generated
    else:
        last_line = generated.split("\n")[-1]
    return last_line


class SelfAsk:
    def __init__(
        self,
        model: str,
        dataset: list[dict],
        retriever: Retriever,
        max_iter: int = 3,
        topk: int = 10,
        dataset_name: str = None,
    ):
        self.model: LanguageModel
        self.model = get_model(model)
        self.dataset = dataset
        self.retriever = retriever
        self.max_iter = max_iter
        self.topk = topk
        self.prompt_template = SELF_ASK_PROMPT_TEMPLATE_MAPPING[dataset_name]
        self.get_answer_prompt_template = GET_ANSWER_PROMPT_TEMPLATE_MAPPING[
            dataset_name
        ]

        self.intermediate = "\nIntermediate answer:"
        self.follow_up = "Follow up:"
        self.final_ans = "So the final answer is:"
        self.check_following_question = "\nAre follow up questions needed here:"
        self.max_iter = 5

    def inference(self, workers: int = 10) -> list[str]:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {
                executor.submit(self.infer_sample, sample): idx
                for idx, sample in enumerate(self.dataset)
            }
            results = []
            for future in tqdm_rich(as_completed(tasks), total=len(tasks)):
                idx = tasks[future]
                try:
                    res = future.result()
                    results.append((idx, res))
                except Exception as e:
                    print(f"Error in processing {idx}: {e}")
                    import traceback

                    traceback.print_exc()
            results = [res[1] for res in sorted(results, key=lambda x: x[0])]
        return results

    def get_answer(self, question):
        knowledge_list = self.retriever.search(question, self.topk)[0]
        knowledge = "\n".join([f"{doc['text']}" for doc in knowledge_list])
        prompt = self.get_answer_prompt_template.format(
            knowledge=knowledge, question=question
        )
        model_response = ask_model(
            self.model,
            prompt,
            type="json",
            check_if_valid=lambda x: type(x) is dict and "answer" in x.keys(),
            mode="chat",
        )
        if model_response is None:
            return "unknown", knowledge_list
        return model_response["answer"], knowledge_list

    def call_model(self, current_prompt, stop: Union[str, list[str]]):
        def check_if_valid(s: str):
            if type(stop) is str:
                return stop in s
            elif type(stop) is list:
                return any([x in s for x in stop])

        response = ask_model(
            self.model,
            current_prompt,
            type="text",
            mode="completion",
            check_if_valid=check_if_valid,
        )
        if response is None:
            return ""

        response = response.strip()
        if type(stop) is str:
            return response.split(stop)[0]
        elif type(stop) is list:
            idx_list = [response.find(x) for x in stop]
            idx_list = [x if x != -1 else float("inf") for x in idx_list]
            min_idx = min(idx_list)
            for idx, stop_word in zip(idx_list, stop):
                if idx == min_idx:
                    return response.split(stop_word)[0].strip()
        return ""

    def infer_sample(self, sample: dict) -> dict:
        question = sample["question"]
        results = {
            "id": sample["id"],
            "answer": sample["answer"],
            "oracle": [
                f"{sample['id']}-{'{:02d}'.format(chunk['positive_paragraph_idx'])}"
                for chunk in sample["decomposed_questions"].values()
            ],
            "question": question,
            "knowledges": list(),
            "internal_questions": [],
        }

        cur_prompt = self.prompt_template.format(question=question)
        cur_iter = 0
        ret_text = self.call_model(cur_prompt, [self.intermediate, self.final_ans])
        while self.follow_up in get_last_line(ret_text) and cur_iter < self.max_iter:
            cur_iter += 1
            cur_prompt += ret_text
            question = extract_question(ret_text)
            results["internal_questions"].append(question)
            external_answer, knowledge_list = self.get_answer(question)
            results["knowledges"].extend(knowledge_list)

            if external_answer is not None:
                cur_prompt += f"{self.intermediate} {external_answer}."
                ret_text = self.call_model(
                    cur_prompt, [self.intermediate, self.final_ans]
                )
            else:
                # very rare when return no answer
                cur_prompt += self.intermediate
                answer = self.call_model(
                    cur_prompt, ["\n" + self.follow_up, self.final_ans]
                )
                cur_prompt += answer

        if self.final_ans not in ret_text:
            cur_prompt += f"{self.final_ans}"
            ret_text = self.call_model(cur_prompt, "\n")

        final_prompt = cur_prompt + ret_text
        final_answer = extract_answer(final_prompt)

        results["model_answer"] = final_answer
        results["history"] = cur_prompt[len(self.prompt_template) :]
        return results


def main(opt: argparse.Namespace):
    passage_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "corpus.jsonl")
    if opt.retriever == "e5-base-v2":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "e5-base")
    elif opt.retriever == "contriever":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "contriever")
    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opt.retriever,
    )
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
            opt.dataset,
            f"{opt.split}.jsonl",
        )
    )
    model = MODEL_DICT[opt.model]
    selfask = SelfAsk(
        model=model,
        dataset=dataset,
        retriever=retriever,
        topk=opt.topk,
        dataset_name=opt.dataset,
    )
    import time
    start = time.time()
    results = selfask.inference(opt.workers)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")
    print(f"Average time per sample: {(end - start) / len(dataset):.2f}s")
    save_path = os.path.join(
        "results/retrieve/selfask", f"{opt.dataset}-{opt.split}.jsonl"
    )
    write_jsonl(results, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--model", type=str, default="llama-8B")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
