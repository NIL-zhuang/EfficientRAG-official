import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from conf import (
    CORPUS_DATA_PATH,
    MODEL_DICT,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from language_models import LanguageModel, get_model
from retrievers import Retriever
from utils import ask_model, load_jsonl, write_jsonl

SYSTEM_PROMPT = """
Answer the questions based on given documents, you must give the answer in the format "So the final answer is".
Think step by step and answer the questions based on given documents. You must answer in JSON format with key "thought" and "answer".
""".strip()

ITER_RETGEN_MUSIQUE_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "In Cold Blood was first published in book form by Random House. Random House was form in 2001.",
    "answer": "2011"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley. Therefore, John Cranley is in charge of the city.",
    "answer": "John Cranley"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Let's think step by step.
<Answer>:
{{
    "thought": "Signal Hill is a hill which overlooks the city of St. John's. St. John's is located on the eastern tip of the Avalon Peninsula.",
    "answer": "eastern tip"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_WIKIMQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Blind Shaft is a 2003 film, while The Mask Of Fu Manchu opened in New York on December 2, 1932. 2003 comes after 1932. Therefore, The Mask Of Fu Manchu came out earlier than Blind Shaft.",
    "answer": "The Mask Of Fu Manchu"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "John V, Prince Of Anhalt-Zerbst was the son of Ernest I, Prince of Anhalt-Dessau. Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.",
    "answer": "12 June 1516"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The director of El Extrano Viaje is Fernando Fernan Gomez, who was born on 28 August 1921. The director of Love In Pawn is Charles Saunders, who was born on 8 April 1904. 28 August 1921 comes after 8 April 1904. Therefore, Fernando Fernan Gomez was born later than Charles Saunders.",
    "answer": "El Extrano Viaje"
}}
```

Now based on the given doc, answer the question after <Question>
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_HOTPOTQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter.",
    "answer": "Little Richard"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{
    "thought": "Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson.",
    "answer": "Chinua Achebe"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{
    "thought": "Remember Me Ballin' is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell's stage name, who was born in August 7, 1979, and is an American rapper.",
    "answer": "1979"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()


class IterRetGen:
    def __init__(
        self,
        model: str,
        dataset: list[dict],
        retriever: Retriever,
        max_iter: int = 3,
        topk: int = 10,
        prompt_template: str = None,
    ) -> None:
        self.model: LanguageModel
        self.model = get_model(model)
        self.dataset = dataset
        self.retriever = retriever
        self.max_iter = max_iter
        self.topk = topk
        self.prompt_template = prompt_template

    def inference(self, workers=10) -> list[dict]:
        print("Start inference")
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
                    print(f"Failed on sample {idx}")
                    import traceback

                    traceback.print_exc()
            results = [res[1] for res in sorted(results, key=lambda x: x[0])]
        return results

    def construct_prompt(self, sample: dict, prev_answer: str = None) -> str:
        q = sample["question"]
        if prev_answer is not None:
            q = f"{q} {prev_answer}"

        docs = self.retriever.search(q, top_k=self.topk)
        d = "\n".join([f"{doc['text']}" for doc in docs[0]])
        prompt = self.prompt_template.format(documents=d, question=sample["question"])
        return prompt, docs

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
            "inter": {},
        }
        internal_query = None
        cur_iter = 0
        while cur_iter < self.max_iter:
            cur_iter += 1
            prompt, docs = self.construct_prompt(sample, internal_query)
            check_if_valid = (
                lambda x: type(x) is dict and "answer" in x and "thought" in x
            )
            model_response = ask_model(
                self.model,
                prompt,
                system_msg=SYSTEM_PROMPT,
                type="json",
                mode="chat",
                check_if_valid=check_if_valid,  # noqa
            )
            internal_query = model_response["thought"]
            results["inter"][cur_iter] = {
                "cur_answer": model_response["answer"],
                "thought": model_response["thought"],
                "docs": docs,
            }
        results["model_answer"] = model_response["answer"]
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-8B")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
        default="musique",
    )
    parser.add_argument("--split", type=str, default="demo")
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=3)
    args = parser.parse_args()
    return args


def main(opts: argparse.Namespace):
    start = time.time()
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
            opts.dataset,
            f"{opts.split}.jsonl",
        )
    )
    print(f"Loaded {len(dataset)} samples")
    passage_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, "corpus.jsonl")
    embedding_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, opts.retriever)
    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opts.retriever,
    )
    model = MODEL_DICT[opts.model]
    prompt_template_mapping = {
        "hotpotQA": ITER_RETGEN_HOTPOTQA_PROMPT,
        "2WikiMQA": ITER_RETGEN_WIKIMQA_PROMPT,
        "musique": ITER_RETGEN_MUSIQUE_PROMPT,
    }
    template = prompt_template_mapping[opts.dataset]
    retgen = IterRetGen(
        model, dataset, retriever, max_iter=opts.max_iter, prompt_template=template
    )
    results = retgen.inference(opts.workers)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")
    print(f"Average time per sample: {(end - start) / len(dataset):.2f}s")
    save_path = os.path.join(
        "results/retrieve/itergen", f"{opts.dataset}-{opts.split}.jsonl"
    )
    write_jsonl(results, save_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
