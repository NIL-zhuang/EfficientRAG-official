import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from conf import SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH
from language_models import LanguageModel, get_model
from utils import ask_model, load_jsonl, write_jsonl

DIRECT_PROMPT_HOTPOTQA = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>:
```json
{{"answer": "Little Richard"}}
```

<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"answer": "Chinua Achebe"}}
```

<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"answer": "1979"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

DIRECT_PROMPT_WIKIMQA = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>:
```json
{{"answer": "The Mask Of Fu Manchu"}}
```

<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"answer": "12 June 1516"}}
```

<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"answer": "El Extrano Viaje"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

DIRECT_PROMPT_MUSIQUE = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: In which year did the publisher of In Cold Blood form?
<Answer>:
```json
{{"answer": "2001"}}
```

<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"answer": "John Cranley"}}
```

<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"answer": "eastern tip"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

COT_PROMPT_HOTPOTQA = """
As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your answer should be after <Answer> in JSON format with key "thought" and "answer" and their values should be string.

There are some examples for you to refer to:
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>:
```json
{{"thought":"Modern Record is a big R&B label with artists including Etta James, Joe Houston, Little Richard, Ike, Tina Turner and John Lee Hooker in the 1950s and 1960s. Little Richard is an American musician, signer actor and songwriter, born in December 5 1932. So the answer is Little Richard.","answer": "Little Richard"}}
```

<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"thought":"Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Chinua Achebe has 4 jobs while Rachel Carson has 3 jobs. So the answer is Chinua Achebe.","answer": "Chinua Achebe"}}
```

<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"thought":"Remember Me Ballin' is the CD singer by Indo G that features Gangsta Boo, who is named Lola Mitchell, an American rapper born in 1979. So the answer is 1979.","answer": "1979"}}

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

COT_PROMPT_WIKIMQA = """
As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your answer should be after <Answer> in JSON format with key "thought" and "answer" and their values should be string.

There are some examples for you to refer to:
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>:
```json
{{"thought": "Blind Shaft is a 2003 Chinese film, and The Mask Of Fu Manchu is a 1932 American pre-Code adventure film. The Mask Of Fu Manchu came out first. So the answer is The Mask Of Fu Manchu.", "answer": "The Mask Of Fu Manchu"}}
```

<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"thought": "The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau. He died on 12 June 1516. So the answer is 12 June 1516.", "answer": "12 June 1516"}}
```

<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"thought": "The director of El Extrano Viaje is Fernando Fernan Gomez, he was born on 29 August 1921. The director of Love In Pawn is Charles Saunders, he was born on 8 April 1904. Fernando Fernan Gomez was born later, so film El Extrano Viaje has the director who was born later. So the answer is El Extrano Viaje.", "answer": "El Extrano Viaje"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

COT_PROMPT_MUSIQUE = """
As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your answer should be after <Answer> in JSON format with key "thought" and "answer" and their values should be string.

There are some examples for you to refer to:
<Question>: In which year did the publisher of In Cold Blood form?
<Answer>:
```json
{{"thought": "The publisher of In Cold Blood is Random house, which was formed in 2001. So the answer is 2001.", "answer": "2001"}}
```

<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"thought": "The killing of a Scared Deer was filmed in Cincinnati, Ohio, where John Cranley is the mayor. So the answer is John Cranley.", "answer": "John Cranley"}}
```

<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"thought": "Signal Hill overlooks the city St. John's, which is located on the eastern tip of the Avalon Peninsula. So the answer is eastern tip.", "answer": "eastern tip"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--model", type=str, default="llama-8B")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--mode", choices=["direct", "cot"], default="direct")
    args = parser.parse_args()
    return args


def process_sample(prompt_template: str, model: LanguageModel, sample: dict):
    question = sample["question"]
    prompt = prompt_template.format(question=question)
    result = ask_model(
        model,
        prompt,
        type="json",
        check_if_valid=lambda x: type(x) is dict and "answer" in x,
        mode="chat",
    )
    prediction = result["answer"]
    # print(f"Predicted answer: {prediction}", '*' * 20)
    result = {
        "question": question,
        "answer": sample["answer"],
        "model_output": prediction,
    }
    return result


def process(
    prompt_template: str, model: LanguageModel, data: list[dict], workers: int = 10
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {
            executor.submit(process_sample, prompt_template, model, sample): idx
            for idx, sample in enumerate(data)
        }
        results = []
        for future in tqdm_rich(as_completed(tasks), total=len(tasks)):
            idx = tasks[future]
            try:
                res = future.result()
                results.append((idx, res))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
    results = [r[1] for r in sorted(results, key=lambda x: x[0])]
    return results


def main(opts: argparse.Namespace):
    prompt_template_mapping = {
        "direct": {
            "musique": DIRECT_PROMPT_MUSIQUE,
            "2WikiMQA": DIRECT_PROMPT_WIKIMQA,
            "hotpotQA": DIRECT_PROMPT_HOTPOTQA,
        },
        "cot": {
            "musique": COT_PROMPT_MUSIQUE,
            "2WikiMQA": COT_PROMPT_WIKIMQA,
            "hotpotQA": COT_PROMPT_HOTPOTQA,
        },
    }
    prompt_template = prompt_template_mapping[opts.mode][opts.dataset]
    model: LanguageModel
    model = get_model(opts.model)
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
            opts.dataset,
            f"{opts.split}.jsonl",
        )
    )
    print(f"Loaded {len(dataset)} data points from {opts.dataset}-{opts.split}")

    import time
    start = time.time()
    results = process(prompt_template, model, dataset, opts.workers)
    end = time.time()
    print(f"Processed {len(results)} samples in {end - start:.2f} seconds")
    print('Average time per sample:', (end - start) / len(dataset))
    output_path = f"results/direct/{opts.mode}/{opts.dataset}-{opts.split}.jsonl"
    write_jsonl(results, output_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
