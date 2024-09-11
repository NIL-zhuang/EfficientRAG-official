import json


def load_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: list, file_path: str):
    with open(file_path, "w+", encoding="utf-8") as f:
        for sample in data:
            info = json.dumps(sample, ensure_ascii=False)
            f.write(info + "\n")
