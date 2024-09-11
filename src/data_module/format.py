import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conf import SEP_TOKEN

INFO_TEMPLATE = "Info: {info}"
QUERY_INFO_SENTENCE_TEMPLATE = f"Query: {{query}} {{info_str}}"


def build_query_info_sentence(info_list: list[str], query: str) -> str:
    infos = [INFO_TEMPLATE.format(info=info) for info in info_list]
    info_str = "; ".join(infos)
    res = QUERY_INFO_SENTENCE_TEMPLATE.format(query=query, info_str=info_str)
    return res
