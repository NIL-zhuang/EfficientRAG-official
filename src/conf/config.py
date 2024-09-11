import os
from os.path import sep

DATA_BASE_PATH = "data"
DATASET_PATH = os.path.join(DATA_BASE_PATH, "dataset")

# Data Synthesize
SYNTHESIZED_DECOMPOSED_DATA_PATH = f"data{sep}synthesized_decomposed"
SYNTHESIZED_TOKEN_LABELING_DATA_PATH = f"data{sep}synthesized_token_labeling"
SYNTHESIZED_TOKEN_EXTRACTED_DATA_PATH = f"data{sep}token_extracted"
SYNTHESIZED_NEXT_QUERY_DATA_PATH = f"data{sep}synthesized_next_query"
SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH = f"data{sep}next_query_extracted"
SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH = f"data{sep}negative_sampling"
SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH = f"data{sep}negative_sampling_labeled"
SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH = (
    f"data{sep}negative_sampling_extracted"
)
EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH = f"data{sep}efficient_rag{sep}labeler"
EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH = f"data{sep}efficient_rag{sep}filter"
CORPUS_DATA_PATH = f"data{sep}corpus"

SYNTHESIZED_SPAN_LABELING_DATA_PATH = f"data{sep}synthesized_span_labeling"

# Results
RETRIEVE_RESULT_PATH = f"results{sep}retrieve"

CONTINUE_TAG = "<CONTINUE>"
FINISH_TAG = "<FINISH>"
TERMINATE_TAG = "<TERMINATE>"

TAG_MAPPING = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 2,
}
TAG_MAPPING_REV = {v: k for k, v in TAG_MAPPING.items()}

TAG_MAPPING_TWO = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 0,
}
TAG_MAPPING_TWO_REV = {
    0: CONTINUE_TAG,
    1: TERMINATE_TAG,
}
TERMINATE_ID = TAG_MAPPING[TERMINATE_TAG]

# Special Tokens
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

MODEL_PATH = f"model_cache{sep}deberta-v3-large"

MODEL_DICT = {
    "gpt35": "gpt-35-turbo-1106",
    "gpt4": "gpt-4-0125-preview",
    "llama": "Meta-Llama-3-70B-Instruct",
    "llama-8B": "Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-chat",
}

EMBEDDING_ALIAS = {
    "contriever": "contriever",
    "e5-base-v2": "e5-base",
    "e5-large-v2": "e5-large",
    "ada-002": "ada-002",
}
