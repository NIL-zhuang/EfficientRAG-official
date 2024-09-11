NEGATIVE_TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT = """
You have been assigned an information extraction task.
Your mission is to extract the words from a given paragraph so that others(GPT3.5) can answer a question using only your extracted words.
Your extracted words should cover information from the question, including entities (e.g. people, location, film) and core relations.
If the paragraph cannot answer the question, you should extract words that are mostly related to the question, if there is none, return an empty string.
Your response should be in JSON format and include the following key:
- "extracted_words": a string composed of a list of words extracted from the paragraph, separated by a space.

Please adhere to the following guidelines:
- Do not reorder, change, miss, or add words. Keep it the same as the original paragraph.
- Identify and extract ONLY the words explicitly mentioned in either the question or its answer, and strongly related to the question or its answer.
- NEVER label any words that do not contribute meaningful information to the question or answer.
- Only extract words that occured in the paragraph.
- Extract as few words as possible.

Question: {question}
Paragraph: {paragraph}
Your response:
""".strip()
#
