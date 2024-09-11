# given question, decomposed question, answer, e.t.c
# decompose the original question into successor sub_questions

hotpotQAFactPrompt = """
document for sub_question #{question_id}
supporting facts: {facts}"""


HotpotQAPromptComparison = """You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting documents, and such that you (GPT-4) can answer each sub_question independently from each document.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. The sub_question MUST NOT contain more information than the original question and its dependent sub_question. NEVER introduce information from the documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": an empty list. Because the sub_question is independent.

The origin multi-hop questions is: {question}
Followings are documents to answer each sub_question.
You MUST decompose the original multi-hoip question based on the given documents. DO NOT change the order or miss anyone of them.
{chunks}

Your output must always be a JSON object only, do not explain yourself or output anything else.
Follow the documents, synthesize the sub_questions and answers one-by-one. NEVER miss any of them.
"""

HotpotQAPromptCompose = """You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting documents, and such that you (GPT-4) can answer each sub_question independently from each document.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. The sub_question MUST NOT contain more information than the original question and its dependent sub_question. NEVER introduce information from the documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here.

The origin multi-hop questions is: {question}
And its answer is: {answer}
Followings are documents to answer each sub_question.
Make sure one sub_question depends on the other! Identify which sub_question depends on the answer of another according to the question.
You MUST decompose the question based on the documents with the sub_questions and answers. DO NOT change the order or miss anyone of them.
{chunks}

Your output must always be a JSON object only, do not explain yourself or output anything else.
Follow the documents, synthesize the sub_questions and answers one-by-one. NEVER miss any of them.
"""
