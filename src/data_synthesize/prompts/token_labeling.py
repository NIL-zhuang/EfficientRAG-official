TOKEN_LABELING_SYSTEM_MSG = """
You are an outstanding linguist, adept at extracting information relevant to specific questions from documents.
""".strip()

TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE = """
You are assigned an information extraction task.
Your mission is to extract the words from a given paragraph so that others can answer the question using only your extracted words.
We will show you the question(starting with <Question>), the document paragraph(starting with <Document>) and the answer(starting with <Answer>).
You should think step by step, and extract the words that relevant to the question and the answer.
Your answer should compose <Thought> part and final extracted words in JSON format starting with <JSON_OUTPUT> embraced by ```json and ```.

Your <Thought> part must contain the following steps:
1. Identify the information in the document that is relevant to the question
2. Identify the information in the document that is relevant to the answer
3. Extract the words that are relevant to both the question and the answer
Your <JSON_OUTPUT> must include the following key:
- "question_related_words": a string composed of a list of words extracted from the paragraph, separated by a space, that are relevant to the question.
- "answer_related_words": a string composed of a list of words extracted from the paragraph, separated by a space, that are relevant to the answer.
- "extracted_words": a string composed of a list of words extracted from the paragraph, separated by a space.

Please adhere to the following guidelines:
- Do not reorder, change, miss, or add words. Keep it the same as the original paragraph.
- Identify and extract ONLY the words explicitly mentioned in either the question or its answer, and strongly related to the question or its answer.
- NEVER label any words that do not contribute meaningful information to the question or answer.
- Only extract words that occur in the paragraph.

Here are some examples for you to refer to:

<Question>: Where is the Alas found?
<Document>: Alas people: The Alas people are an ethnic group that inhabits Southeast Aceh Regency, Aceh, Indonesia. They speak the Alas language, which is related to the Batak languages.
<Answer>: Indonesia
<Thought>:
1. The question is asking about the location of Alas. "Alas" in the document refers to the question.
2. "Southeast Aceh Regency Aceh Indonesia" in the document refers to the answer.
3. The extracted words are "Alas Southeast Aceh Regency Aceh Indonesia"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "Alas",
    "answer_related_words": "Southeast Aceh Regency Aceh Indonesia",
    "extracted_words": "Alas Southeast Aceh Regency Aceh Indonesia"
}}
```

<Question>: What is the meaning of Hindu in the Arabic dictionary?
<Document>: Hindus: The word Hindu is derived from the Indo - Aryan and Sanskrit word Sindhu, which means ``a large body of water '', covering`` river, ocean''. It was used as the name of the Indus river and also referred to its tributaries. The actual term 'hindu' first occurs, states Gavin Flood, as ``a Persian geographical term for the people who lived beyond the river Indus (Sanskrit: Sindhu) '', more specifically in the 6th - century BCE inscription of Darius I. The Punjab region, called Sapta Sindhava in the Vedas, is called Hapta Hindu in Zend Avesta. The 6th - century BCE inscription of Darius I mentions the province of Hi (n) dush, referring to northwestern India. The people of India were referred to as Hinduv\u0101n (Hindus) and hindav\u012b was used as the adjective for Indian in the 8th century text Chachnama. The term 'Hindu' in these ancient records is an ethno - geographical term and did not refer to a religion. The Arabic equivalent Al - Hind likewise referred to the country of India.
<Answer>: the country of India
<Thought>:
1. The question is asking the meaning of Hindu in the Arabic dictionary. The words "Hindu" and "Arabic" in the document refer to the question.
2. "the country of India" in the document refers to the answer.
3. The extracted words are "Hind Arabic the country of India"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "Hind Arabic",
    "answer_related_words": "the country of India",
    "extracted_words": "Hind Arabic the country of India"
}}
```

<Question>: What band was Nick Rhodes a member of?
<Document>: Only After Dark (album): \"Only After Dark\" is a compilation album that was compiled by Nick Rhodes and John Taylor from Duran Duran, and recreates a night at Birmingham's Rum Runner nightclub, during the post punk days of the late 70s/early 80s when a new sound of glam/punk/electronica started to crystallize. The CD captures some of the discs that Nick spun when he was deejaying for £10 a night at the club and Duran Duran were the resident band. The inspiration for it came when in 2000 John and Nick spent hours selecting 50 tracks for a 4-hour radio broadcast entitled \"A Night At The Rum Runner\". The 18 track CD was released on 8 May 2006 and presented in a silver gatefold card sleeve in shocking pink metallic print featuring photographs taken from this period, first published in the book \"Duran Duran Unseen\" by Paul Edmond, the front cover photo being of fashion designer Patti Bell.
<Answer>: Duran Duran
<Thought>:
1. The question is asking about the band that Nick Rhodes was a member of. The words "Nick Rhodes" and "band" from the document refer to the question.
2. "Duran Duran" from the document refers to the answer.
3. The extracted words are "Nick Rhodes band Duran Duran"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "Nich Rhodes band",
    "answer_related_words": "Duran Duran",
    "extracted_words": "Nich Rhodes band Duran Duran"
}}
```

<Question>: Who has established herself as a Queen of Popular Music?
<Document>: Madonna (entertainer): Madonna's music has been the subject of much analysis and scrutiny. Robert M. Grant, author of Contemporary Strategy Analysis (2005), commented that what has brought Madonna success is \"certainly not outstanding natural talent. As a vocalist, musician, dancer, songwriter, or actress, Madonna's talents seem modest.\" He asserts Madonna's success is in relying on the talents of others, and that her personal relationships have served as cornerstones to the numerous reinventions in the longevity of her career. Madonna's approach was far from the music industry wisdom of \"Find a winning formula and stick to it.\" Her musical career has been a continuous experimentation with new musical ideas and new images and a constant quest for new heights of fame and acclaim. Grant concluded that \"having established herself as the queen of popular music, Madonna did not stop there, but continued re-inventing.\" Musicologist Susan McClary wrote that \"Madonna's art itself repeatedly deconstructs the traditional notion of the unified subject with finite ego boundaries. Her pieces explore various ways of constituting identities that refuse stability, that remain fluid, that resist definition.\"
<Answer>: Madonna
<Thought>:
1. The words "established herself" and "queen of popular music" from the document refer to the question.
2. "Madonna" from the document refers to the answer.
3. The extracted words are "established herself queen of popular music Madonna"
<JSON_OUTPUT>:
```json
{{
    "question_related_words": "established herself queen of popular music",
    "answer_related_words": "Madonna",
    "extracted_words": "established herself queen of popular music Madonna"
}}
```

Now it's your turn!

<Question>: {question}
<Document>: {paragraph}
<Answer>: {answer}
<Thought>:
""".strip()

TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_WIKIMQA = """
You have been assigned an information extraction task.
Your mission is to extract the words from a given paragraph so that others(GPT3.5) can answer a question using only your extracted words.
Your extracted words should cover information from both the question and the answer, including entities (e.g. people, location, film) and core relations.
Your response should be in JSON format and include the following key:
- "extracted_words": a string composed of a list of words extracted from the paragraph, separated by a space.

Please adhere to the following guidelines:
- Do not reorder, change, miss, or add words. Keep it the same as the original paragraph.
- Identify and extract ONLY the words explicitly mentioned in either the question or its answer, and strongly related to the question or its answer.
- NEVER label any words that do not contribute meaningful information to the question or answer.
- Only extract words that occured in the paragraph.
- Extract as few words as possible.

Question: Who is the director of film Polish-Russian War (Film)?
Paragraph: Polish-Russian War (film): Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery \u017bu\u0142awski based on the novel Polish-Russian War under the white-red flag by Dorota Mas\u0142owska.
Answer: Xawery \u017bu\u0142awski
Your response:
```json
{{"extracted_words": "Alas Southeast Aceh Regency Aceh Indonesia"}}
```

Question: When did Elio Petri die?
Paragraph: Elio Petri: Elio Petri( 29 January 1929 \u2013 10 November 1982) was an Italian political filmmaker best known for the 1970 Academy Award- winning film\" Investigation of a Citizen Above Suspicion\".
Answer: 10 November 1982
Your response:
```json
{{"extracted_words": "Elio Petri 10 November 1982"}}
```

Question: When was The Ballad of Josie released?
Paragraph: The Ballad of Josie: The Ballad of Josie is a 1967 Technicolor American comedy western film directed by Andrew V. McLaglen and starring Doris Day, Peter Graves and George Kennedy. It humorously tackles 1960s themes of feminism in a traditional Western setting. The film featured the last acting role for William Talman. It was filmed on two locations in Thousand Oaks, California: North Ranch and Wildwood Regional Park.
Answer: 1967
Your response:
```json
{{"extracted_words": "The Ballad of Josie 1967"}}
```

Question: {question}
Paragraph: {paragraph}
Answer: {answer}
Your response:
""".strip()

TOKEN_LABEL_REDUNDANT_SYSTEM_MSG = """
You are an outstanding linguist, adept at evaluation if the answer is redundant or not.
""".strip()

TOKEN_LABEL_REDUNDANT_EVALUATION_PROMPT = """
You have been assigned an information evaluation task.
Your mission is to evaluate if the extracted words contain enough information to answer the question, and if the extracted words contains redundant.
I will provide you with the question, the answer and the extracted words. You should check if the extracted words contain irrelevant information, or if the extracted words missed any important information.

Your response should be in JSON format and contain the following key:
- "redundant": a boolean value indicating if the extracted words contain too much redundant information.
- "missing": a boolean value indicating if the extracted words missed any important information.

# Examples

Question: Where is the Alas found?
Answer: Indonesia
Extracted Words: Alas Southeast Aceh Regency Aceh Indonesia
You should response:
{{"redundant": false, "missing": false}}
Explanation: Alas is found in Southeast Aceh Regency, Aceh, Indonesia. The extracted words are relevant to the question and answer.

Question: What is the meaning of Hindu in the Arabic dictionary?
Answer: the country of India
Extracted Words: Arabic Hind country of India 8th century text Chachnama
You should response:
{{"redundant": true, "missing": false}}
Explanation: The words "8th century text Chachnama" are irrelevant to the question.

Question: Who is the performer of Green?
Answer: Steve Hillage
Extracted Words: Ron Ehrenreich Vice - presidential Socialist Party USA election Willa Kenoyer New Syracuse Federal Credit Green Party Sondra Roth
You should response:
{{"redundant": true, "missing": true}}
Explanation: The extracted words are irrelevant to the question, and the performer of Green is not included in the extracted words.

# Task

Question: {question}
Answer: {answer}
Extracted Words: {extracted_words}
Your response:
""".strip()
