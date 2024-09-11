# Relevant Document
MuSiQueSupportingFactPrompt = """
<DocID>: #{question_id}
<Hint>: {sub_question}
<Sub-Answer>: {sub_answer}
""".strip()

# Q -> A -> B
MuSiQueCompose2HopPrompt = """
You are assigned a multi-hop question decomposition task.
Your mission is to decompose the original multi-hop question into a list of single-hop sub_questions, and such that you can answer each sub_question independently from each document.
We will show you the original multi-hop question(starting with <Question>). Then we will provide you with a list of document hints(embraced with <doc> and </doc>) that contain the evidence to answer the sub_questions.
Each document starts with the document id(starting with <DocID>), contains the hint of the sub_question(starting with <Hint>), and its corresponding sub_answer(starting with <Sub-Answer>).
You should think step by step, and synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.

Your answer should compose <Thought> part and final decomposition in JSON format starting with <JSON_OUTPUT> embraced by ```json and ```.
Your <Thought> part must contain the following steps:
1. Follow the <Hint> and identify each sub-question
2. Judge if the sub-question is sincere and appropriate to the original multi-hop question
3. If the sub-question is not appropriate, you should modify it to be sincere to the original multi-hop question.

Your <JSON_OUTPUT> must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.

Here are some examples for you to refer to:

<Question>: What does the name of the organization the Haiti national football team belongs to stand for?
<doc>
<DocID>: #1
<Hint>: Haiti national football team >> member of
<Sub-Answer>: FIFA
<DocID>: #2
<Hint>: What does #1 stand for?
<Sub-Answer>: International Federation of Association Football
</doc>
<Thought>:
1. The first sub_question is "What organization is the Haiti national football team a member of?", and the corresponding answer is "FIFA". The second sub_question is "What does FIFA stand for?", and the corresponding answer is "International Federation of Association Football".
2. Both sub_questions are sincere to the original multi-hop question.
<JSON_OUTPUT>:
```json
{{
    "question": "What does the name of the organization the Haiti national football team belongs to stand for?",
    "decomposed_questions":{{
        "1": {{
            "sub_question": "What organization is the Haiti national football team a member of?",
            "answer": "FIFA",
            "dependency": []
        }},
        "2": {{
            "sub_question": "What does FIFA stand for?",
            "answer": "International Federation of Association Football",
            "dependency': ["1"]
        }}
    }}
}}
```

<Question>: When was the band Nick Rhodes was a member of established?
<doc>
<DocID>: #1
<Hint>: Nick Rhodes >> member of
<Sub-Answer>: Duran Duran
<DocID>: #2
<Hint>: When was #1 established?
<Sub-Answer>: 1978
</doc>
<Thought>:
1. The first sub_question is "What band was Nick Rhodes a member of?", and the corresponding answer is "Duran Duran". The second sub_question is "When was Duran Duran established?", and the corresponding answer is "1978".
2. Both sub_questions are sincere to the original multi-hop question.
<JSON_OUTPUT>:
```json
{{
    "question": "When was the band Nick Rhodes was a member of established?",
    "decomposed_questions":{{
        "1": {{
            "sub_question": "What band was Nick Rhodes a member of?",
            "answer": "Duran Duran",
            "dependency": []
        }},
        "2": {{
            "sub_question": "When was Duran Duran established?",
            "answer": "1978",
            "dependency": ["1"]
        }}
    }}
}}
```

<Question>: Who is the author of the biography of the star that established herself as a Queen of Popular Music?
<doc>
<DocID>: #1
<Hint>: Who has established herself as a Queen of Popular Music?
<Sub-Answer>: Madonna
<DocID>: #2
<Hint>: #1 >> author
<Sub-Answer>: Andrew Morton
</doc>
<Thought>:
1. The first sub_question is "Who has established herself as a Queen of Popular Music?", and the corresponding answer is "Madonna". The second sub_question is "Who is the author of the biography of Madonna?", and the corresponding answer is "Andrew Morton".
2. The first sub-question is not sincere to the original multi-hop question, so I modify it to "Who is the star that established herself as a Queen of Popular Music?". And the second sub-question is sincere to the original multi-hop question.
<JSON_OUTPUT>:
```json
{{
    "question": "Who is the author of the biography of the star that established herself as a Queen of Popular Music?",
    "decomposed_questions":{{
        "1": {{
            "sub_question": "Who is the star that established herself as a Queen of Popular Music?",
            "answer": "Madonna",
            "dependency": []
        }},
        "2": {{
            "sub_question": "Who is the author of the biography of Madonna?",
            "answer": "Andrew Morton",
            "dependency": ["1"]
        }}
    }}
}}
```

Now your question and reference information are as follows:

<Question>: {question}
<doc>
{decomposed_questions}
</doc>
<Thought>:
""".strip()

# Q -> A -> B -> C
MuSiQueCompose3HopPrompt = """
You are assigned a multi-hop question decomposition refactor task.
Your mission is to refactor the original decomposition of one multi-hop question into a list of single-hop sub_questions, and such that you (GPT-4) can answer each sub_question independently.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.
Your output must always be a JSON object only, do not explain yourself or output anything else.

The origin multi-hop questions is: Where did the spouse of Moderen's composer die?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Moderen >> composer
sub_answer: Carl Nielsen

sub_question id: #2
sub_question description: #1 >> spouse
sub_answer: Anne Marie Carl-Nielsen

sub_question id: #3
sub_question description: In what place did #2 die?
sub_answer: Copenhagen

Your response:
{{
    'question': 'Where did the spouse of Moderen's composer die?',
    'decomposed_questions':{{
        '1': {{
            'sub_question': "Who is Modern's composer?",
            'answer': 'Carl Nielsen',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Who is the spouse of Carl Nielsen?',
            'answer': 'Anne Marie Carl-Nielsen',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'In what place did Anne Marie Carl-Nielsen die?',
            'answer': 'Copenhagen',
            'dependency': ['2']
        }}
    }}
}}

The origin multi-hop questions is: When was the creation of the record label that the performer of The Galaxy Kings belongs to?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: The Galaxy Kings >> performer
sub_answer: Bob Schneider

sub_question id: #2
sub_question description: #1 >> record label
sub_answer: Kirtland Records

sub_question id: #3
sub_question description: When was #2 created?
sub_answer: 2003

Your response:
{{
    'question': 'When was the creation of the record label that the performer of The Galaxy Kings belongs to?',
    'decomposed_questions':{{
        '1': {{
            'sub_question': 'Who is the performer of The Galaxy Kings?',
            'answer': 'Bob Schneider',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What is the record label that Bob Schneider belongs to?',
            'answer': 'Kirtland Records',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'When was the Kirtland Records created?',
            'answer': '2003',
            'dependency': ['2']
        }}
    }}
}}

The origin multi-hop questions is: When does Meet Me in the birthplace of From the Sky Down's director take place?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: From the Sky Down >> director
sub_answer: Davis Guggenheim

sub_question id: #2
sub_question description: What is the birthplace of #1 ?
sub_answer: St. Louis

sub_question id: #3
sub_question description: when does meet me in #2 take place
sub_answer: starting with Summer 1903

Your response:
{{
    'question': 'When does Meet Me in the birthplace of From the Sky Down's director take place?',
    'decomposed_questions':{{
        '1': {{
            'sub_question': "Who is From the Sky Down's director?",
            'answer': 'Davis Guggenheim',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What is the birthplace of Davis Guggenheim?',
            'answer': 'St. Louis',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'When does Meet Me in St. Louis take place?',
            'answer': '1903',
            'dependency': ['2']
        }}
    }}
}}

The origin multi-hop questions is: {question}
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.
{decomposed_questions}

Your response:
""".strip()

# Q -> A -> B -> C -> D
MuSiQueCompose4HopPrompt = """
You are assigned a multi-hop question decomposition refactor task.
Your mission is to refactor the original decomposition of one multi-hop question into a list of single-hop sub_questions, and such that you (GPT-4) can answer each sub_question independently.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question. NEVER introduce information from the documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.
Your output must always be a JSON object only, do not explain yourself or output anything else.

The origin multi-hop questions is: When do they hold elections for the house of the body providing oversight for David Vladeck's employer that has the power to introduce appropriation bills?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: David Vladeck >> employer
sub_answer: Federal Trade Commission

sub_question id: #2
sub_question description: Who has over-sight of #1 ?
sub_answer: Congress

sub_question id: #3
sub_question description: which house of #2 has the power to introduce appropriation bills
sub_answer: the House of Representatives

sub_question id: #4
sub_question description: when are elections for #3 held
sub_answer: November 6, 2018

Your response:
{{
    'question': "When do they hold elections for the house of the body providing oversight for David Vladeck's employer that has the power to introduce appropriation bills?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': "Who is David Vladeck's employer?",
            'answer': 'Federal Trade Commission',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Who has over-sight of the Federal Trade Commission?',
            'answer': 'Congress',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'Which house of Congress has the power to introduce appropriation bills?',
            'answer': 'the House of Representatives',
            'dependency': ['2']
        }},
        '4': {{
            'sub_question': "When are elections for the House of Representatives held?",
            'answer': 'November 6, 2018',
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: When was Way Down released by performer on the disc box set of live recordings in the birth city of the singer of If It Wasn't True?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: If It Wasn't True >> performer
sub_answer: Shamir

sub_question id: #2
sub_question description: #1 >> place of birth
sub_answer: Las Vegas

sub_question id: #3
sub_question description: Live in #2 >> performer
sub_answer: Elvis Presley

sub_question id: #4
sub_question description: when was way down by #3 released
sub_answer: August 16, 1977

Your response:
{{
    'question': "When was Way Down released by performer on the disc box set of live recordings in the birth city of the singer of If It Wasn't True?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': "Who is the singer of If It Wasn't True?",
            'answer': 'Shamir',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What is the birth city of Shamir?',
            'answer': 'Las Vegas',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'Who is the performer on the disc box set of Live in Las Vegas?',
            'answer': 'Elvis Presley',
            'dependency': ['2']
        }},
        '4': {{
            'sub_question': "When was Way Down by Elvis Presley released?",
            'answer': 'August 16, 1977',
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: Who fathered the person leading the first expedition to reach Asia by sailing west across the ocean having Ryler DeHeart's birthplace?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Ryler DeHeart >> place of birth
sub_answer: Kauai

sub_question id: #2
sub_question description: #1 >> located in or next to body of water
sub_answer: Pacific Ocean

sub_question id: #3
sub_question description: who led the first expedition to reach asia by sailing west across #2
sub_answer: Vasco da Gama

sub_question id: #4
sub_question description: Who fathered #3 ?
sub_answer: Estêvão da Gama

Your response:
{{
    'question': "Who fathered the person leading the first expedition to reach Asia by sailing west across the ocean having Ryler DeHeart's birthplace?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': "What is Ryler DeHeart's birthplace?",
            'answer': 'Kauai',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What is the ocean having Kauai',
            'answer': 'Pacific Ocean',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': 'Who lead the first expedition to reach Asia by sailing west across the Pacific Ocean?',
            'answer': 'Vasco da Gama',
            'dependency': ['2']
        }},
        '4': {{
            'sub_question': "Who fathered Vasco da Gama?",
            'answer': 'Estêvão da Gama',
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: {question}
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.
{decomposed_questions}

Your response:
""".strip()

# Q -> (A, B) -> C
MuSiQue3HopSeparateComposePrompt = """
You are assigned a multi-hop question decomposition refactor task.
Your mission is to refactor the original decomposition of one multi-hop question into a list of single-hop sub_questions, and such that you (GPT-4) can answer each sub_question independently.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.
Your output must always be a JSON object only, do not explain yourself or output anything else.

The origin multi-hop questions is: How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Star Cola is produced?
Follow the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: New coins were a proclamation of independence by the Somali Muslim Ajuran Empire from whom?
sub_answer: the Portuguese

sub_question id: #2
sub_question description: The country for Star Cola was what?
sub_answer: Myanmar

sub_question id: #3
sub_question description: How were the #1 expelled from #2 ?
sub_answer: The dynasty regrouped and defeated the Portuguese

Your response:
{{
    'question': 'How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Star Cola is produced?',
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'New coins were a proclamation of independence by the Somali Muslim Ajuran Empire from whom?',
            'answer': 'the Portuguese',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What was the country of Star Cola?',
            'answer': 'Myanmar',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'How were the the Portuguese expelled from Myanmar?',
            'answer': 'The dynasty regrouped and defeated the Portuguese',
            'dependency': ['1', '2']
        }}
    }}
}}

The origin multi-hop questions is: When did the people who received strong support in Posen come to the area where Baptist missionaries were active in the anti-slavery movement?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: What was there strong support of in Posen?
sub_answer: the French

sub_question id: #2
sub_question description: Where did Baptist missionaries take an active role in the anti-slavery movement?
sub_answer: the Caribbean

sub_question id: #3
sub_question description: when did the #1 come to the #2
sub_answer: 1625

Your response:
{{
    'question': 'When did the people who received strong support in Posen come to the area where Baptist missionaries were active in the anti-slavery movement?',
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'Who received strong support in Posen?',
            'answer': 'the French',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Where were Baptist missionaries active in the anti-slavery movement?',
            'answer': 'the Caribbean',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'When did the French come to the Caribbean?',
            'answer': '1625',
            'dependency': ['1', '2']
        }}
    }}
}}

The origin multi-hop questions is: Are other languages learned in the country that started the Battle of the Coral Sea, as popular as the language originating in the country with the world's oldest navy?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: who has the oldest navy in the world
sub_answer: The Spanish

sub_question id: #2
sub_question description: who started the battle of the coral sea
sub_answer: The U.S.

sub_question id: #3
sub_question description: Are these other languages learned in the #2 as popular as #1 ?
sub_answer: totals remain relatively small in relation to the total U.S population.

Your response:
{{
    'question': "Are other languages learned in the country that started the Battle of the Coral Sea, as popular as the language originating in the country with the world's oldest navy?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'Who has the oldest navy in the world?',
            'answer': 'The Spanish',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Who started the Battle of the Coral Sea?',
            'answer': 'The U.S.',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'Are other languages learned in the U.S. as popular as the Spanish?',
            'answer': 'totals remain relatively small in relation to the total U.S population.',
            'dependency': ['1', '2']
        }}
    }}
}}

The origin multi-hop questions is: {question}
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.
{decomposed_questions}

Your response:
""".strip()


# 4hop2
# Q -> (A, B) -> C -> D
MuSiQue4HopComposeBridgePrompt = """
You are assigned a multi-hop question decomposition refactor task.
Your mission is to refactor the original decomposition of one multi-hop question into a list of single-hop sub_questions, and such that you (GPT-4) can answer each sub_question independently.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.
Your output must always be a JSON object only, do not explain yourself or output anything else.

The origin multi-hop questions is: Hana Mandlikova was born in Country A that invaded Country B because the military branch the Air Defense Artillery is part of was unprepared. Country B was the only communist country to have an embassy where?
Follow the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Hana Mandlikova >> born place
sub_answer: Czechoslovakia

sub_question id: #2
sub_question description: The Air Defense Artillery is a branch of what?
sub_answer: the Army

sub_question id: #3
sub_question description: What #2 unprepared for the invasion of #1 ?
sub_answer: Yugoslavia

sub_question id: #4
sub_question description: #3 was the only communist country to have an embassy where?
sub_answer: Alfredo Stroessner's Paraguay

Your response:
{{
    'question': 'Hana Mandlikova was born in Country A that invaded Country B because the military branch the Air Defense Artillery is part of was unprepared. Country B was the only communist country to have an embassy where?',
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'Where was Hana Mandlikova born?',
            'answer': 'Czechoslovakia',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'The Air Defense Artillery is a branch of what?',
            'answer': 'the Army',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'What the Army was unprepared for the invasion of Czechoslovakia?',
            'answer': 'Yugoslavia',
            'dependency': ['1', '2']
        }},
        '4': {{
            'sub_question': 'Where was the only communist country to have an embassy in Yugoslavia?',
            'answer': "Alfredo Stroessner's Paraguay",
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: Who fathered the leader of the first expedition to reach Hanoi's continent by sailing west across the ocean bordering eastern Russia?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Hanoi >> continent
sub_answer: Asia

sub_question id: #2
sub_question description: Which ocean is along eastern Russia?
sub_answer: the Pacific Ocean

sub_question id: #3
sub_question description: who led the first expedition to reach #1 by sailing west across #2
sub_answer: Vasco da Gama

sub_question id: #4
sub_question description: Who fathered #3 ?
sub_answer: Estêvão da Gama

Your response:
{{
    'question': "Who fathered the leader of the first expedition to reach Hanoi's continent by sailing west across the ocean bordering eastern Russia?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': "What is Hanoi's continent?",
            'answer': 'Asia',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Which ocean is bordering eastern Russia?',
            'answer': 'the Pacific Ocean',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'Who is the leader of the first expedition to reach Asia by sailing west across the Pacific Ocean?',
            'answer': 'Vasco da Gama',
            'dependency': ['1', '2']
        }},
        '4': {{
            'sub_question': 'Who fatered Vasco da Gama?',
            'answer': 'Estêvão da Gama',
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: When was the death penalty abolished in the country that recognized the government of the person most closely associated with Libya's new government, along with the country the provides the most oil in the US?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: where does most of the oil in the us come from
sub_answer: the U.S.

sub_question id: #2
sub_question description: Whose face was most closely associated with Libya's new government?
sub_answer: Gaddafi

sub_question id: #3
sub_question description: Along with the #1 , what major power recognized #2 's government at an early date?
sub_answer: U.K.

sub_question id: #4
sub_question description: when was the death penalty abolished in #3
sub_answer: 1998

Your response:
{{
    'question': 'When was the death penalty abolished in the country that recognized the government of the person most closely associated with Libya's new government, along with the country the provides the most oil in the US?',
    'decomposed_questions': {{
        '1': {{
            'sub_question': "Which country provides the most oil in the US?",
            'answer': 'the U.S.',
            'dependency': []
        }},
        '2': {{
            'sub_question': "Who most closely associated with Libya's new government?",
            'answer': 'Gaddafi',
            'dependency': []
        }},
        '3': {{
            'sub_question': 'What country recognized Gaddafi, along with the U.S.?',
            'answer': 'U.K.',
            'dependency': ['1', '2']
        }},
        '4': {{
            'sub_question': 'When was the death penalty abolished in the U.K.?',
            'answer': '1998',
            'dependency': ['3']
        }}
    }}
}}

The origin multi-hop questions is: {question}
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.
{decomposed_questions}

Your response:
""".strip()

# 4hop3
# Q -> A -> B, Q -> C, (B, C) -> D


MuSiQueAsymmetryBridgePrompt = """
You are assigned a multi-hop question decomposition refactor task.
Your mission is to refactor the original decomposition of one multi-hop question into a list of single-hop sub_questions, and such that you (GPT-4) can answer each sub_question independently.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the refactored single-hop sub_question. It should not contain any # tag, and the # tag must be replaced by the answer of the sub_question.The sub_question MUST be sincere to the original multi-hop question.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions(the sub_question has # tag referring to other sub_questions), you should list the sub_question number here. Leave it empty if the sub_question does not rely on any other sub_questions.
Your output must always be a JSON object only, do not explain yourself or output anything else.

The origin multi-hop questions is: Despite being located in East Belgium, the Carnival of the birth place of Guido Maus harks purely to an area. What was the language having the same name as this area of the era with Fastrada's spouse's name later known as?
Follow the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Guido Maus >> place of birth
sub_answer: Malmedy

sub_question id: #2
sub_question description: Despite being located in East Belgium, #1 's Carnival harks purely to what area?
sub_answer: Latin

sub_question id: #3
sub_question description: What is Fastrada's spouse's name?
sub_answer: Charlemagne

sub_question id: #4
sub_question description: What was the #2 of #3 's era later known as?
sub_answer: Medieval Latin

Your response:
{{
    'question': "Despite being located in East Belgium, the Carnival of the birth place of Guido Maus harks purely to an area. What was the language having the same name as this area of the era with Fastrada's spouse's name later known as?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'What is the place of birth of Guido Maus?',
            'answer': 'Malmedy',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'Despite being located in East Belgium, the Carnival of Malmedy harks purely to what area?',
            'answer': 'Latin',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': "What is Fastrada's spouse's name?",
            'answer': 'Charlemagne',
            'dependency': []
        }},
        '4': {{
            'sub_question': 'What was the Latin of Charlemagne's era later known as?',
            'answer': 'Medieval Latin',
            'dependency': ['2', '3']
        }}
    }}
}}

The origin multi-hop questions is: What weekly publication in the city where Steven Segaloff died is issued by the school attended by the author of America-Lite: How Imperial Academia Dismantled Our Culture?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: America-Lite: How Imperial Academia Dismantled Our Culture >> author
sub_answer: David Gelernter

sub_question id: #2
sub_question description: #1 >> educated at
sub_answer: Yale University

sub_question id: #3
sub_question description: Steven Segaloff >> place of birth
sub_answer: New Haven

sub_question id: #4
sub_question description: What weekly publication in #3 is issued by #2 ?
sub_answer: Yale Herald

Your response:
{{
    'question': "What weekly publication in the city where Steven Segaloff died is issued by the school attended by the author of America-Lite: How Imperial Academia Dismantled Our Culture?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'Who is the author of America-Lite: How Imperial Academia Dismantled Our Culture?',
            'answer': 'David Gelernter',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What is the school attended by David Gelernter?',
            'answer': 'Yale University',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': "What is the city where Steven Segaloff died?",
            'answer': 'New Haven',
            'dependency': []
        }},
        '4': {{
            'sub_question': 'What weekly publication in New Haven is issued by Yale University?',
            'answer': 'Yale Herald',
            'dependency': ['2', '3']
        }}
    }}
}}

The origin multi-hop questions is: A US legislature has over-sight of the government agency issuing a report describing the effects of antibiotic developments. What part of the supreme law not containing the phrase "Wall of separation" as pointed out by Stewart is the basis for the implied powers of this legislature?
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.

sub_question id: #1
sub_question description: Who issued a report describing the effects of antibiotic developments?
sub_answer: Federal Trade Commission

sub_question id: #2
sub_question description: Who has over-sight of #1 ?
sub_answer: Congress

sub_question id: #3
sub_question description: Stewart pointed out that the phrase "Wall of separation" was nowhere to be found in what?
sub_answer: the Constitution

sub_question id: #4
sub_question description: what part of the #3 is the basis for the implied powers of #2
sub_answer: ``general welfare clause ''and the`` necessary and proper clause''

Your response:
{{
    'question': "A US legislature has over-sight of the government agency issuing a report describing the effects of antibiotic developments. What part of the supreme law not containing the phrase \"Wall of separation\" as pointed out by Stewart is the basis for the implied powers of this legislature?",
    'decomposed_questions': {{
        '1': {{
            'sub_question': 'Which government agency issued a report describing the effects of antibiotic developments?',
            'answer': 'Federal Trade Commission',
            'dependency': []
        }},
        '2': {{
            'sub_question': 'What US legislature has over-sight of the Federal Trade Commission?',
            'answer': 'Congress',
            'dependency': ['1']
        }},
        '3': {{
            'sub_question': "What supreme law not containing the phrase 'Wall of separation' was pointed out by Stewart?",
            'answer': 'the Constitution',
            'dependency': []
        }},
        '4': {{
            'sub_question': 'What part of the Constitution is the basis for the implied powers of Congress?',
            'answer': "``general welfare clause ''and the`` necessary and proper clause''",
            'dependency': ['2', '3']
        }}
    }}
}}

The origin multi-hop questions is: {question}
Followings are the not properly decomposed sub_questions and sub_answers, synthesize the sub_questions and answers one-by-one. NEVER change the order or miss any of them.
The # tag in the sub_question means the answer of corresponding sub_question, you must replace them with the actual answer.
{decomposed_questions}

Your response:
""".strip()
