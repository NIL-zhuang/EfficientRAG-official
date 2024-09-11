QUERY_LABEL_SYSTEM_PROMPT = """
You are an outstanding linguist, and very good at refactoring questions by eliminating known information and focusing on the unknown part.
""".strip()

INFO_TEMPLATE = "<Known Info>: {info}"

SUB_ANSWER_TEMPLATE = "Must Include: {sub_answer}"

QUERY_LABEL_AFTER_ORIGINAL_QUESTION_MUSIQUE = """
You are assigned a multi-hop question rephrasing task.
We will show you the original multi-hop question(starting with <Question>) and the known information(starting with <Known Info>). The known information is composed of several words, where some of the words refer to a single-hop question which is a part of the original multi-hop question, and the other words are the answer to the single-hop question.
You should think step by step following the guidelines:
1. Identify the single-hop question which is part of the multi-hop question, and rephrase the question by removing part of the multi-hop question referring to the single-hop question.
2. Identify the answer to the single-hop question
3. Rephrase the original multi-hop question. You should remove the part of sentence referring to the single-hop question identified in step 1, and concatenate the answer identified in step 2 in the end.
You can only generate the single-hop question, answer, and rephrased question by extracting and removing words from the question and known information. You are not allowed to add, change, or reorder words. Keep the words same as the original multi-hop question and known information.

You answer should compose <Thought> part and final answer in JSON format starting with <JSON_OUTPUT>
Your <Thought> should consist two parts:
1. The single-hop question as part of the multi-hop question, and the rephrased question without the single-hop question part.
2. The answer to the single-hop question
And then, you should generate the rephrased question in JSON format embraced in code block ```json and ```,  consisting the following key:
- "filtered_query": a string representing the concatenation of the words from both the question and newly added information, separated by a space.

Here are some examples for you to refer to:

<Question>: When was the institute that owned The Collegian founded?
<Known Info>: The Collegian Houston Baptist University
<Thought>:
1. <Known Info> refers "The Collegian", it refers to the single-hop question part "the institute that owned The Collegian". So the rephrased question without the single-hop question part is "When was founded ?"
2. The answer to the single-hop question is "Houston Baptist University"
<JSON_OUTPUT>:
```json
{{"filtered_query": "When was founded ? Houston Baptist University"}}
```

<Question>: When was the abolishment of the studio that distributed The Game?
<Known Info>: The Game PolyGram Filmed Entertainment
<Thought>:
1. <Known Info> refers "The Game", it refers to the single-hop question part "the studio that distributed The Game". So the rephrased question without the single-hop question part is "When was the abolishment of ?"
2. The answer to the single-hop question is "PolyGram Filmed Entertainment"
<JSON_OUTPUT>:
```json
{{"filtered_query": "When was the abolishment of ? PolyGram Filmed Entertainment"}}
```

<Question>: What type of building in the EDSA Shangri-La in the city where Bartolome Ramos was educated?
<Known Info>: Bartolome Far Eastern University
<Thought>:
1. <Known Info> refers "Bartolome", it refers to the single-hop question "where Bartolome Ramos was educated". So the rephrased question without the single-hop question part is "What type of building in the EDSA Shangri-La in the city ?"
2. The answer to the single-hop question is "Far East University"
<JSON_OUTPUT>:
```json
{{"filtered_query": "What type of building in the EDSA Shangri-La in the city ? Far Eastern University"}}
```

<Question>: Who fathered the leader of the first expedition to Asia that sailed west across the ocean containing the island where Six Days Seven Nights was filmed?
<Known Info>: Six Days Seven Nights filmed Kauai
<Thought>:
1. <Known Info> refers "Six Days Seven Nights filmed", it refers to the single-hop question "the island where Six Days Seven Nights was filmed". So the rephrased question without the single-hop question part is "Who fathered the leader of the first expedition to Asia that sailed west across the ocean containing ?"
2. The answer to the single-hop question is "Kauai"
<JSON_OUTPUT>:
```json
{{"filtered_query": "Who fathered the leader of the first expedition to Asia that sailed west across the ocean containing ? Kauai"}}
```

<Question>: What's the most popular sport in the country that provided the most legal immigrants in 2013 in the continent for the country that won the 2002 World Cup in Japan?
<Known Info>: 2002 FIFA World Cup Japan Brazil
<Thought>:
1. <Known Info> refers "2002 FIFA World Cup Japan", it refers to the single-hop question "the country that won the 2002 World Cup in Japan". So the rephrased question without the single-hop question part is "What's the most popular sport in the country that provided the most legal immigrants in 2013 in the continent for ?"
2. The answer to the single-hop question is "Brazil"
<JSON_OUTPUT>:
```json
{{"filtered_query": "What's the most popular sport in the country that provided the most legal immigrants in 2013 in the continent for ? Brazil"}}
```

<Question>: What is the corporation tax rate in the country that, along with the nation that produces most of its own oil, recognized Gaddafi's government at an early date?
<Known Info>: U.S.
<Thought>:
1. <Known Info> U.S. refers a nation, it refers to the nation that produces oil, which is the single-hop question "the nation that produces most of its own oil"
2. The answer to the single-hop question is "U.S."
<JSON_OUTPUT>:
```json
{{"filtered_query": "What is the corporation tax rate in the country that, along with recognized Gaddafi's government at an early date ? U.S."}}
```

Now your question and reference known info are as follows.

<Question>: {question}
{info_list}
<Thought>:
""".strip()

QUERY_LABEL_AFTER_FILTERED_QUESTION_MUSIQUE = """
You are assigned a multi-hop question rephrasing task.
We will show you the original multi-hop question(starting with <Question>) and the known information(starting with <Known Info>). The known information is composed of several words, where some of the words refer to a single-hop question which is a part of the original multi-hop question, and the other words are the answer to the single-hop question.
You should think step by step following the guidelines:
1. Identify the single-hop question which is part of the multi-hop question, and rephrase the question by removing part of the multi-hop question referring to the single-hop question.
2. Identify the answer to the single-hop question
3. Rephrase the original multi-hop question. You should remove the part of sentence referring to the single-hop question identified in step 1, and concatenate the answer identified in step 2 in the end.
You can only generate the single-hop question, answer, and rephrased question by extracting and removing words from the question and known information. You are not allowed to add, change, or reorder words. Keep the words same as the original multi-hop question and known information.

You answer should compose <Thought> part and final answer in JSON format starting with <JSON_OUTPUT>
Your <Thought> should consist two parts:
1. The single-hop question as part of the multi-hop question, and the rephrased question without the single-hop question part.
2. The answer to the single-hop question
And then, you should generate the rephrased question in JSON format embraced in code block ```json and ```,  consisting the following key:
- "filtered_query": a string representing the concatenation of the words from both the question and newly added information, separated by a space.

Here are some examples for you to refer to:

<Question>: Where did the spouse of die ? Carl Nielsen
<Known Info>: Carl Nielsen Anne Marie Carl - Nielsen
<Thought>:
1. <Known Info> refers "Carl Nielsen", it refers to the single-hop question part "the spouse of Carl Nielsen". So the rephrased question without the single-hop question part is "Where did die ?"
2. The answer to the single-hop question is "Anne Marie Carl - Nielsen"
<JSON_OUTPUT>:
```json
{{"filtered_query": "Were did die ? Anne Marie Carl - Nielsen"}}
```

<Question>: What county shares a border with the area that houses ? Brockton Massachusetts
<Known Info>: Brockton Plymouth Country Massachusetts
<Thought>:
1. <Known Info> refers "Brockton Massachusetts", it refers to the single-hop question part "the area that houses Brockton Massachusetts". So the rephrased question without the single-hop question part is "What country shares a border with ?"
2. The answer to the single-hop question is "Plymouth Country"
<JSON_OUTPUT>:
```json
{{"filtered_query": "What country shares a border with ? Plymouth Country"}}
```

<Question>: What weekly publication in the city is issued by the school attended by ? David Gelernter New Haven
<Known Info>: David Gelernter Yale University
<Thought>:
1. <Known Info> refers "David Gelernter", it refers to the single-hop question "the school attended by David Gelernter". So the rephrased question without the single-hop question part is "What weekly publication in the city is issued by ? New Haven"
2. The answer to the single-hop question is "Yale University"
<JSON_OUTPUT>:
```json
{{"filtered_query": "What weekly publication in the city is issued by ? New Haven Yale University"}}
```

Now your question and reference known info are as follows.

<Question>: {question}
{info_list}
<Thought>:
""".strip()


QUERY_LABEL_FROM_MULTI_SOURCE_MUSIQUE = """
You are assigned a multi-hop question rephrasing task.
We will show you the original multi-hop question(starting with <Question>) and the known information(starting with <Known Info>). The known information is composed of several words, where some of the words refer to a single-hop question which is a part of the original multi-hop question, and the other words are the answer to the single-hop question.
You should think step by step following the guidelines:
1. Identify the single-hop question which is part of the multi-hop question, and rephrase the question by removing part of the multi-hop question referring to the single-hop question.
2. Identify the answer to the single-hop question
3. Rephrase the original multi-hop question. You should remove the part of sentence referring to the single-hop question identified in step 1, and concatenate the answer identified in step 2 in the end.
You may encounter several <Known Info>, handle them one by one.
You can only generate the single-hop question, answer, and rephrased question by extracting and removing words from the question and known information. You are not allowed to add, change, or reorder words. Keep the words same as the original multi-hop question and known information.

You answer should compose <Thought> part and final answer in JSON format starting with <JSON_OUTPUT>
Your <Thought> should consist two parts:
1. The single-hop question as part of the multi-hop question, and the rephrased question without the single-hop question part.
2. The answer to the single-hop question
And then, you should generate the rephrased question in JSON format embraced in code block ```json and ```,  consisting the following key:
- "filtered_query": a string representing the concatenation of the words from both the question and newly added information, separated by a space.

Here are some examples for you to refer to:

<Question>: When was the last time Jose Dominguez's sports team beat the winner of the 1894-95 FA Cup?
<Known Info>: 1894–95 FA Cup Aston Villa
<Known Info>: José Dominguez player England Birmingham City
<Thought>:
The first <Known Info> refers "1894-95 FA Cup", it refers to the single-hop question "the winner of the 1894-95 FA Cup". The answer to the single-hop question is "Aston Villa".
The second <Known Info> refers "José Dominguez", it refers to the single-hop question "Jose Dominguez's sports teams". The answer to the single-hop question is "England Birmingham City".
So the original multi-hop question without these single-hop questions is "When was the last time beat ?".
<JSON_OUTPUT>:
```json
{{"filtered_query": "When was the last time beat ? Aston Villa England Birmingham City"}}
```

<Question>: Who started the Bethel branch of the religion founded by the black community in the city that used to be the US capitol?
<Known Info>: black community African Methodist Episcopal Church
<Known Info>: Capitol Philadelphia
<Thought>:
The first <Known Info> refers "black community", it refers to the single-hop question "the religion founded by the black community". The answer to the single-hop question is "African Methodist Episcopal Church".
The second <Known Info> refers "Capitol", it refers to the single-hop question "the city that used to be the US capitol". The answer to the single-hop question is "Philadelphia".
So the original multi-hop question without these single-hop questions is "Who started the Bethel branch of in the city ?".
<JSON_OUTPUT>:
```json
{{"filtered_query": "Who started the Bethel branch of in the city ? African Methodist Episcopal Church Philadelphia"}}
```

<Question>: How many square miles is the source of the most legal immigrants to the location of Gotham's filming from the region where Andy from The Office sailed to?
<Known Info>: Gotham New York City
<Known Info>: Andy Caribbean
<Thought>:
The first <Known Info> refers "Gotham", it refers to the single-hop question "the location of Gotham's filming". The answer to the single-hop question is "New York City".
The second <Known Info> refers "Andy", it refers to the single-hop question "the region where Andy from The Office sailed to". The answer to the single-hop question is "Caribbean".
So the original multi-hop question without these single-hop questions is "How many square miles in the source of the most legal immigrants to the location of from ?".
<JSON_OUTPUT>:
```json
{{"filtered_query": "How many square miles in the source of the most legal immigrants to the location of from ? New York City Caribbean"}}
```

<Question>: When did the capital of Virginia move from John Nicholas' birthplace to the city that shares a border with Laurel?
<Known Info>: Laurel Henrico County Virginia
<Known Info>: John Nicholas Williamsburg Virginia
<Thought>:
The first <Known Info> refers "Laurel" and the location of Laurel, which refers to the single-hop question "Laurel". The answer to the single-hop question is "Henrico Country Virginia".
The second <Known Info> refers "John Nicolas", which refers to the single-hop question "John Nicholas' birthplace". The answer to the single-hop question is "Williamsburg Virginia".
So the original multi-hop question without these single-hop questions is "When did the capital of Virginia move from to the city that shares a border with ?".
<JSON_OUTPUT>
```json
{{"filtered_query": "When did the capital of Virginia move from to the city that shares a border with ? Henrico Country Virginia Williamsburg Virginia"}}
```

<Question>: {question}
{info_list}
<Thought>:
""".strip()

QUERY_LABEL_COMPOSITIONAL_2WIKIMQA = """
You are assigned a multi-hop question refactoring task.
Given a complex question along with a set of related known information, you are required to refactor the question by applying the principle of retraining difference and removing redundancies. Specificaly, you should eliminate the content that is duplicated between the question and the known information, leaving only the parts of the question that have not been answered, and the new knowledge points in the known information. The ultimate goal is to reorganize these retrained parts to form a new question.
You can only generate the question by picking words from the question and known information. You should first pick up words from the question, and then from each known info, and concatenate them fianlly. You are not allowed to add, change, or reorder words. The given knwon information starts with the word "Info: ".
You response should be in JSON format and include the following key:
- "filtered_query": a string representing the concatenation of the words from both the question and newly added information, separated by a space.

Please adhere to the following guidelines:
- Do not reorder, change, or add words. Keep it the same as the original question.
- Identify and remove ONLY the words that are already known, keep the unknown infomation from both the question and information.

Questions: Who is the mother of the director of film Polish-Russian War (Film)?
<Known Info> Polish - Russian War directed Xawery Żuławski
Your response:
```json
{{"filtered_query": "Who is the mother of Xawery Żuławski"}}
```

Questions: When did John V, Prince Of Anhalt-Zerbst's father die?
<Known Info> John V Prince Anhalt - Zerbst Ernest I Prince Anhalt - Dessau
Your response:
```json
{{"filtered_query": "When did die Ernest I Prince Anhalt - Dessau"}}
```

Questions: When is the director of film Pretty Clothes 's birthday?
<Known Info> Pretty Clothes directed Phil Rosen
Your response:
```json
{{"filtered_query": "When is the director 's birthday Phil Rosen"}}
```

Questions: Who is Alys Of France, Countess Of Vexin's maternal grandmother?
<Known Info> Alys of France daughter Constance of Castile
Your response:
```json
{{"filtered_query": "Who is maternal grandmother Constance of Castile"}}
```

Questions: {question}
{info_list}
Your response:
""".strip()

QUERY_LABEL_BRIDGE_COMPARISON_2WIKIMQA = """
You are assigned a multi-hop question refactoring task.
Given a complex question along with a set of related known information, you are required to refactor the question by applying the principle of retraining difference and removing redundancies. Specificaly, you should eliminate the content that is duplicated between the question and the known information, leaving only the parts of the question that have not been answered, and the new knowledge points in the known information. The ultimate goal is to reorganize these retrained parts to form a new question.
You can only generate the question by picking words from the question and known information. You should first pick up words from the question, and then from each known info, and concatenate them fianlly. You are not allowed to add, change, or reorder words. The given knwon information starts with the word "Info: ".
You response should be in JSON format and include the following key:
- "filtered_query": a string representing the concatenation of the words from both the question and newly added information, separated by a space.

Please adhere to the following guidelines:
- Do not reorder, change, or add words. Keep it the same as the original question.
- Identify and remove ONLY the words that are already known, keep the unknown infomation from both the question and information.

Questions: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
<Known Info> Crimen a las tres Luis Saslavsky
<Known Info> The Working Class Goes to Heaven Elio Petri
Your response:
```json
{{"filtered_query": "Which film has the director died first, Luis Saslavsky Elio Petri"}}
```

Questions: Do both films No Rest for the Wicked (film) and Voodoo Woman have the directors from the same country?
<Known Info> Voodoo Woman directed Edward L. Cahn
<Known Info> No Rest for the Wicked directed Enrique Urbizu
Your response:
```json
{{"filtered_query": "Do both films have the directors from the same country Edward L. Cahn Enrique Urbizu"}}
```

Questions: Which film has the director born first, Hell'S Belles (Film) or The City Of Youth?
<Known Info> The City of Youth directed E. H. Calvert
<Known Info> Hell's Belles directed Maury Dexter
Your response:
```json
{{"filtered_query": "Which film has the director born first E. H. Calvert Maury Dexter"}}
```

Questions: {question}
{info_list}
Your response:
""".strip()
