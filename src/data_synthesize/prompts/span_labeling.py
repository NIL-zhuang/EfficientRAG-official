SPAN_LABELING_SYSTEM_PROMPT = """
You are an outstanding linguistic, and very good at identifying information following the user instructions.
""".strip()

SPAN_LABELING_PROMPT = """
You are assigned an information labeling task.
We will show you a multi-hop question, a single-hop question, a document and the answer to the single-hop question. You should identify the span in the document that contains the answer to the single-hop question, and the span in the multi-hop question that contains the single-hop question.
You should embrace the span from the multi-hop question with <q-span> and </q-span> tags, containing the single-hop question. And the span from the document with <a-span> and </a-span> tags, containing the answer to the single-hop question.
You first should think step by step, finding out the single-hop question span and corresponding answer, and then figure out the rephrased question, which represents the multi-hop question with the single-hop question span replaced by the answer span. Try to make the rephrased question more fluent by modifying the question span.
Your response should be in JSON format and include the following keys:
- "labeled_document": a string representing the document with the answer span embraced by <a-span> and </a-span>.
- "labeled_question": a string representing the multi-hop question with the single-hop question span embraced by <q-span> and </q-span>. If the multi-hop question share the same meaning with the single-hop question, embrace the whole question.

Please adhere to the following guidelines:
- Do not reorder, change, or add words. All the words from your response should be present in the document or the multi-hop question.
- You must label both the multi-hop question and the document.
- You must label ONLY ONE span in the document and the multi-hop question.

Multi-hop Question: "What does the name of the organization the Haiti national football team belongs to stand for?"
Single-hop Question: "What organization is the Haiti national football team a member of?"
Document: "2014 Kosovo v Haiti football match: Kosovo vs Haiti was the first international match involving the Kosovar national football team to be recognised by FIFA, and the first to take place within Kosovo. The match was an international friendly between representative teams from Kosovo and Haiti."
Answer: "FIFA"
Your response:
Thought: The single-hop question span should cover "organization the Haiti national football team belongs to", whose answer is FIFA. The rephrased question should be "What does FIFA stand for?", so the labeled question should be "What does <q-span>the name of the organization the Haiti national football team belongs to<q-span> stand for".
```json
{{
    "labeled_document": "2014 Kosovo v Haiti football match: Kosovo vs Haiti was the first international match involving the Kosovar national football team to be recognised by <a-span>FIFA</a-span>, and the first to take place within Kosovo. The match was an international friendly between representative teams from Kosovo and Haiti.",
    "labeled_question": "What does <q-span>the name of the organization the Haiti national football team belongs to</q-span> stand for?"
}}
```

Multi-hop Question: "What does FIFA stand for?"
Single-hop Question: "What is the meaning of FIFA?"
Document: "Switzerland: Swiss are fans of football and the national team is nicknamed the 'Nati'. The headquarters of the sport's governing body, the International Federation of Association Football (FIFA), is located in Z\u00fcrich. Switzerland hosted the 1954 FIFA World Cup, and was the joint host, with Austria, of the Euro 2008 tournament. The Swiss Super League is the nation's professional club league. For the Brasil 2014 World Cup finals tournament, the country's German-speaking cantons will be closely monitored by local police forces to prevent celebrations beyond one hour after matches end. Europe's highest football pitch, at 2,000 metres (6,600 ft) above sea level, is located in Switzerland and is named the Ottmar Hitzfeld Stadium."
Answer: "International Federation of Association Football"
Your response:
Thought: The question is already a single-hop question, which shares the same meaning with the single-hop question. So we should embrace the whole queston. The labeled question should be "<q-span>What does FIFA stand for?</q-span>".
```json
{{
    "labeled_document": "Switzerland: Swiss are fans of football and the national team is nicknamed the 'Nati'. The headquarters of the sport's governing body, the <a-span>International Federation of Association Football</a-span> (FIFA), is located in Z\u00fcrich. Switzerland hosted the 1954 FIFA World Cup, and was the joint host, with Austria, of the Euro 2008 tournament. The Swiss Super League is the nation's professional club league. For the Brasil 2014 World Cup finals tournament, the country's German-speaking cantons will be closely monitored by local police forces to prevent celebrations beyond one hour after matches end. Europe's highest football pitch, at 2,000 metres (6,600 ft) above sea level, is located in Switzerland and is named the Ottmar Hitzfeld Stadium.",
    "labeled_question": "<q-span>What does FIFA stand for?</q-span>"
}}
```

Multi-hop Question: "The military group of which the Air Defense Artillery is a branch was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?"
Single-hop Question: "The Air Defense Artillery is a branch of what"
Document: "United States Army: Currently, the army is divided into the Regular Army, the Army Reserve, and the Army National Guard. The army is also divided into major branches such as Air Defense Artillery, Infantry, Aviation, Signal Corps, Corps of Engineers, and Armor. Before 1903 members of the National Guard were considered state soldiers unless federalized (i.e., activated) by the President. Since the Militia Act of 1903 all National Guard soldiers have held dual status: as National Guardsmen under the authority of the governor of their state or territory and, when activated, as a reserve of the U.S. Army under the authority of the President."
Answer: "the US Army"
Your response:
Thought: The single-hop question span should cover "the Air Defense Artillery is a branch of what", whose answer is "the US Army". The answer "the US Army" is corresponding to "United States Army" from the document. So the rephrased question should be "The military group of United States Army was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?". The labeled question should be "The military group of <q-span>which the Air Defense Artillery is a branch</q-span> was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?".
```json
{{
    "labeled_document": "<a-span>United States Army</a-span>: Currently, the army is divided into the Regular Army, the Army Reserve, and the Army National Guard. The army is also divided into major branches such as Air Defense Artillery, Infantry, Aviation, Signal Corps, Corps of Engineers, and Armor. Before 1903 members of the National Guard were considered state soldiers unless federalized (i.e., activated) by the President. Since the Militia Act of 1903 all National Guard soldiers have held dual status: as National Guardsmen under the authority of the governor of their state or territory and, when activated, as a reserve of the U.S. Army under the authority of the President.",
    "labeled_question": "The military group of <q-span>which the Air Defense Artillery is a branch</q-span> was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?"
}}
```

Multi-hop Question: "The military group of United States Army was unprepared for the invasion of Czechoslovakia. The country of this group was the only communist country to have an embassy where?"
Single-hop Question: "What's the country of the Army that was unprepared for the invasion of Czechoslovakia?"
Document: "Josip Broz Tito: In 1968, Tito offered Czechoslovak leader Alexander Dub\u010dek to fly to Prague on three hours notice if Dub\u010dek needed help in facing down the Soviets. In April 1969, Tito removed generals Ivan Go\u0161njak and Rade Hamovi\u0107 in the aftermath of the invasion of Czechoslovakia due to the unpreparedness of the Yugoslav army to respond to a similar invasion of Yugoslavia."
Answer: "Yugoslavia"
Your response:
Thought: The single-hop question span should cover "The country of the Army unprepared for the invasion of Czechoslovakia?", whose answer is "Yugoslavia". So the rephrased question should be "Yugoslavia was the only communist country to have an embassy where?". The labeled question should be "<q-span>The military group of United States Army was unprepared for the invasion of Czechoslovakia. The country of this group</q-span> was the only communist country to have an embassy where?".
```json
{{
    "labeled_document": "Josip Broz Tito: In 1968, Tito offered Czechoslovak leader Alexander Dub\u010dek to fly to Prague on three hours notice if Dub\u010dek needed help in facing down the Soviets. In April 1969, Tito removed generals Ivan Go\u0161njak and Rade Hamovi\u0107 in the aftermath of the invasion of Czechoslovakia due to the unpreparedness of the Yugoslav army to respond to a similar invasion of <a-span>Yugoslavia</a-span> .",
    "labeled_question": "<q-span>The military group of United States Army was unprepared for the invasion of Czechoslovakia. The country of this group</q-span> was the only communist country to have an embassy where?"
}}
```

Multi-hop Question: "{multi_hop_question}"
Single-hop Question: "{single_hop_question}"
Document: "{document}"
Answer: "{answer}"
Your response:
""".strip()
