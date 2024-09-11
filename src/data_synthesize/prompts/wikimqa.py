WikiMQAFactPrompt = """
DocId: #{question_id}
Title: {doc_title}
Document: {facts}
Evidence: {evidence}
""".strip()

WikiMQAPromptComparison = """
You are assigned a multi-hop question decomposition task.
Your mission is to decompose the original multi-hop question into a list of single-hop sub_questions, based on supporting document for each sub_question, and such that you can answer each sub_question independently from each document. Each document infers a sub_question id which starts with `#`. The evidence in the document indicates the relation of two entities, in the form of `entity1 - relation - entity2`.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. It MUST NOT contain information more than the original question and its dependencies. NEVER introduce information from documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here. Leave it empty for now because the questions now are all comparison type.
    - "document": a string, the document id that supports the sub_question.
Notice that you don't need to come out the compare question, just the sub_questions and answers.

The original multi-hop question is: Which film came out first, The Love Route or Engal Aasan?
<document>
DocId: #1
Title: The Love Route
Document: The Love Route is a 1915 American Western silent film directed and written by Allan Dwan based upon a play by Edward Henry Peple. The film stars Harold Lockwood, Winifred Kingston, Donald Crisp, Jack Pickford, Dick La Reno, and Juanita Hansen. The film was released on February 25, 1915, by Paramount Pictures.
Evidence: The Love Route - publication data - 1915

DocId: #2
Title: Engal Aasan
Document: Engal Aasan is a 2009 Tamil action comedy- drama film directed by R. K. Kalaimani. The film starring Vijayakanth in the lead role and Vikranth, Sheryl Brindo, Akshaya and Suja Varunee playing supporting roles. It began its first schedule on 12 March 2008 and released in July 2009. The film, upon release could not release the big theatres and became a colossal flop. It was dubbed in Telugu as\" Captain\".
Evidence: Engal Aasan - publication date - 2009
</document>
Your response:
{{
    "question": "Which film came out first, The Love Route or Engal Aasan?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "When does the film The Love Route come out?",
            "answer": "1915",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "When does the film Engal Aasan come out?",
            "answer": "2009",
            "dependency": [],
            "document": "2"
        }}
    }}
}}

The original multi-hop question is: Are Matraville Sports High School and Wabash High School both located in the same country?
<document>
DocId: #1
Title: Matraville Sports High School
Document: Matraville Sports High School( abbreviated as MSHS) is a government co-educational comprehensive and specialist secondary school, with speciality in sports, located on Anzac Parade, Chifley, an eastern suburb of Sydney, New South Wales, Australia. Established in 1960 as Matraville High School, the school became a specialist high school in December 2001 and caters for approximately 300 students from Year 7 to Year 12. The school is operated by the New South Wales Department of Education; the principal is Nerida Walker. Its alumni include Bob Carr and a number of professional sportsmen and women, particularly rugby league players. Matraville Sports High School is a member of the NSW Sports High Schools Association.
Evidence: Matraville Sports High School - country - Australia

DocId: #2
Title: Wabash High School
Document: Wabash High School is a high school in Wabash, Indiana, United States with approximately 500 students in grades 9- 12. The nickname of the students and the athletic teams is\" Wabash Apaches.\"
Evidence: Wabash High School - country - United States
</document>
Your response:
{{
    "question": "Are Matraville Sports High School and Wabash High School both located in the same country?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Where does Waabash High School located?",
            "answer": "United States",
            "dependency": [],
            "document": "2"
        }},
        "2": {{
            "sub_question": "Where does Matraville Sports High School located?",
            "answer": "Australia",
            "dependency": [],
            "document": "1"
        }}
    }}
}}


The original multi-hop question is: Does Mukasa Mbidde have the same nationality as Erich Maas?
<document>
DocId: #1
Title: Mukasa Mbidde
Document: Fred Mukasa Mbidde( Born 15 October 1974) is a Ugandan lawyer, human-rights activist, mass communication specialist, motivational speaker and politician. He is an elected member of the 3rd East African Legislative Assembly( EALA), representing the Republic of Uganda. He has been in this office since June 2012. He serves on three EALA committees: the Committee on Communication, Trade and Investments; the Committee on Legal, Rules and Privileges; and the Committee on Regional Affairs and Conflict Resolution.
Evidence: Mukasa Mbidde - country of citizenship - Uganda

DocId: #2
Title: Erich Maas
Document: Erich Maas( born 24 December 1940 in Pr\u00fcm, Rhine Province) is a German former footballer. He spent eight seasons in the Bundesliga, as well as five seasons in the French Division 1, and was capped three times for the German national team.
Evidence: Erich Maas - country of citizenship - Germany
</document>
Your response:
{{
    "question": "Does Mukasa Mbidde have the same nationality as Erich Maas?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "What is Mukasa Mbidde's nationality?",
            "answer": "Uganda",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "What is Erich Maas's nationality?",
            "answer": "Germany",
            "dependency": [],
            "document": "2"
        }}
    }}
}}


The original multi-hop question is: {question}
<document>
{chunks}
</document>
Your response:
""".strip()

WikiMQAPromptInference = """
You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting document for each sub_question, and such that you (GPT-4) can answer each sub_question independently from each document. Each document infers a sub_question id which starts with `#`. The evidence in the document indicates the relation of two entities, in the form of `entity1 - relation - entity2`.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. It MUST NOT contain information more than the original question and its dependencies. NEVER introduce information from documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here.
    - "document": a string, the document id that supports the sub_question.

The origin multi-hop questions is: Who is Rhescuporis I (Odrysian)'s paternal grandfather?
<document>
DocId: #1
Title: Rhescuporis I (Odrysian)
Document: Rhescuporis I (Ancient Greek: \u03a1\u03b1\u03b9\u03c3\u03ba\u03bf\u03cd\u03c0\u03bf\u03c1\u03b9\u03c2) was a king of the Odrysian kingdom of Thrace in 240 BC - 215 BC, succeeding his father, Cotys III.
Evidence: Rhescuporis I - father - Cotys III

DocId: #2
Title: Cotys III (Odrysian)
Document: Cotys III (Ancient Greek: \u039a\u03cc\u03c4\u03c5\u03c2) was a king of the Odrysian kingdom of Thrace in ca. 270 BC, succeeding his father, Raizdos.
Evidence: Cotys III - father - Raizdos
</document>
Your response:
{{
    "question": "Who is Rhescuporis I (Odrysian)'s paternal grandfather?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is Rhesuporis I (Odrysian)'s father?",
            "answer": "Cotys III",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "Who is Cotys III's father?",
            "answer": "",
            "dependency": ["1"],
            "document": "2"
        }}
    }}
}}

The origin multi-hop questions is: Who is the mother-in-law of Andrew Murray (Scottish Soldier)?
<document>
DocId: #1
Title: Christina Bruce
Document: Christina Bruce (c. 1278 \u2013 1356/1357), also known as Christina de Brus, was a daughter of Marjorie, Countess of Carrick, and her husband, Robert de Brus, \"jure uxoris\" Earl of Carrick, as well as a sister of Robert the Bruce, King of Scots. It is presumed that she and her siblings were born at Turnberry Castle in Carrick. In 1326 he married Christina Bruce, a sister of King Robert I of Scotland. Murray was twice chosen as Guardian of Scotland, first in 1332, and again from 1335 on his return to Scotland after his release from captivity in England.
Evidence: Christina Bruce - mother - Marjorie, Countess of Carrick

DocId: #2
Title: Andrew Murray (Scottish soldier)
Document: Sir Andrew Murray (1298\u20131338), also known as Sir Andrew Moray, or Sir Andrew de Moray, was a Scottish military and political leader who supported David II of Scotland against Edward Balliol and King Edward III of England during the so-called Second War of Scottish Independence. He held the lordships of Avoch and Petty in north Scotland, and Bothwell in west-central Scotland. In 1326 he married Christina Bruce, a sister of King Robert I of Scotland. Murray was twice chosen as Guardian of Scotland, first in 1332, and again from 1335 on his return to Scotland after his release from captivity in England. He held the guardianship until his death in 1338.
Evidence: Sir Andrew Murray - spouse - Christina Bruce
</document>
Your response:
{{
    "question": "Who is the mother-in-law of Andrew Murray (Scottish Soldier)?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the spouse of Andrew Murray (Scottish Soldier)?",
            "answer": "Christina Bruce",
            "dependency": [],
            "document": "2"
        }},
        "2": {{
            "sub_question": "Who is the mother of Christina Bruce?",
            "answer": "Marjorie, Countess of Carrick",
            "dependency": ["1"],
            "document": "1"
        }}
    }}
}}


The origin multi-hop questions is: Who is the uncle of Manuel Komnenos (Kouropalates)?
<document>
DocId: #1
Title: Manuel Komnenos (kouropalates)
Document: Manuel Komnenos (\"Manou\u0113l Komn\u0113nos\"; \u2013 17 April 1071) was a Byzantine aristocrat and military leader, the oldest son of John Komnenos and brother of emperor Alexios I Komnenos. A relative by marriage of emperor Romanos IV Diogenes, he was placed in charge of expeditions against Turkish raids in 1070\u20131071, until his sudden death by illness in April 1071.
Evidence: Manuel Komnenos - father - John Komnenos

DocId: #2
Title: John Komnenos (Domestic of the Schools)
Document: John Komnenos (\"I\u014dann\u0113s Komn\u0113nos\"; \u2013 12 July 1067) was a Byzantine aristocrat and military leader. The younger brother of Emperor Isaac I Komnenos, he served as Domestic of the Schools during Isaac's brief reign (1057\u201359). When Isaac I abdicated, Constantine X Doukas became emperor and John withdrew from public life until his death in 1067. Through his son Alexios I Komnenos, who became emperor in 1081, he was the progenitor of the Komnenian dynasty that ruled the Byzantine Empire from 1081 until 1185, and the Empire of Trebizond from 1204 until 1461.
Evidence: John Komnenos - sibling - Isaac I Komnenos
</document>
Your response:
{{
    "question": "Who is the uncle of Manuel Komnenos (Kouropalates)?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the father of Manuel Komnenos (Kouropalates)?",
            "answer": "John Komnenos",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "Who is the sibling of John Komnenos?",
            "answer": "Isaac I Komnenos",
            "dependency": ["1"],
            "document": "2"
        }}
    }}
}}


The origin multi-hop questions is: {question}
<document>
{chunks}
</document>
Your response:
""".strip()

WikiMQAPromptCompositional = """
You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting document for each sub_question, and such that you (GPT-4) can answer each sub_question independently from each document. Each document infers a sub_question id which starts with `#`. The evidence in the document indicates the relation of two entities, in the form of `entity1 - relation - entity2`.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. It MUST NOT contain information more than the original question and its dependencies. NEVER introduce information from documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here.
    - "document": a string, the document id that supports the sub_question.

The origin multi-hop questions is: When is the composer of film Sruthilayalu 's birthday?
<document>
DocId: #1
Title: Sruthilayalu
Document: Sruthilayalu is a 1987 Indian Telugu-language musical drama film, written and directed by K. Viswanath. The film stars Rajasekhar and Sumalata with soundtrack composed by K. V. Mahadevan. The film garnered Nandi Awards for Best feature film; Best direction, and a Filmfare Award for Best Director \u2013 Telugu. The film was premiered at the International Film Festival of India, and AISFM Film Festival. The film was dubbed in Tamil as \"Isaikku Oru Koil\".
Evidence: Sruthilayalu - composer - K. V. Mahadevan

DocId: #2
Title: K. V. Mahadevan
Document: Krishnankoil Venkadachalam Mahadevan (14 March 1918 \u2013 21 June 2001) was an Indian composer, singer-songwriter, music producer, and musician known for his works in Tamil cinema, Telugu cinema, Kannada cinema, and Malayalam cinema. He is best known for his contributions in works such as \"Manchi Manasulu\" (1962), \"Lava Kusa\" (1963), \"Thiruvilaiyadal\" (1965), \"Saraswathi Sabatham\" (1966), \"Kandan Karunai\" (1967), \"Thillana Mohanambal\" (1968), \"Adimai Penn\" (1969), \"Balaraju Katha\" (1970), \"Athiparasakthi\" (1971), \"Sankarabharanam\" (1979), \"Saptapadi\" (1981), \"Sirivennela\" (1986), \"Sruthilayalu\" (1987), \"Pelli Pustakam\" (1991), and \"Swathi Kiranam\" (1992). A contemporary of M. S. Viswanathan and T. K. Ramamoorthy, starting his career in the 1942 with \"Manonmani\" , Mahadevan scored music for over six hundred feature films, spanning four decades, and has garnered two National Film Awards, the Tamil Nadu State Film Award for Best Music Director, three Nandi Awards for Best Music Director, and the Filmfare Best Music Director Award (Telugu). He was also conferred the title of Thirai Isai Thilagam (Pride of Cine Music Directors) in Tamil cinema.
Evidence: K. V. Mahadevan - date of birth - 14 March 1918
</document>
Your response:
{{
    "question": "When is the composer of film Sruthilayalu 's birthday?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the composer of film Sruthilayalu?",
            "answer": "K. V. Mahadevan",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "When is K. V. Mahadevan's birthday?",
            "answer": "14 March 1918",
            "dependency": ["1"],
            "document": "2"
        }}
    }}
}}


The origin multi-hop questions is: Where was the director of film The Fascist born?
<document>
DocId: #1
Title: Luciano Salce
Document: Luciano Salce (25 September 1922, in Rome \u2013 17 December 1989, in Rome) was an Italian film director, actor and lyricist. His 1962 film \"Le pillole di Ercole\" was shown as part of a retrospective on Italian comedy at the 67th Venice International Film Festival. As a writer of pop music, he used the pseudonym Pilantra. During World War II, he was a prisoner in Germany. He later worked for several years in Brazil.
Evidence: Luciano Salce - place of birth - Rome

DocId: #2
Title: The Fascist
Document: The Fascist  is a 1961 Italian film directed by Luciano Salce. It was coproduced with France. It was also the first feature film scored by Ennio Morricone.
Evidence: The Fascist - director - Luciano Salce
</document>
Your response:
{{
    "question": "Where was the director of film The Fascist born?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the director of film The Fascist?",
            "answer": "Luciano Salce",
            "dependency": [],
            "document": "2"
        }},
        "2": {{
            "sub_question": "Where was Luciano Salce born?",
            "answer": "Rome",
            "dependency": ["1"],
            "document": "1"
        }}
    }}
}}

The origin multi-hop questions is: Which country the performer of song Soldier (Neil Young Song) is from?
<document>
DocId: #1
Title: Soldier (Neil Young song)
Document: \"Soldier\" is a song by Neil Young from the 1972 soundtrack album, \"Journey Through the Past\". It was the only new track included on the album, and was later released on the 1977 compilation \"Decade\", although it was slightly edited. The song observes how a soldier's eyes \"shine like the sun. \" In the second verse, Young sings he does not believe Jesus because he \"can't deliver right away\".
Evidence: Soldier - preformer - Neil Young

DocId: #2
Title: Neil Young
Document: Neil Percival Young (born November 12, 1945) is a Canadian singer-songwriter. After embarking on a music career in the 1960s, he moved to Los Angeles, where he formed Buffalo Springfield with Stephen Stills, Richie Furay and others. Young had released two solo albums and three as a member of Buffalo Springfield by the time he joined Crosby, Stills & Nash in 1969. From his early solo albums and those with his backing band Crazy Horse, Young has recorded a steady stream of studio and live albums, sometimes warring with his recording company along the way.
Evidence: Neil Young - country of citizenship - Canadian
</document>
Your response:
{{
    "question": "Which country the performer of song Soldier (Neil Young Song) is from?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the performer of song Soldier (Neil Young Song)?",
            "answer": "Neil Young",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "Which country is Neil Young from?",
            "answer": "Canadian",
            "dependency": ["1"],
            "document": "2"
        }}
    }}
}}

The origin multi-hop questions is: {question}
<document>
{chunks}
</document>
Your response:
""".strip()

WikiMQAPromptBridgeComparison = """
You are assigned a multi-hop question decomposition task.
Your mission is to decompose a multi-hop question into a list of single-hop sub_questions based on supporting document for each sub_question, and such that you (GPT-4) can answer each sub_question independently from each document. Each document infers a sub_question id which starts with `#`. The evidence in the document indicates the relation of two entities, in the form of `entity1 - relation - entity2`.
The JSON output must contain the following keys:
- "question": a string, the original multi-hop question.
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string format), and each value should be a dict containing:
    - "sub_question": a string, the decomposed single-hop sub_question. It MUST NOT contain information more than the original question and its dependencies. NEVER introduce information from documents.
    - "answer": a string, the answer of the sub_question.
    - "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other sub_questions, you should list the sub_question number here.
    - "document": a string, the document id that supports the sub_question.

The origin multi-hop questions is: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?
<document>
DocId: #1
Title: The Falcon (film)
Document: Banovi\u0107 Strahinja( Serbian Cyrillic:\" \u0411\u0430\u043d\u043e\u0432\u0438\u045b \u0421\u0442\u0440\u0430\u0445\u0438\u045a\u0430\", released internationally as The Falcon) is a 1981 Yugoslavian- German adventure film written and directed by Vatroslav Mimica based on Strahinja Banovi\u0107, a hero of Serbian epic poetry. It entered the section\" Officina Veneziana\" at the 38th Venice International Film Festival.
Evidence: The Falcon (film) - director - Vatroslav Mimica

DocId: #2
Title: Valentin the Good
Document: Valentin the Good is a 1942 Czech comedy film directed by Martin Fri\u010d.
Evidence: Valentin the Good - director - Martin Fri\u010d

DocId: #3
Title: Vatroslav Mimica
Document: Vatroslav Mimica( born 25 June 1923) is a Croatian film director and screenwriter. Born in Omi\u0161, Mimica had enrolled at the University of Zagreb School of Medicine before the outbreak of World War II. In 1942 he joined Young Communist League of Yugoslavia( SKOJ) and in 1943 he went on to join the Yugoslav Partisans, becoming a member of their medical units. After the war Mimica wrote literary and film reviews, and his career in filmmaking began in 1950 when he became the director of the Jadran Film production studio.
Evidence: Vatroslav Mimica - country of citizenship - Croatia, Yugoslavia

DocId: #4
Title: Martin Fri\u010d
Document: Martin Fri\u010d( 29 March 1902 \u2013 26 August 1968) was a Czech film director, screenwriter and actor. He had more than 100 directing credits between 1929 and 1968, including feature films, shorts and documentary films. Throughout his life, Fri\u010d struggled with alcoholism. On the day of the Warsaw Pact invasion of Czechoslovakia in 1968, he attempted suicide, after battling cancer. He died in the hospital five days later.
Evidence: Martin Fri\u010d - country of citizenship - Czech
</document>
Your response:
{{
    "question": "Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the director of The Falcon (Film)?",
            "answer": "Vatroslav Mimica",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "Who is the director of Valentin The Good?",
            "answer": "Martin Fri\u010d",
            "dependency": [],
            "document": "2"
        }},
        "3": {{
            "sub_question": "What is the nationality of Vatroslav Mimica?",
            "answer": "Croatia and Yugoslavia",
            "dependency": ["1"],
            "document": "3"
        }},
        "4": {{
            "sub_question": "Who is the director of Valentin The Good?",
            "answer": "Czech",
            "dependency": ["2"],
            "document": "4"
        }}
    }}
}}

The origin multi-hop questions is: Which film whose director is younger, Charge It To Me or Danger: Diabolik?
<document>
DocId: #1
Title: Roy William Neill
Document: Roy William Neill (4 September 1887 \u2013 14 December 1946) was an Irish-born American film director best known for directing the last eleven of the fourteen Sherlock Holmes films starring Basil Rathbone and Nigel Bruce, made between 1943 and 1946 and released by Universal Studios.
Evidence: Roy William Neill - date of birth - 4 September 1887

DocId: #2
Title: Danger: Diabolik
Document: Danger: Diabolik  is a 1968 action film directed and co-written by Mario Bava, based on the Italian comic series \"Diabolik\" by Angela and Luciana Giussani. The film is about a criminal named Diabolik (John Phillip Law), who plans large-scale heists for his girlfriend Eva Kant (Marisa Mell). Diabolik is pursued by Inspector Ginko (Michel Piccoli), who blackmails the gangster Ralph Valmont (Adolfo Celi) into catching Diabolik for him.
Evidence: Diabolik - director - Mario Bava

DocId: #3
Title: Charge It to Me
Document: Charge It to Me is a 1919 American silent comedy film directed by Roy William Neill and written by L.V. Jefferson. The film stars Margarita Fischer and Emory Johnson. The film was released on May 4, 1919, by Path\u00e9 Exchange.
Evidence: Charge It to Me - director - Roy William Neill

DocId: #4
Title: Mario Bava
Document: Mario Bava (31 July 1914 \u2013 27 April 1980) was an Italian cinematographer, director, special effects artist and screenwriter, frequently referred to as the \"Master of Italian Horror\" and the \"Master of the Macabre\". His low-budget genre films, known for their distinctive visual flair and technical ingenuity, feature recurring themes and imagery concerning the conflict between illusion and reality, and the destructive capacity of human nature. Born to sculptor, cinematographer and special effects pioneer Eugenio Bava, the younger Bava followed his father into the film industry, and eventually earned a reputation as one of Italy's foremost cameramen, lighting and providing the special effects for such films as \"Hercules\" (1958) and its sequel \"Hercules Unchained\" (1959)
Evidence: Mario Bava - date of birth - 31 July 1914
</document>
Your response:
{{
    "question": "Which film whose director is younger, Charge It To Me or Danger: Diabolik?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the director of Charge It To Me",
            "answer": "Roy William Neill",
            "dependency": [],
            "document": "3"
        }},
        "2": {{
            "sub_question": "Who is the director of Danger: Diabolik",
            "answer": "Mario Bava",
            "dependency": [],
            "document": "2"
        }},
        "3": {{
            "sub_question": "When was Roy William Neill born?",
            "answer": "4 September 1887",
            "dependency": ["1"],
            "document": "1"
        }},
        "4": {{
            "sub_question": "When was Mario Bava born?",
            "answer": "31 July 1914",
            "dependency": ["2"],
            "document": "4"
        }}
    }}
}}

The origin multi-hop questions is: Which film has the director who was born earlier, Along Came Jones (Film) or King Of The Hotel?
<document>
DocId: #1
Title: Along Came Jones (film)
Document: Along Came Jones is a 1945 American Western comedy film directed by Stuart Heisler and starring Gary Cooper, Loretta Young, William Demarest, and Dan Duryea. The film was adapted by Nunnally Johnson from the novel\" Useless Cowboy\" by Alan Le May, and directed by Stuart Heisler. It was the only feature film produced by Cooper during his long film career. Much of the film was shot at the Iverson Movie Ranch in Chatsworth, California.
Evidence: Along Came Jones (film) - director - Stuart Heisler

DocId: #2
Title: Stuart Heisler
Document: Stuart Heisler( December 5, 1896 \u2013 August 21, 1979) was an American film and television director. He was a son of Luther Albert Heisler( 1855- 1916), a carpenter, and Frances Baldwin Heisler( 1857- 1935). He worked as a motion picture editor from 1921 to 1936, then worked as film director for the rest of his career. He directed the 1944 propaganda film\" The Negro Soldier\", a documentary- style recruitment piece targeting African- Americans.
Evidence: Stuart Heisler - date of birth - December 5, 1896

DocId: #3
Title: Carmine Gallone
Document: Carmine Gallone( 10 September 1885 \u2013 4 April 1973) was an early acclaimed Italian film director, screenwriter, and film producer. Considered one of Italian cinema's top early directors, he directed over 120 films in his fifty- year career between 1913 and 1963.
Evidence: Carmine Gallone - date of birth - 10 September 1885

DocId: #4
Title: King of the Hotel
Document: King of the Hotel( French: Le roi des palaces) is a 1932 British- French comedy film directed by Carmine Gallone and starring Jules Berry, Betty Stockfeld and Armand Dranem. It was based on a play by Henry Kistemaeckers. The film's sets were designed by the art director Serge Pim\u00e9noff. A separate English- language version\" King of the Ritz\" was also made.
Evidence: King of the Hotel - director - Carmine Gallone
</document>
Your response:
{{
    "question": "Which film has the director who was born earlier, Along Came Jones (Film) or King Of The Hotel?",
    "decomposed_questions": {{
        "1": {{
            "sub_question": "Who is the director of the film Along Came Jones (Film)?",
            "answer": "Stuart Heisler",
            "dependency": [],
            "document": "1"
        }},
        "2": {{
            "sub_question": "Who is the director of the film King Of The Hotel?",
            "answer": "Carmine Gallone",
            "dependency": [],
            "document": "4"
        }},
        "3": {{
            "sub_question": "When was Stuart Heisler born?",
            "answer": "December 5, 1896",
            "dependency": ["1"],
            "document": "2"
        }},
        "4": {{
            "sub_question": "When was Carmine Gallone born?",
            "answer": "10 September 1885",
            "dependency": ["2"],
            "document": "3"
        }}
    }}
}}

The origin multi-hop questions is: {question}
<document>
{chunks}
</document>
Your response:
""".strip()
