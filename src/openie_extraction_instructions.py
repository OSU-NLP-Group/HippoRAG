from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

## General Prompts

one_shot_passage = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_passage_entities = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

one_shot_passage_triples = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

## NER Prompts

ner_instruction = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

ner_input_one_shot = """Paragraph:
```
{}
```
""".format(one_shot_passage)

ner_output_one_shot = one_shot_passage_entities

ner_user_input = "Paragraph:```\n{user_input}\n```"
ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(ner_instruction),
                                                HumanMessage(ner_input_one_shot),
                                                AIMessage(ner_output_one_shot),
                                                HumanMessagePromptTemplate.from_template(ner_user_input)])

## Post NER OpenIE Prompts

openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

openie_post_ner_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

openie_post_ner_input_one_shot = openie_post_ner_frame.replace("{passage}", one_shot_passage).replace("{named_entity_json}", one_shot_passage_entities)

openie_post_ner_output_one_shot = one_shot_passage_triples

openie_post_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(openie_post_ner_instruction),
                                                            HumanMessage(openie_post_ner_input_one_shot),
                                                            AIMessage(openie_post_ner_output_one_shot),
                                                            HumanMessagePromptTemplate.from_template(openie_post_ner_frame)])
