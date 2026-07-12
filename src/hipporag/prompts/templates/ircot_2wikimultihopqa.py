one_shot_ircot_demo_docs = (
    """Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\n"""
    """Wikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners-up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners-up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter-finals.\n\n"""
    """Wikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\n"""
    """Wikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\n"""
    """Wikipedia Title: Telephone numbers in Ascension Island\nCountry Code: +247. International Call Prefix: 00. Ascension Island does not share the same country code (+290) with the rest of St Helena.\n"""
)

one_shot_ircot_demo = (
    f'{one_shot_ircot_demo_docs}'
    '\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?'
    '\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n'
)

ircot_system = (
    'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    '\n\n'
    f'{one_shot_ircot_demo}'
)

prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"}
]
