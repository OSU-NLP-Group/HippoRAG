DOCS = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is born in Minsk.",
    "Montebello is a part of Rockland County.",
]
QUERIES = ["What is George Rankin's occupation?", "How did Cinderella reach her happy ending?", "What county is Erik Hort's birthplace a part of?"]
ANSWERS = [["Politician"], ["By going to the ball."], ["Rockland County"]]
GOLD_DOCS = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.", "The prince used the lost glass slipper to search the kingdom.", "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.", "Montebello is a part of Rockland County."],
]
EXTRA_DOCS = ["Tom Hort's birthplace is Montebello.", "Sam Hort's birthplace is Montebello.", "Bill Hort's birthplace is Montebello.", "Cam Hort's birthplace is Montebello."]
