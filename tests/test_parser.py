import dill

from src.Parser.Parser import format_rules

rules =  {
    'prog': [
        ('statement', lambda x: x)
    ],
    'statement': [
        ('if? else?', lambda x, y: (x, y)),
        ('for instruction', lambda x, y: (x, y))
    ],
    'list': [
        ("", lambda: 1)
    ]
}

formatted_rules = format_rules(rules)

with open('test.p', "wb") as file:
    dill.dump(formatted_rules, file)

with open('test.p', "rb") as file:
    formatted_rules = dill.load(file)

print(formatted_rules)
print(formatted_rules['statement'][3][1]())