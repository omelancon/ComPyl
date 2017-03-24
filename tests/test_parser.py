from src.Parser.Parser import format_rules

rules =  {
    'prog': [
        ('statement', lambda x: x)
    ],
    'statement': [
        ('if else?', lambda x, y: (x, y)),
        ('for instruction', lambda x, y: (x, y))
    ]
}

formatted_rules = format_rules(rules)

print(formatted_rules)
print(formatted_rules['statement'][1][1]('yo'))