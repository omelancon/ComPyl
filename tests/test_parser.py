import dill

from src.Parser.Parser import format_rules
from src.Parser.FiniteAutomaton.FiniteAutomaton import DFA, Token
from src.Parser.Parser import Parser

rules = {
    'prog':  [
        ('stat', lambda x: x),
        ('stat prog', lambda x, y: x)
    ],
    'stat': [
        ('lol', lambda x: x),
        ('if stat end', lambda x, y, w: x),
        ('if stat else stat end', lambda x, y, z, w: x)
    ],
    'if': [
        ('IF exp', lambda x, y: x)
    ],
    'lol': [
        ('prog', lambda x: x)
    ]
}

formatted_rules = format_rules(rules)

print(formatted_rules)

dfa = DFA(rules=formatted_rules, terminal='stat')

with open('test.p', "wb") as file:
    dill.dump(dfa, file)

with open('test.p', "rb") as file:
    dfa = dill.load(file)

dfa.push(Token('if', None))
dfa.push(Token('stat', None))
dfa.push(Token('end', None))

result = dfa.end()


print(dfa)
