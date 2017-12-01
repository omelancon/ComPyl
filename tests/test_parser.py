import dill

from src.Parser.Parser import format_rules
from src.Parser.FiniteAutomaton.FiniteAutomaton import DFA, Token

rules = {
    'stat': [
        ('if stat end', lambda x, y, w: x),
        ('if stat else stat end', lambda x, y, z, w: x)
    ],
    'if': [
        ('IF exp', lambda x, y: x)
    ]
}

formatted_rules = format_rules(rules)

with open('test.p', "wb") as file:
    dill.dump(formatted_rules, file)

with open('test.p', "rb") as file:
    formatted_rules = dill.load(file)

print(formatted_rules)

dfa = DFA(rules=formatted_rules, terminal='stat')

dfa.push(Token('if', None))
dfa.push(Token('stat', None))
dfa.push(Token('end', None))

dfa.push(Token(None, None))


print(dfa)
