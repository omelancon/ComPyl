import dill
from compyl.Parser.FiniteAutomaton import DFA, Token
from compyl.Parser.Parser import format_rules

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


class Declaration:
    def __init__(self, *args):
        self.value = args


formatted_rules = format_rules(rules)

print(formatted_rules)

dfa = DFA(rules=formatted_rules, terminal='prog')

with open('test.p', "wb") as file:
    dill.dump(dfa, file)

with open('test.p', "rb") as file:
    dfa = dill.load(file)

dfa.push(Token('LEFT_BRACKET', 'a'))
dfa.push(Token('element', 'b'))
dfa.push(Token('RIGHT_BRACKET', 'c'))

result = dfa.end()


print(dfa)
