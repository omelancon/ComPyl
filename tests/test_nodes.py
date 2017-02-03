from src.FiniteAutomata import FiniteAutomata
import re
import sre_compile

empty_node = FiniteAutomata.FiniteAutomata()

print empty_node

print empty_node.add_or_recover_lookout("a")

empty_node.set_terminal_token("EMPTY")

sentence = "[a-z]b+g*a\\*"

print sentence

for c in FiniteAutomata.parse_regexp(sentence):
	print c

a = FiniteAutomata.parse_regexp("ab[cd]h*")
a.getwidth()
print a

def B(t):
    return "B_token"

rules = {
    'a+(\n)*': "A",
    'b+': B,
    '#+': None
}

rules = [
    ("for", "FOR"),
    ("while", "WHILE"),
    (" ", None),
    ("[a-z]+", "ID")
]

a = FiniteAutomata.LexerDFA()

a.build(rules)
print a