from src.FiniteAutomata import NodeFiniteAutomaton
import re
import sre_compile

empty_node = NodeFiniteAutomaton.NodeFiniteAutomaton()

print empty_node

print empty_node.add_or_recover_lookout("a")

empty_node.set_terminal_token("EMPTY")

sentence = "[a-z]b+g*a\\*"

print sentence

for c in NodeFiniteAutomaton.parse_regexp(sentence):
	print c

a = NodeFiniteAutomaton.parse_regexp("ab[cd]h*")
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

a = NodeFiniteAutomaton.NodeDFA()

a.build(rules)
print a