from src.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse

def B(t, value):
    return "B_token"

rules = [
    ("for", "F"),
    ("while", "W"),
    ('[a-zA-Z]+', "ID")
]

nfa = FiniteAutomata.LexerNFA()

nfa.build(rules)

visual_lexer.plot_nfa(nfa)