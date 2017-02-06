from src.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse

def B(t, value):
    return "B_token"

rules = [
    ("A|(B{1,3})", "Accept")
]

nfa = FiniteAutomata.LexerNFA()

r = FiniteAutomata.format_regexp(rules[0][0])

print r.print_regexp()

nfa.build(rules)

visual_lexer.plot_nfa(nfa)