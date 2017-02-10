from src.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse

rules = [
    ("for", "FOR"),
    ("if", "IF"),
    (" ", None)
]

dfa = FiniteAutomata.LexerDFA.build(rules)

visual_lexer.plot_dfa(dfa)