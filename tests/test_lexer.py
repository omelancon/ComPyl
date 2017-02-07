from src.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse


def B(t, value):
    return "B_token"

rules = [
    ("[ac]+", "Accept")
]

dfa = FiniteAutomata.LexerDFA.build(rules)

visual_lexer.plot_dfa(dfa)