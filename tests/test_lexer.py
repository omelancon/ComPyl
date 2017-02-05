from src.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse

def B(t, value):
    return "B_token"

rules = [
    ("[1-9]+bcd", "Accept"),
    ("[a-z]bc", "Good")
]

tree = FiniteAutomata.format_regexp('[1-9]+bcd|j')

print tree