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

code = "bc"

lexer = Lexer.Lexer(rules=rules)

rse_regexps = [sre_parse.parse(rule[0]) for rule in rules]

alphabet = FiniteAutomata.get_alphabet(rse_regexps)

print [chr(a) for a in alphabet]