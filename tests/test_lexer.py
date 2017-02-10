from src.Lexer.Lexer import Lexer
from src.FiniteAutomata import FiniteAutomata
from src.Visual import visual_lexer
import sre_parse

number = "[1-9][0-9]*"
decimal = number + "\.[0-9]+"

def INT(t, v):
    print v
    return "INT"

rules = [
    (number, INT),
    (decimal, "FLOAT"),
    ("\+", "PLUS"),
    ("\*", "TIMES"),
    ("=", "EQUAL"),
    (" ", None),
    ("#.*", None)
]

lexer = Lexer()
lexer.add_rules(rules)
lexer.set_line_rule("\n")
lexer.build()

visual_lexer.plot_dfa(lexer.dfa)

buffer = """1 + 3.08 * 5 = 1.54
# We can put some comments here
1+1=2"""

lexer.read(buffer)

tk = True

while tk:
    tk = lexer.lex()
    print tk