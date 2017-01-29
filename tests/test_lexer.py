from src.Lexer import Lexer
from src.Visual import visual_lexer

def B(t, value):
    return "B_token"

rules = [
    (" ", None),
    ("je", "sujet"),
    ("tu", "sujet"),
    ("suis", "verbe"),
    ("es", "verbe"),
    ("est", "verbe"),
    ("content", "adjectif"),
    ("heureux", "adjectif"),
    (",", "ponctuation")
]

code = """je suis content, es tu content
tu es heureux"""

lexer = Lexer.Lexer(rules=rules, line_rule="\n")

lexer.build()

lexer.read(code)

while True:
    token = lexer.lex()

    if token:
        print token
    else:
        break

visual_lexer.plot_lexer_automata(lexer.fsa)
