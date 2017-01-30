from src.Lexer import Lexer
from src.Visual import visual_lexer

def B(t, value):
    return "B_token"

rules = [
    ("[cd]{2,1000}", "Accept"),
]

code = "cda"

lexer = Lexer.Lexer(rules=rules)

lexer.build()

visual_lexer.plot_lexer_automata(lexer.fsa)

lexer.read(code)

while True:
    token = lexer.lex()

    if token:
        print token
    else:
        break
