from src.Lexer import Lexer
from src.Visual import visual_lexer

def B(t):
    return "B_token"

rules = [
    ("[af]*j", "A"),
    ("bca", "B")
]

code = "ajajbcaajbcaaj"

lexer = Lexer.Lexer(rules=rules, line_rule='\n')

lexer.build()

lexer.read(code)

while True:
    token = lexer.lex()

    if token:
        print token
    else:
        break

visual_lexer.plot_lexer_automata(lexer.fsa)
