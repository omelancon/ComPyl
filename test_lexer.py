from src.Lexer import Lexer


def B(t):
    return "B_token"

rules = [
    ("for", "FOR"),
    ("while", "WHILE"),
    (" ", None),
    ("hello", "ID")
]

code = "for while hello for"

lexer = Lexer.Lexer(rules=rules, line_rule='\n')

lexer.build()

lexer.read(code)

while True:
    token = lexer.lex()

    if token:
        print token
    else:
        break
