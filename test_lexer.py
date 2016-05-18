from src.Lexer import Lexer


def B(t):
    return "B_token"


rules = {
    'a+(\n)*': "A",
    'b+': B,
    '#+': None
}

rules = [
    ("for", "FOR"),
    ("while", "WHILE"),
    (" ", None),
    ("[a-z]+", "ID")
]

lexer = Lexer.Lexer(rules=rules, line_rule='\n')

code = "aaaa\n\naabbbbb\naabaa##a#abbbbbbbbaa\na#ababababaaaa#ba\nbba\nb#\n"

code = "for while forwhile"

lexer.read(code)

while True:
    tok = lexer.lex()

    if tok:
        print tok
    else:
        break

tok1 = lexer.lex()
tok2 = lexer.lex()

print tok1 == tok2
