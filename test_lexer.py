from src.Lexer import Lexer


def B(t):
    return "B_token"


rules = {
    'a+(\n)*': "A",
    'b+': B,
    '#+': None
}

lexer = Lexer.Lexer(rules=rules, line_rule='\n')

code = "aaaa\n\naabbbbb\naabaa##a#abbbbbbbbaa\na#ababababaaaa#ba\nbba\nb#\n"

lexer.read(code)

while True:
    tok = lexer.lex()

    if tok:
        print tok
    else:
        break

print "Done"

print lexer.lex()

print "Reset"

lexer.drop_buffer(drop_lineno=True)
lexer.read(code)
lexer.read("aaaaaabbbb")

while True:
    tok = lexer.lex()

    if tok:
        print tok
    else:
        break

lexer.read("a\na")

tok1 = lexer.lex()
tok2 = lexer.lex()

print tok1 == tok2
