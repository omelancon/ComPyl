import Lexer

def B():
    return

rules = {
    'a+(\n)*': "A",
    'b+': "B"
}

lexer = Lexer.Lexer(rules=rules, line_rule='\n')

code = "aaaa\n\naabbbbbaabaaaabbbbbbbbaa\naababababaaaaba\nbba\n"

lexer.read(code)

while True:
    tok = lexer.lex()

    if tok:
        print tok.type + " at line " + str(tok.lineno) + " with match " + tok.value.strip()
    else:
        break
