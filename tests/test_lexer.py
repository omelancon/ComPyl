from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

def COMMENT(t, v):
    print("A one-line comment")

rules = [
    (r"for", "FOR"),
    (r"if", "IF"),
    (r"else", "ELSE"),
    (r"print", "PRINT"),
    (r"to", "TO"),
    (r"\(", "L_PAR"),
    (r"\)", "R_PAR"),
    (r":", "SEMICOLON"),
    (r"/-.*-/", COMMENT, "non_greedy"),
    (r"/--_*--/", None, "non_greedy"),
    (r"\"[a-zA-Z0-9]*\"", "STRING"),
    (r"[a-zA-Z]+", "ID"),
    (r"[1-9][0-9]*", "INT"),
    (r"\053", "PLUS"),
    (r"\x2d", "MINUS"),
    (r"=", "ASSIGN"),
    (r"==", "EQ"),
    (r"[ \t]+", None)
]

buffer = """
for i = 1 to 20:
    x = i
    if x == i:
        /- We add a comment here -/
        print x + 2
    """

lexer = Lexer(rules=rules)
lexer.set_line_rule("\n")
lexer.build()

visual_lexer.plot_dfa(lexer.dfa.start)

lexer.read(buffer)

new = copy.deepcopy(lexer)

new.save("test.p")

loaded = Lexer.load("test.p")

tk = True

while tk:
    tk = loaded.lex()
    print(tk)
