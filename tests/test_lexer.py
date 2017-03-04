from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

def COMMENT(t, v):
    print("A one-line comment")

rules = [
    ("for", "FOR"),
    ("if", "IF"),
    ("else", "ELSE"),
    ("print", "PRINT"),
    ("to", "TO"),
    ("\(", "L_PAR"),
    ("\)", "R_PAR"),
    (":", "SEMICOLON"),
    ("/-.*-/", COMMENT, "non_greedy"),
    ("/--_*--/", None, "non_greedy"),
    ("\"[a-zA-Z0-9]*\"", "STRING"),
    ("[a-zA-Z]+", "ID"),
    ("[1-9][0-9]*", "INT"),
    ("\+", "PLUS"),
    ("-", "MINUS"),
    ("=", "ASSIGN"),
    ("==", "EQ"),
    ("[ \t]+", None)
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

# visual_lexer.plot_dfa(lexer.dfa.start)

lexer.read(buffer)

new = copy.deepcopy(lexer)

new.save("test.p")

loaded = Lexer.load("test.p")

tk = True

while tk:
    tk = loaded.lex()
    print(tk)
