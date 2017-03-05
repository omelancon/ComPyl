from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy


def DISPLAY_STATE(t):
    from pprint import pprint
    pprint(t.params)


def COMMENT(t, v):
    t.params["comments"] += 1

rules = [
    (r"for", "FOR"),
    (r"if", "IF"),
    (r"else", "ELSE"),
    (r"print", "PRINT"),
    (r"break", "BREAK"),
    (r"to", "TO"),
    (r"\(", "L_PAR"),
    (r"\)", "R_PAR"),
    (r":", "COLON"),
    (r";", "SEMICOLON"),
    (r"/--_*--/", COMMENT, "non_greedy"),
    (r"\"[a-zA-Z0-9]*\"", "STRING"),
    (r"[a-zA-Z]+", "ID"),
    (r"0|([1-9][0-9]*)", "INT"),
    (r"\053", "PLUS"),
    (r"\x2d", "MINUS"),
    (r"=", "ASSIGN"),
    (r"==", "EQ"),
    (r"[ \t]+", None)
]

terminal_actions = [
    DISPLAY_STATE
]

buffer = """
x = 0

for:
    if x == 10:
        break;
    else:
        /-- We can comment a bit here --/
        print x;
"""

lexer = Lexer(
    rules=rules,
    terminal_actions=terminal_actions,
    params={'comments': 0}
)
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
