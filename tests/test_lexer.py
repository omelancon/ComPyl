from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

def COMMENT(t, v):
    return "COMMENT", v

rules = [
    ("##_*##", COMMENT, "non_greedy"),
    (" ", "SPACE"),
    ("a", "A")
]

buffer = """##



##aaaaaa """

lexer = Lexer(rules=rules, params={'tokens_seen': 0})
lexer.set_line_rule("aaa")
lexer.build()

visual_lexer.plot_dfa(lexer.dfa.start)

lexer.read(buffer)

new = copy.deepcopy(lexer)

new.save("test.p")

loaded = Lexer.load("test.p")

tk = True

while tk:
    tk = loaded.lex()
    print tk
