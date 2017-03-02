from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

def COMMENT(t, v):
    return "COMMENT", v

def SPACE(t, v):
    print "I have seen a space"

rules = [
    (r"##_*##", COMMENT, "non_greedy"),
    (r"##AAA##yo", "A_TOKEN"),
    (" |\t", SPACE),
    (r"foo", "FOO"),
    ("a*b?!", "HI")
]

buffer = """##ANYTHIN GOES


 HERE##
 \tfoo"""

lexer = Lexer(rules=rules, params={'tokens_seen': 0})
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
    print tk
