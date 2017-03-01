from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

rules = [
    ("##_*##", "COMMENT", "non_greedy"),
    ("##AAA##yo", "A_TOKEN"),
    (" ", None),
    ("foo", "FOO")
]

buffer = """##A## foo"""

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
    print tk
