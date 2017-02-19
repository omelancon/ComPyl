from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

number = r"[1-9][0-9]*"
decimal = number + r"\.[0-9]+"

rules = [
    ("1{3}|(X3{1,2}1)", "THREE_ONE_OR_IS_IT"),
    ("X2{0,4}", "XWITHTWOS"),
    ("X3{1,2}", "XWITHTHREES")
]

buffer = "111XX22X31X33"

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
