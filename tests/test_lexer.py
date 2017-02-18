from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

number = r"[1-9][0-9]*"
decimal = number + r"\.[0-9]+"


def INT(t, v):
    t.params[v] = "I have seen that one"
    return "INT"


rules2 = [
    (number, INT),
    (decimal, "FLOAT"),
    (r"\+", "PLUS"),
    (r"\*", "TIMES"),
    (r"=", "EQUAL"),
    (r" ", None),
    (r"#.*", None)
]

buffer2 = """1 + 3.08 * 5 = 1.54
# We can put some comments here
1+1=2"""

rules = [
    ("x[ab]*", "A"),
    ("[cd](jk)*",  "CD"),
    ("[e-z]!", "EZ"),
    ("(AB*)+|Z+",  "COMPLEX")
]
buffer = "xabaaabAZxbabadcjkjke!j!t!xABBBAABABBB"

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
