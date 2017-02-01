from src.Lexer import Lexer
from src.Visual import visual_lexer

def B(t, value):
    return "B_token"

rules = [
    ("a+bcd", "Accept"),
    ("abc", "Good")
]

code = "bc"

lexer = Lexer.Lexer(rules=rules)

lexer.build()

lexer.save("test.p")

loaded_lexer = Lexer.Lexer.load("test.p")

visual_lexer.plot_lexer_automata(loaded_lexer.fsa)

loaded_lexer.read(code)

while True:
    token = loaded_lexer.lex()

    if token:
        print token
    else:
        break