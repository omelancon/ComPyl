from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy

rules = [
    ("[a-z]+", "WORD"),
    ("[A-Z][a-z]*", "NAME"),
    ("certificat en intervention psychosociale", "DOMAINE_DE_PAM"),
    (" ", None)
]

buffer = "bonjour Pam comment vas tu et tu fais un certificat en intervention psychosociale"

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
