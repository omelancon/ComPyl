import copy
import dill

# EX of a rule

class AST(object):
    def __init__(self, *args):
        self.type = args[0]
        self. children = args[1:]

terminals = [
    "prog"
]

rules = {
    "prog": ("state*", lambda x : AST("prog", *x)),
    "state": ("if", None)
}



class ParserException(Exception):
    pass


class Parser:
    def __init__(self, lexer=None, rules=None):
        self.lexer = None
        self.rules = {}

        if rules:
            self.add_rules(rules)

        if lexer:
            self.lexer = copy.copy(lexer)

    def add_rules(self, rules):
        pass

    def build(self):
        pass

    def save(self, filename="parser.p"):
        with open(filename, "wb") as file:
            dill.dump(self, file)

    def parse(self):
        pass
