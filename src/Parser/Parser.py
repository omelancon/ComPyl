import copy
import pickle


# Brainstorm for syntax

# We probably want to keep the same idea of using dicts as in the Lexer. Something like the following EXAMPLE


class Node:
    def __init__(self, *args):
        self.children = args
        self.lineno= args[0].lineno


def A(*args):
    print("Do something here")
    return Node(args[0])


def B(*args):
    return Node(args)


rules = {
    "many_A": {
        ("A", "A", "A"): A,
        ("A", "B", "A"): Node
    },
    "many_B": {
        "B": B
    }
}

parser = Parser(rules=rules)

# I'm still pondering on whether or not we should pass patterns as tuples, it seems like the natural way,
#   since what we want is a sub-stream/array, but hashable, i.e. a n-tuple

# Notice: this time we force passing functions and do not allow strings and implicitly build nodes
#   this choice imitates most other parsers and still allows to pass a class (see the Node) as it can
#   be used to call the __init__ function and actually return a node.
#   Thus the AST elements must be built by the user

# Parser will have an optional start keyword, otherwise


# We then want to generate a PDA, and have a .export() function to store it in a file
# It seems pretty obvious that we want to use an underlying dict to build the PDA

class ParserException(Exception):
    pass


class PDAState:
    def __init__(self, token=None):
        self.token = token
        self.shifts = {}
        self.reduce = None


class Parser:
    def __init__(self, lexer=None, rules=None):
        self.lexer = None
        self.rules = {}

        if rules:
            self.add_rules(rules)

        if lexer:
            self.lexer = copy.copy(lexer)

    def add_rules(self, rules):
        for expression, pattern_rules in rules.items():

            patterns = {}

            for pattern, rule in pattern_rules.items():
                patterns[pattern] = rule

            self.rules[expression] = patterns

    def build(self):
        self.states = {}
        instructions_queue = []

        for expression, pattern_rules in self.rules.items():
            for pattern, rule in pattern_rules.items():

                # Take note of the shift rule
                for i in range(len(pattern)):
                    pass


    def export(self, filename=None):
        pass

# Is it really useful to build and export separately? We are never going to just build and not want to save...
