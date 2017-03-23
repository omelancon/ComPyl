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
    "prog": ("statement*", lambda x : AST("prog", *x)),
    "statement": ("if", None)
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


# Algorithm:
# Build DFA for given rules (when merging NFA, look for shift/reduce and reduce/reduce conflicts)
# Traverse tokens with DFA
# When an error is seen, put state in a stack and restart DFA from current position, take return value as lookout
# Only raise when an error is seen at starting state of DFA


def format_rules(rules):

    for token, rule in rules.items():
        pass


def parse_rule(rule):
    pass


def spread_arguments_with_none(sorted_none_pos, fn):
    return lambda *args: fn(*insert_element_at_positions(sorted_none_pos, None, args))


def insert_element_at_positions(sorted_pos_list, element, list):
    new_list = []
    next_element_pos = sorted_pos_list.pop()

    index = 0
    while sorted_pos_list:

        while index != next_element_pos:
            new_list.append(list.pop())
            index += 1

        new_list.append(element)
        next_element_pos = sorted_pos_list.pop()

    return new_list + list[index:]
