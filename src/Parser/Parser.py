import copy
import dill
import re

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

    formatted_rules = {}

    for token, token_rules in rules.items():

        if hasattr(formatted_rules, token):
            raise ParserException("duplicate rule: %s is defined twice" % token)

        else:
            formatted_rules[token] = []
            for rule in token_rules:
                formatted_rules[token].extend(parse_rule(rule))

    return formatted_rules


def append_many(lists, element, at_sub_pos=None):

    for ls in lists:
        if at_sub_pos is not None:
            ls[at_sub_pos].append(element)
        else:
            ls.append(element)


def parse_rule(rule):
    """
    Parse a rule (pattern, function) by turning the pattern into a list of token and adjusts the function when the
    pattern has optional token (? operator)
    :param rule: A rule is a tuple with a pattern as string and a function
    :return:
    """

    pattern, fn = rule
    token_list = pattern.split()

    parsed_rule = [([], [])]

    for pos, token in enumerate(token_list):
        if not is_token_valid(token):
            raise ParserException('Parser only accepts token composed of letters, numbers and underscores')

        if token[-1] == "?":
            token = token[:-1]

            split_rule = copy.deepcopy(parsed_rule)
            append_many(split_rule, pos, at_sub_pos=1)
            append_many(parsed_rule, token, at_sub_pos=0)

            parsed_rule += split_rule

        else:
            append_many(parsed_rule, token, at_sub_pos=0)

    return [(pattern_as_list, spread_arguments_with_none(nones, fn)) for pattern_as_list, nones in parsed_rule]


def is_token_valid(token): return True if re.compile(r'^\w+\??$').match(token) else False


def spread_arguments_with_none(sorted_none_pos, fn):
    """
    :param sorted_none_pos: A list of position of argument which must be None
    :param fn: Any function
    :return: A new function that takes len(sorted_none_pos) more argument than fn, but which arguments at positions
             in sorted_none_pos are expected to be None
    """
    return lambda *args: fn(*insert_none_at_positions(sorted_none_pos, args)) if sorted_none_pos else fn


def insert_none_at_positions(sorted_pos_list, list):
    """
    :param sorted_pos_list: The positions where None must be inserted, this list must be sorted
    :param list: The list of length >= max(sorted_pos_list)
    :return: A list but where None has been inserted at the given position provided
    """
    new_list = []
    new_list_index = old_list_index = 0

    while sorted_pos_list:
        next_element_pos = sorted_pos_list.pop(0)

        while new_list_index != next_element_pos:
            new_list.append(list[old_list_index])
            old_list_index += 1
            new_list_index += 1

        new_list.append(None)
        new_list_index += 1

    return new_list + [arg for arg in list[old_list_index:]]
