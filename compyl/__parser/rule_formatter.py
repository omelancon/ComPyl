import copy
import re

from compyl.__parser.grammar_error import ReduceCycle
from compyl.__parser.error import GrammarError


def rules_are_valid(rules):
    for _, rule in rules.items():
        if not isinstance(rule, (list, tuple)):
            return False
        else:
            for el in rule:
                if not isinstance(el, (list, tuple)):
                    return False
    return True


def format_rules(rules):
    """
    :param rules: rules are given in the following format
        rules = {
            'token': [
                ('keyword1 keyword2 keyword3', function),
                ('keyword4 keyword5', other_function)
            ]
            'other_token': [
                ...
            ],
            ...
        }
    :return: The return format is similar but strings of keywords have been replaced by lists of keywords and special
        symbols (such as ?) have been parsed to add corresponding rules
    """
    formatted_rules = {}

    for token, token_rules in rules.items():

        formatted_rules[token] = []
        for rule in token_rules:
            formatted_rules[token].extend(parse_rule(rule))

    reduce_cycles = get_reduce_cycles(formatted_rules)

    if reduce_cycles:
        reduce_cycles = [ReduceCycle(c) for c in reduce_cycles]
        raise GrammarError(reduce_cycles=reduce_cycles)

    else:
        return formatted_rules


def get_reduce_cycles(formatted_rules):
    formatted_rules = copy.deepcopy(formatted_rules)
    one_to_one_chains = []
    cycles = []

    for reduce_token, rules in formatted_rules.items():
        for rule, _ in rules:
            if len(rule) == 1:
                if reduce_token == rule[0]:
                    cycles.append((reduce_token, rule[0]))
                else:
                    one_to_one_chains.append((reduce_token, rule[0]))

    one_to_one_chains = list(set(one_to_one_chains))

    # Sorting makes the following behavior deterministic
    one_to_one_chains.sort()

    while one_to_one_chains:
        chain = one_to_one_chains.pop()
        new_chains = []

        for next_chain in one_to_one_chains:

            if chain[-1] == next_chain[0] and chain[0] == next_chain[-1]:
                cycles.append(chain + next_chain[1:])

            elif chain[0] == next_chain[-1]:
                # Note: It is important that chaining is inspected toward the reduction
                # This way we might miss chains, but only when there are reduce/reduce conflicts
                new_chain = next_chain[:-1] + chain
                new_chains.append(new_chain)

            else:
                new_chains.append(next_chain)

        one_to_one_chains = new_chains

    return cycles


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
            raise ValueError('Parser only accepts token composed of letters, numbers and underscores')

        if token[-1] == "?":
            token = token[:-1]

            split_rule = copy.deepcopy(parsed_rule)
            append_many(split_rule, pos, at_sub_pos=1)
            append_many(parsed_rule, token, at_sub_pos=0)

            parsed_rule += split_rule

        else:
            append_many(parsed_rule, token, at_sub_pos=0)

    return [(pattern_as_list, spread_arguments_with_none(optional_pos, fn)) for pattern_as_list, optional_pos in
            parsed_rule]


def is_token_valid(token): return True if re.compile(r'^\w+\??$').match(token) else False


def spread_arguments_with_none(sorted_none_pos, fn):
    """
    :param sorted_none_pos: A list of position of argument which must be None
    :param fn: Any function
    :return: A new function that takes len(sorted_none_pos) more argument than fn, but which arguments at positions
             in sorted_none_pos are expected to be None
    """
    return (lambda *args: fn(*insert_none_at_positions(sorted_none_pos, args))) if sorted_none_pos else fn


def insert_none_at_positions(sorted_pos_list, list):
    """
    :param sorted_pos_list: The positions where None must be inserted, this list must be sorted
    :param list: The list of length >= max(sorted_pos_list) + len(sorted_pos_list)
    :return: A list but where None has been inserted at the given position provided
    """
    new_list = []
    pop_list_index = new_list_index = old_list_index = 0

    len_sorted_pos_list = len(sorted_pos_list)

    while pop_list_index < len_sorted_pos_list:
        next_element_pos = sorted_pos_list[pop_list_index]
        pop_list_index += 1

        while new_list_index != next_element_pos:
            new_list.append(list[old_list_index])
            old_list_index += 1
            new_list_index += 1

        new_list.append(None)
        new_list_index += 1

    return new_list + [arg for arg in list[old_list_index:]]