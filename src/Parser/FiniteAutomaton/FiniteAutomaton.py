from copy import copy

class ParserAutomatonError(Exception):
    pass

class NodeFiniteAutomaton:
    def __init__(self, closure=tuple(), is_terminal=False, counter=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shifts from current state.
        # Keys of the dict are token (string), values are a NodeFiniteAutomaton
        self.shifts = {}

        # Dict of shifts from current state.
        # Keys of the dict are token (string), values are a tuple (NodeFiniteAutomaton, function)
        self.reduce = {}

        # Indicates if the node is terminal, i.e. accepting state
        self.is_terminal = is_terminal

        # Parsing closure representation of the state
        self.closure = tuple(closure)

    def add_shift(self, lookout, target_node):
        self.shifts[lookout] = target_node


def build_initial_node(rules, terminal_tokens):
    """
        :param rules: parsed rules
        :param terminal_tokens: list of terminal tokens (string)
        :return: initial NodeFiniteAutomaton
        """
    initial_lr_items = []

    for token in terminal_tokens:
        try:
            token_rules = rules[token]
        except KeyError:
            raise ParserAutomatonError("Terminal Token is not present in rules")

        for rule, reducer in token_rules:
            initial_lr_items.append(LrItem([], rule))

    closure = get_closure(initial_lr_items, rules)

    return NodeFiniteAutomaton(
        closure=closure,
        is_terminal=True
    )

def get_items_with_lookout(lookout, lf_items):
    """
    :param lookout: token (string)
    :param lf_items: list of LfItem's
    :return: filtered list of LfItem which accept lookout
    """
    return filter(lambda item: item.is_lookout_acctepted(lookout), lf_items)


def build_dfa_shifts(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return: initial node and dict of node keyed by closure
    """
    initial_node = build_initial_node(rules, terminal_tokens)

    nodes_dict_by_closure = {initial_node.closure: initial_node}

    pending_nodes = [initial_node]

    while pending_nodes:
        node = pending_nodes.pop(0)

        lf_items = copy(node.closure)

        while lf_items:
            item = lf_items[0]

            if item.is_fully_parsed:
                # Case where the LF item cannot shift
                lf_items = lf_items[1:]
            else:
                # Case where LF item can shift

                # Recover the token/lookout with which the item can shift
                lookout = item.get_next_expected_token()
                same_lookout_items = get_items_with_lookout(lookout, lf_items)

                # We will treat the items which accept that lookout, so remove them from pending items
                lf_items = [item for item in lf_items if item not in same_lookout_items]

                new_state_closure = get_closure(same_lookout_items, rules)

                if new_state_closure in nodes_dict_by_closure:
                    # The state associated to the calculated closure exists, just add it to shifts
                    node.add_shift(lookout, nodes_dict_by_closure[new_state_closure])
                else:
                    # The state associated to the closure does not exist, create it then add it to shifts
                    new_state = NodeFiniteAutomaton(closure=new_state_closure)
                    nodes_dict_by_closure[new_state_closure] = new_state
                    node.add_shift(lookout, new_state)
                    pending_nodes.append(new_state)

    return initial_node, nodes_dict_by_closure

def build_dfa(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return:
    """

    shift_only_dfa = build_dfa_shifts(rules, terminal_tokens)
    








# ======================================================================================================================
# Closure Computing
# ======================================================================================================================


class LrItem:
    def __init__(self, parsed, expected):
        self.parsed = tuple(parsed)
        self.expected = tuple(expected)

    def __hash__(self):
        return hash((self.parsed, self.expected))

    def __eq__(self, other):
        return self.parsed == other.parsed and self.expected == other.expected

    def is_fully_parsed(self):
        return False if self.expected else True

    def is_lookout_accepted(self, lookout):
        if not self.is_fully_parsed():
            return self.get_next_expected_token() == lookout
        else:
            return False

    def get_next_expected_token(self):
        return self.expected[0]

    def move_demarkator(self):
        return LrItem(self.parsed + [self.expected[0]], self.expected[1:])


def get_closure(initial_items, rules):
    """
    :param initial_states: List of LR_Item's
    :param rules: Parsed rules
    :return: Closure as tuple of LR_Item's
    """
    closure = set()
    pending_items = set(initial_items)

    while pending_items:
        next_pending_items = set()

        for item in pending_items:
            if not item.is_fully_parsed():
                next_token = item.get_next_expected_token()

                try:
                    next_token_rules = rules[next_token]
                except KeyError:
                    continue

                for rule, _ in next_token_rules:
                    new_item = LrItem([], rule)
                    next_pending_items.add(new_item)

        closure = closure.union(pending_items)
        pending_items = next_pending_items

    return tuple(closure)





