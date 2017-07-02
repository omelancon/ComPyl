from copy import copy


class ParserAutomatonError(Exception):
    pass


class NodeFiniteAutomaton:
    def __init__(self, is_terminal=False, counter=None, transitions=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shift and reduce transition from current state.
        # Keys of the dict are token (string), values are a NodeFiniteAutomaton
        self.transitions = transitions

        # Indicates if the node is terminal, i.e. accepting state
        self.is_terminal = is_terminal


class Conflict:
    def __init__(self, type, node):
        path = []


        if type == "shift/reduce":

        elif type == "reduce/reduce":

        else:
            raise ValueError("Invalid type for Conflict: " + str(type))


class TmpNodeFiniteAutomaton:
    """
    Temporary object to build the parsing DFA nodes.
    Meant to accept conflicts.
    """
    def __init__(self, closure=tuple(), is_terminal=False, counter=None, shift_parent=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shifts from current state.
        # Keys of the dict are token (string), values are a TmpNodeFiniteAutomaton
        self.shifts = {}

        # Dict of reduce from current state.
        # Keys of the dict are token (string), values are a tuple (TmpNodeFiniteAutomaton, function)
        # The values can be list of such before the final state. This is to keep track of reduce/reduce conflicts
        self.reduce = {}

        # Indicates if the node is terminal, i.e. accepting state
        self.is_terminal = is_terminal

        # Parsing closure representation of the state
        self.closure = tuple(closure)

        # Node that shifts to the current one
        self.shift_parent = shift_parent

    def add_shift(self, lookout, target_node):
        self.shifts[lookout] = target_node
        target_node.shift_parent = self

    def get_nth_shift_parent(self, n):
        return self if n < 1 else self.shift_parent.get_nth_shift_parent(n - 1)

    def add_reduce(self, lookouts, target, reducer, reduce_len):
        reduce_element = {
            "target": target,
            "reducer": reducer,
            "reduce_len": reduce_len
        }
        if lookouts is None:
            lookouts = [None]

        for lookout in lookouts:
            if lookout in self.reduce:
                self.reduce[lookout].append(reduce_element)
            else:
                self.reduce[lookout] = [reduce_element]


def build_initial_node(rules, terminal_tokens):
    """
        :param rules: parsed rules
        :param terminal_tokens: list of terminal tokens (string)
        :return: initial TmpNodeFiniteAutomaton
        """
    initial_lr_items = []

    for token in terminal_tokens:
        try:
            token_rules = rules[token]
        except KeyError:
            raise ParserAutomatonError("Terminal Token is not present in rules")

        for rule, reducer in token_rules:
            initial_lr_items.append(LrItem([], rule, token, None, reducer))

    closure = get_closure(initial_lr_items, rules)

    return TmpNodeFiniteAutomaton(
        closure=closure,
        is_terminal=True
    )


def get_items_with_lookout(lookout, lf_items):
    """
    :param lookout: token (string)
    :param lf_items: list of LfItem's
    :return: filtered list of LfItem which accept lookout
    """
    return list(filter(lambda item: item.is_lookout_accepted(lookout), lf_items))


def build_dfa_shifts(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return: initial node and dict of node keyed by closure
    """
    initial_node = build_initial_node(rules, terminal_tokens)

    nodes = [initial_node]

    pending_nodes = [initial_node]

    while pending_nodes:
        node = pending_nodes.pop(0)

        lf_items = copy(node.closure)

        while lf_items:
            item = lf_items[0]

            if item.is_fully_parsed():
                # Case where the LF item cannot shift
                lf_items = lf_items[1:]
            else:
                # Case where LF item can shift

                # Recover the token/lookout with which the item can shift
                lookout = item.get_next_expected_token()
                same_lookout_items = get_items_with_lookout(lookout, lf_items)

                # We will treat the items which accept that lookout, so remove them from pending items
                lf_items = [item for item in lf_items if item not in same_lookout_items]

                # Shift the items and create the closure
                shifted_items = [item.get_shifted_item() for item in same_lookout_items]
                new_state_closure = get_closure(shifted_items, rules)

                # Create the new state and keep track of it
                new_state = TmpNodeFiniteAutomaton(closure=new_state_closure)
                nodes.append(new_state)
                pending_nodes.append(new_state)

                # Add the transition/shift to the current node
                node.add_shift(lookout, new_state)

    return initial_node, nodes


def add_reduces_to_node(node):
    for lr_item in [item for item in node.closure if item.is_fully_parsed()]:
        step_back_for_reduce = lr_item.get_parsed_length()
        reduce_to_node = node.get_nth_shift_parent(step_back_for_reduce)
        reducer = lr_item.reducer
        reduce_len = lr_item.len_token_reduce()

        node.add_reduce(lr_item.lookouts, reduce_to_node, reducer, reduce_len)


def add_reduces_to_dfa(shift_only_dfa_nodes):
    for node in shift_only_dfa_nodes:
        add_reduces_to_node(node)


def scan_conflicts(dfa):



def build_dfa(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return:
    """

    initial_node, dfa = build_dfa_shifts(rules, terminal_tokens)
    add_reduces_to_dfa(dfa)

    return initial_node, dfa


# ======================================================================================================================
# Closure Computing
# ======================================================================================================================


class LrItem:
    def __init__(self, parsed, expected, token, lookouts, reducer):
        self.parsed = tuple(parsed)
        self.expected = tuple(expected)
        self.token = token
        self.lookouts = lookouts
        self.reducer = reducer

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

    def get_next_expected_token(self, pos=0):
        return self.expected[pos]

    def get_next_expected_atomics(self, rules, pos=0):
        atomics = set()
        nullable = False

        # If there is no expected token at position, return empty set
        if len(self.expected) > pos:
            inspected_tokens = []
            tokens_to_inspect = [self.expected[pos]]

            # Breadth-first-search like dig down of the rules to get atomic tokens under a given token
            #
            while tokens_to_inspect:
                token = tokens_to_inspect.pop(0)
                inspected_tokens.append(token)

                if token in rules:
                    rule = rules[token]
                    for sequence, _ in rule:
                        if not sequence:
                            nullable = True
                        elif sequence[0] not in inspected_tokens:
                            tokens_to_inspect.append(sequence[0])
                else:
                    # An atomic token is one that is not in the rules' keys
                    atomics.add(token)

        return atomics, nullable

    def get_shifted_item(self):
        return LrItem(
            self.parsed + tuple([self.expected[0]]),
            self.expected[1:],
            self.token,
            self.lookouts,
            self.reducer
        )

    def get_parsed_length(self):
        return len(self.parsed)

    def len_token_reduce(self):
        return len(self.parsed) + len(self.expected)


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

                for rule, reducer in next_token_rules:
                    pos = 1
                    lookouts = set()
                    atomics, nullable = item.get_next_expected_atomics(rules, pos)

                    lookouts = lookouts.union(atomics)

                    while nullable:
                        pos += 1
                        atomics, nullable = item.get_next_expected_atomics(rules, pos)
                        lookouts = lookouts.union(atomics)

                    if not lookouts:
                        lookouts = item.lookouts

                    new_item = LrItem([], rule, next_token, lookouts, reducer)
                    next_pending_items.add(new_item)

        closure = closure.union(pending_items)
        pending_items = next_pending_items

    return tuple(closure)
