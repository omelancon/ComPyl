from copy import copy
from compyl.Lexer import Lexer
from compyl.Parser.GrammarError import find_conflicts, GrammarError

initial_rule_name = '@Start'


class ParserRulesError(Exception):
    pass


class ParserSyntaxError(Exception):
    pass


class Token:
    """
    An internal token used by the parser to store type and value.
    Contrary to the Lexer.Token, this should never be returned to the user since self.value will contain a custom
    token defined by the user and built by the reducer.
    """
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return "<Parser Token %s>" % self.type

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type

        else:
            raise NotImplemented


class DFA:
    """
    Build the DFA according to the given rules, save its starting node as self.start and initialize its
    current_state to the start
    """

    def __init__(self, rules=None, terminal=None):
        self.start = None
        self.current_state = None
        self.stack = []
        self.done = False
        self.output = None

        if rules:
            self.build(rules, terminal)

    def build(self, rules, terminal_tokens):
        """
        Build the DFA corresponding to the rules
        :param terminal_tokens:
        :param rules: formated rules
        :return:
        """
        if terminal_tokens is None:
            raise ParserRulesError("No terminal token was given")
        elif not isinstance(terminal_tokens, list):
            terminal_tokens = [terminal_tokens]
        self.stack = []
        self.start = self.current_state = build_dfa(rules, terminal_tokens)

    def push(self, token):
        if not self.done:
            try:
                self._push(token)
            except ParserSyntaxError:
                if token and token.type == initial_rule_name:
                    self.done = True
                    self.output = token.value
                else:
                    raise ParserSyntaxError
        elif token is not None:
            raise ParserSyntaxError('Parser was done but was given extra token')

    def end(self):
        self.push(None)
        return self.output

    def _push(self, token):
        lookout = token.type if token else None
        if lookout not in self.current_state.transitions:
            raise ParserSyntaxError

        transition = self.current_state.transitions[lookout]

        if transition['type'] == 'reduce':
            self._reduce(transition['instruction'])
            self.push(token)

        elif transition['type'] == 'shift':
            self.stack.append((self.current_state, token))
            self.current_state = transition['instruction']

    def _reduce(self, reduce_instruction):
        reducer = reduce_instruction['reducer']
        length = reduce_instruction['reduce_len']
        token_type = reduce_instruction['token']

        reducer_args = [token if isinstance(token, Lexer.Token) else token.value for node, token in
                        self.stack[-length:]] if length > 0 else []
        reduced_value = reducer(*reducer_args)

        new_token = Token(token_type, reduced_value)

        if length > 0:
            self.current_state = self.stack[-length][0]
            self.stack = self.stack[:-length]

        self.push(new_token)

    def reset(self):
        self.current_state = self.start
        self.stack = []
        self.done = False
        self.output = None


class NodeFiniteAutomaton:
    def __init__(self, counter=None, transitions=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shift and reduce transition from current state.
        # Keys of the dict are token (string), values are a NodeFiniteAutomaton
        self.transitions = transitions

    def set_transitions(self, transitions):
        self.transitions = transitions


class TmpNodeFiniteAutomaton:
    """
    Temporary object to build the parsing DFA nodes.
    Meant to accept conflicts.
    """

    def __init__(self, closure=tuple(), counter=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shifts from current state.
        # Keys of the dict are token (string), values are a TmpNodeFiniteAutomaton
        self.shifts = {}

        # Dict of reduce from current state.
        # Keys of the dict are token (string), values are a tuple (TmpNodeFiniteAutomaton, function)
        # The values can be list of such before the final state. This is to keep track of reduce/reduce conflicts
        self.reduce = {}

        # Parsing closure representation of the state
        self.closure = tuple(closure)

    def add_shift(self, lookout, target_node):
        self.shifts[lookout] = target_node

    def add_reduce(self, lookouts, reduce_len, reducer, token):
        reduce_element = {
            "reducer": reducer,
            "reduce_len": reduce_len,
            "token": token
        }
        if lookouts is None:
            lookouts = [None]

        for lookout in lookouts:
            if lookout in self.reduce:
                self.reduce[lookout].append(reduce_element)
            else:
                self.reduce[lookout] = [reduce_element]

    def accept(self, dfa_nodes):
        """
        Recursively converts graph of TmpNodeFiniteAutomaton to graph of NodeFiniteAutomaton.
        Must be called at the root of the DFA
        This will fail if the rules have conflicts and raise GrammarError exception
        :return: NodeFiniteAutomaton
        """

        conflicts = find_conflicts(self)

        if conflicts:
            raise GrammarError(conflicts=conflicts)

        node_translation = {node: NodeFiniteAutomaton() for node in dfa_nodes}

        for node in dfa_nodes:
            transitions = {
                lookout:
                    {'type': 'shift', 'instruction': node_translation[target]}
                for lookout, target in node.shifts.items()
                }
            transitions.update(
                {lookout:
                    {'type': 'reduce', 'instruction': reduce_element[0]}
                 for lookout, reduce_element in node.reduce.items()
                 })
            node_translation[node].set_transitions(transitions)

        return node_translation[self]


# ======================================================================================================================
# Build DFA
# ======================================================================================================================


def build_initial_node(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return: initial TmpNodeFiniteAutomaton
    """

    for token in terminal_tokens:
        if token not in rules:
            raise ParserRulesError("Terminal Token is not present in rules")

    initial_lr_items = []

    # Add rule for the initial node
    rules[initial_rule_name] = [([t], lambda x: x) for t in terminal_tokens]

    for rule, reducer in rules[initial_rule_name]:
        initial_lr_items.append(LrItem([], rule, initial_rule_name, None, reducer))

    closure = get_closure(initial_lr_items, rules)

    return TmpNodeFiniteAutomaton(
        closure=closure
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

    nodes_dict_by_closure = {}

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
                next_state_closure = get_closure(shifted_items, rules)

                if next_state_closure in nodes_dict_by_closure:
                    # A state with this closure already exists
                    next_state = nodes_dict_by_closure[next_state_closure]
                else:
                    # Create the new state and keep track of it
                    next_state = TmpNodeFiniteAutomaton(closure=next_state_closure)
                    nodes.append(next_state)
                    pending_nodes.append(next_state)
                    nodes_dict_by_closure[next_state_closure] = next_state

                # Add the transition/shift to the current node
                node.add_shift(lookout, next_state)

    return initial_node, nodes


def add_reduces_to_node(node):
    for lr_item in [item for item in node.closure if item.is_fully_parsed()]:
        reduce_len = lr_item.get_parsed_length()
        reducer = lr_item.reducer

        node.add_reduce(lr_item.lookouts, reduce_len, reducer, lr_item.token)


def add_reduces_to_dfa(shift_only_dfa_nodes):
    for node in shift_only_dfa_nodes:
        add_reduces_to_node(node)


def build_dfa(rules, terminal_tokens):
    """
    :param rules: parsed rules
    :param terminal_tokens: list of terminal tokens (string)
    :return:
    """

    initial_node, dfa = build_dfa_shifts(rules, terminal_tokens)
    add_reduces_to_dfa(dfa)

    initial_node = initial_node.accept(dfa)

    return initial_node


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
        return hash(
            (
                self.parsed,
                self.expected,
                self.token,
                tuple(self.lookouts) if self.lookouts else None,
                self.reducer
            )
        )

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
    :param initial_items: List of LR_Item's
    :param rules: Parsed rules
    :return: Closure as tuple of LR_Item's
    """
    closure = set()
    pending_items = set(initial_items)
    seen_items = {}

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
                    if new_item not in seen_items:
                        seen_items[new_item] = True
                        next_pending_items.add(new_item)

        closure = closure.union(pending_items)
        pending_items = next_pending_items

    return tuple(closure)
