from copy import copy, deepcopy

from compyl import lexer
from compyl.__parser.grammar_error import find_conflicts
from compyl.__parser.error import GrammarError, ParserBuildError, ParserSyntaxError

initial_rule_name = '@Start'


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
            return self.type == other.type and self.value == other.value

        else:
            return NotImplemented


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

    def __copy__(self):
        """
        Identical as deepcopy as their is no point at returning a shallow copy
        """
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        """
        Return a copy of the DFA with a deepcopy of the graph
        """
        dup = DFA()

        dup.start = deepcopy(self.start)
        dup.current_state = dup.get_dfa_state_by_id(self.current_state.id)
        dup.stack = deepcopy(self.stack)
        dup.done = self.done
        dup.output = self.output

        return dup

    def build(self, rules, terminal_tokens):
        """
        Build the DFA corresponding to the rules
        :param terminal_tokens:
        :param rules: formated rules
        :return:
        """
        if not terminal_tokens:
            raise ParserBuildError("No terminal token was given")
        elif not isinstance(terminal_tokens, (list, tuple)):
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

        reducer_args = [token if isinstance(token, lexer.Token) else token.value for node, token in
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

    def get_dfa_state_by_id(self, id):
        """
        Return the first state found with given 'id' (should be unique), None if no such state exists
        This is used when copying to recover the copied current_state object
        """
        seen_states = set()
        todo_states = [self.start]

        while todo_states:
            state = todo_states.pop()

            if state.id == id:
                return state

            else:
                seen_states.add(state)

            for _, shift_transition in [t for t in state.transitions if t['type'] == 'shift']:
                child_state = shift_transition['instruction']
                if child_state not in seen_states:
                    todo_states.append(child_state)

        return None


class NodeFiniteAutomaton:
    def __init__(self, counter=None, transitions=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # Dict of shift transitions from current state.
        # Keys of the dict are token (string), values are a NodeFiniteAutomaton
        self.transitions = transitions

    def set_transitions(self, transitions):
        self.transitions = transitions

    def __copy__(self):
        """
        Copy the NodeFiniteAutomaton node, linking it to the next states without copying those
        """
        dup = NodeFiniteAutomaton()
        dup.id = self.id
        dup.transitions = copy(self.transitions)

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the NodeFiniteAutomaton node, recursively copying the transitions
        """
        dup = NodeFiniteAutomaton()
        dup.id = self.id
        memo[id(self)] = dup
        dup.transitions = {}
        for lookout, instruction in self.transitions.items():

            if instruction['type'] == 'shift':
                dup.transitions[lookout] = {
                    'type': instruction['type'],
                    'instruction': deepcopy(instruction['instruction'], memo)
                }

            elif instruction['type'] == 'reduce':
                dup.transitions[lookout] = copy(instruction)

        return dup


class TmpNodeFiniteAutomaton:
    """
    Temporary object to build the parsing DFA nodes.
    Meant to accept conflicts.
    """

    def __init__(self, closure=None, counter=None):
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
        self.closure = closure

    def add_shift(self, lookout, target_node):
        self.shifts[lookout] = target_node

    def add_reduce(self, lookout, reduce_len, reducer, token):
        reduce_element = {
            "reducer": reducer,
            "reduce_len": reduce_len,
            "token": token
        }

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
            raise ParserBuildError("Terminal Token is not present in rules")

    initial_lr_items = []

    # Add rule for the initial node
    rules[initial_rule_name] = [([t], lambda x: x) for t in terminal_tokens]

    for rule, reducer in rules[initial_rule_name]:
        initial_lr_items.append(LrItem([], rule, initial_rule_name, None, reducer))

    closure = get_closure(initial_lr_items, rules)

    return TmpNodeFiniteAutomaton(
        closure=closure
    )


def get_items_with_lookout(lookout, lr_items):
    """
    :param lookout: token (string)
    :param lr_items: list of LrItem's
    :return: filtered list of LrItem which accept lookout
    """
    return list(filter(lambda item: item.is_lookout_accepted(lookout), lr_items))


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

        lr_items = copy(node.closure)

        while lr_items:
            item = lr_items[0]

            if item.is_fully_parsed():
                # Case where the LR item cannot shift
                lr_items = lr_items[1:]
            else:
                # Case where LR item can shift

                # Recover the token/lookout with which the item can shift
                lookout = item.get_next_expected_token()
                same_lookout_items = get_items_with_lookout(lookout, lr_items)

                # We will treat the items which accept that lookout, so remove them from pending items
                lr_items = [item for item in lr_items if item not in same_lookout_items]

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

        node.add_reduce(lr_item.lookout, reduce_len, reducer, lr_item.token)


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


class Closure:
    def __init__(self, lr_items):
        self.lr_items = tuple(sorted(lr_items))

    def __eq__(self, other):
        if isinstance(other, Closure):
            return self.lr_items == other.lr_items

        else:
            return False

    def __hash__(self):
        return hash(self.lr_items)

    def __getitem__(self, index):
        lr_items = self.lr_items[index]

        if isinstance(lr_items, tuple):
            # Wrap the slice of the lr_items in a new Closure
            return Closure(lr_items)

        else:
            # If single element, return it
            return lr_items

    def __len__(self):
        return len(self.lr_items)

    def __bool__(self):
        return bool(self.lr_items)


class LrItem:
    def __init__(self, parsed, expected, token, lookout, reducer):
        self.parsed = tuple(parsed)
        self.expected = tuple(expected)
        self.token = token
        self.lookout = lookout
        self.reducer = reducer

    def __hash__(self):
        return hash(
            (
                self.parsed,
                self.expected,
                self.token,
                self.lookout,
                self.reducer
            )
        )

    def __eq__(self, other):
        return self.parsed == other.parsed and\
                self.expected == other.expected and\
                self.token == other.token and\
                self.lookout == other.lookout and\
                self.reducer == other.reducer

    def __lt__(self, other):
        if isinstance(other, LrItem):
            return self._to_tuple() < other._to_tuple()

        else:
            raise NotImplemented

    def __gt__(self, other):
        if isinstance(other, LrItem):
            return self._to_tuple() > other._to_tuple()

        else:
            raise NotImplemented

    def _to_tuple(self):
        """
        Internal helper to inherit tuple ordering
        This assumes following types:
        self.parsed -> string
        self.expected -> string
        self.token -> string
        self.lookout -> string or None
        self.reducer -> function
        """
        # Hack: since lookout can be None, we need the empty tuple hack to make comparison work
        return self.parsed, self.expected, self.token, (self.lookout,) if self.lookout else tuple(), self.reducer

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
                            # The rule is nullable if it has an empty reduction
                            nullable = True
                        elif sequence[0] not in inspected_tokens:
                            tokens_to_inspect.append(sequence[0])
                else:
                    # An atomic token is one that is not in the rules' keys
                    atomics.add(token)

            # If the rule is nullable, we recover the atomics from the next expected token
            if nullable:
                next_atomics = self.get_next_expected_atomics(rules, pos=pos+1)
                atomics |= next_atomics

        # If the position is out of range, return the item lookouts as atomics
        else:
            atomics.add(self.lookout)

        return atomics

    def get_shifted_item(self):
        return LrItem(
            self.parsed + tuple([self.expected[0]]),
            self.expected[1:],
            self.token,
            self.lookout,
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
    closure_set = set()
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
                    atomics = item.get_next_expected_atomics(rules, pos)

                    lookouts = lookouts.union(atomics)

                    for lookout in lookouts:
                        new_item = LrItem([], rule, next_token, lookout, reducer)
                        if new_item not in seen_items:
                            seen_items[new_item] = True
                            next_pending_items.add(new_item)

        closure_set = closure_set.union(pending_items)
        pending_items = next_pending_items

    return Closure(closure_set)
