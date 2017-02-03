import sre_parse
from itertools import count


class NodeIsNotTerminalState(Exception):
    pass


class NodeAlreadyHasTerminalToken(Exception):
    pass


class LookoutMustBeSingleCharacter(Exception):
    pass


class LookoutTupleMissFormated(Exception):
    pass


class UnrecognizedLookoutFormat(Exception):
    pass


class FiniteAutomata():
    """
    Basic skeleton for Deterministic and Non-deterministic Finite Automata.
    A FiniteAutomata object is a node of the graph representation of the automata.


    """
    # Counter for state id
    _ids = count(0)

    # Special lookout values
    EMPTY = (-1, -1)

    # Special terminal values
    IGNORED = object()

    # Maximum number of repetition of a same lookout handled, anything above is considered as infinite repetition
    max_repeat_handled = 100

    def __init__(self, terminal_token=None, max_repeat_handled=None):
        self.id = self._ids.next()

        # List of next states from current state.
        # Elements of the list are tuples (lookout, next state), lookout takes the range format (min_ascii, max_ascii)
        self.next_states = []

        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token

        # Allows to by pass the default max_repeat_handled of the class
        if max_repeat_handled:
            self.max_repeat_handled = max_repeat_handled

    def __str__(self):
        return "<State '%d'>" % (self.id)

    def add_transition_range(self, min_ascii, max_ascii, *args, **kwargs):
        """
        Add the edge corresponding to the transition when a character from min_ascii to max_ascii is seen.
        For a single ascii transition, let min_ascii = max_ascii, ex: 'a' would be (97, 97)
        Extra arguments and keyword arguments are passed to the object generator.

        REMARK: in the case of a DFA, be aware that add_transition does not check for already existing transitions.
        """
        lookout = (min_ascii, max_ascii)
        new_state = self.__class__(*args, **kwargs)
        self.next_states.append((lookout, new_state))

        return new_state

    def add_empty_transition(self, *args, **kwargs):
        """
        Same ad add_transition, but for an empty string match
        """
        new_state = self.__class__(*args, **kwargs)
        self.next_states.append((self.EMPTY, new_state))

        return new_state

    def get_transition_states_for_lookout(self, lookout):
        """
        Given a lookout (ascii value as int), return all the transition states attained by this lookout
        """

        states = []

        for transition in self.next_states:
            if value_is_in_range(lookout, transition[0]):
                states.append(transition[1])

        return states

    def get_transition_for_empty_string(self):
        """
        Return transition states corresponding to the empty string
        """

        states = []

        for transition in self.next_states:
            if transition[0] == self.EMPTY:
                states.append(transition[1])

        return states

    def add_transition_to_state(self, min_ascii, max_ascii, state):
        """
        Form an edge from the current state to a pre-existing state
        :param max_ascii: int
        :param min_ascii: int
        :param state: a pre-existing node
        :return: None
        """

        self.next_states.append(((min_ascii, max_ascii), state))

    def add_empty_transition_to_state(self, state):
        """
        Same as add_transition_to_state but with empty string
        :param state:
        :return:
        """
        self.next_states.append((self.EMPTY, state))

    def set_terminal_token(self, terminal_token):
        """
        Set the terminal token if it is not already set to another value
        """
        if not self.terminal_exists():
            if terminal_token is None:
                self.set_terminal_to_ignored()
            else:
                self.terminal_token = terminal_token

    def set_terminal_to_ignored(self):
        """
        Set the terminal token value to ignored
        """
        if not self.terminal_token:
            self.terminal_token = self.IGNORED

    def terminal_is_ignored(self):
        return self.terminal_token is self.IGNORED

    def terminal_exists(self):
        if self.terminal_token:
            return True
        else:
            return False

    def delete_terminal_token(self):
        """
        Delete the terminal token
        """
        self.terminal_token = None

    def get_terminal_token(self):
        """
        Return the terminal token and raise and exception if the node is not a terminal state
        """
        if self.terminal_exists():
            return None if self.terminal_is_ignored() else self.terminal_token
        else:
            raise NodeIsNotTerminalState


class LexerNFA(FiniteAutomata):
    def add_rule(self, regexp, token):
        """
        Add the given rule to the NFA.
        See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html
        :param regexp: A parsed regexp formated as a RegexpTree object
        :param token: the token returned by the rule (a string or a function FiniteAutomata -> string -> string)
        :return: a tuple (first, last) where first and last are respectively the first and last nodes of the rule
        """

        if regexp is not None:

            first = self.add_empty_transition()

            if regexp.type == 'single':
                min_ascii = regexp.min_ascii
                max_ascii = regexp.max_ascii

                next = first.add_transition_range(min_ascii, max_ascii)
                last = next.add_rule(regexp.next, token)

            elif regexp.type == 'union':
                fst_branch = first.add_rule(regexp.fst, token)
                snd_branch = first.add_rule(regexp.snd, token)

                last = fst_branch[1].add_empty_transition()
                snd_branch[1].add_empty_transition_to_state(last)

            elif regexp.type == 'kleene':
                # The regexp A* leads to the following NFA
                #
                # s0 ---> s1 -A-> s2 ---> s3 (ACCEPT)
                # ^        ^------|       |
                # |-----------------------|
                #
                # See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html

                s0, s2 = first.add_rule(regexp.pattern, token)
                s3 = s2.add_empty_transition()
                # There should be a unique empty transition at this point
                s1 = s0.get_transition_for_empty_string()[0]

                s2.add_empty_transition_to_state(s1)
                s3.add_empty_transition_to_state(s0)

                last = s3

        else:
            self.set_terminal_token(token)
            first = last = self

        return first, last

    def build(self, rules):
        for rule, token in rules:
            formated_rule = None
            self.add_rule(rule, token)


class LexerDFA(FiniteAutomata):
    def add_rule(self, regexp, token):
        """
        Add a rule to the FSA.
        First tokenize the regexp using rse_parse module
        Then follow and extend the graph of the FSA with tokenized regexp
            add_or_recover_lookout handles the translation of rse tokens
        Finally add the token as terminal instruction of the terminal states
        """
        tokenized_regexp = parse_regexp(regexp)
        current_states = [self]

        for lookout in tokenized_regexp.data:

            next_states = []

            for current_state in current_states:
                next_states += current_state.add_or_recover_lookout(lookout)

            current_states = next_states

        for current_state in current_states:
            current_state.set_terminal_token(token)

    def build(self, rules):
        for rule, token in rules:
            self.add_rule(rule, token)

    def add_or_recover_lookout(self, lookout):
        """
        Given a lookout, recover the corresponding states of the FSA, creating them if they do not exist.

        Since some lookout will be given as 'max_repeat' rse token, we might traverse/generate chains of states. ex:

             State 97 -> ...
        S -> State 98 -> ... -> State 110
             State 99 -> ... -> State 111

        The function returns a tuple.
        1) The first element of the tuple is a list of tuples (int, node) representing the first layer of nodes from
           the chain preceded by the lookout to attain it. In the above example the corresponding list would be:
           [(97, State 97), (98, State 98), (99, State 99)]
        2) The second element is the list of terminal nodes of the chain. In the above example, it would be:
           [State 110, State 111]

        Note: if the node S loops to itself, it will be contained in the first list.
        Note 2: If the returned chain is of length 1, then the two list will correspond to the same set of states


        The 'lookout' is stored as int (ascii) in the nodes but can be given as string or rse token
        """

        first_states = None
        next_states = []
        current_states = [self]

        # __class__ is used instead of FiniteAutomata for inheritance
        automata_class = self.__class__

        if isinstance(lookout, tuple) and lookout[0] == 'max_repeat':
            # Case with repetition token 'max_repeat'
            # This requires special attention as it creates a chain of states and not simply adds a layer of states.
            # It also leads to the creation of loops in the FSA

            # There are four different cases
            # In all case, the FSA is greedy and tries to recover the longest sequence
            # 1) We allow any number of repetitions, but a minimum is required (ex: 'a+')
            # 2) We allow any number of repetitions, including none (ex: 'a*')
            # 3) We allow up to a finite amount of repetition, but a minimum i required (ex: 'a{2, 6}')
            # 4) We allow up to a finite amount of repetition, including none (ex: 'a{0,6}')
            min_repeat = lookout[1][0]
            max_repeat = lookout[1][1]
            lookout = lookout[1][2].data

            formated_lookouts = get_formated_lookouts(lookout)

            # 1) Case n to inf
            if min_repeat > 0 and max_repeat > self.max_repeat_handled:

                # Generate the nodes chain forcing at least n repetitions
                count = 1
                node_layer = [self]

                while count <= min_repeat:

                    lookout_nodes = {lookout: automata_class(lookout) for lookout in formated_lookouts}

                    for node in node_layer:

                        for formated_lookout in formated_lookouts:

                            if not node.lookout_exists(formated_lookout):
                                node.next_states[formated_lookout] = lookout_nodes[formated_lookout]

                    node_layer = lookout_nodes.values()
                    count += 1

                # Due to infinite repetition, link the final layer of nodes between themselves
                for node in node_layer:
                    for target in node_layer:
                        if not node.lookout_exists(target.current_state):
                            node.next_states[target.current_state] = target

                next_states = node_layer

            # 2) Case 0 to inf
            if min_repeat == 0 and max_repeat > self.max_repeat_handled:

                # Add node corresponding to empty string
                if not self.lookout_exists(-1):
                    empty_state = automata_class(-1)
                    self.next_states[-1] = empty_state
                else:
                    empty_state = self.next_states[-1]

                # Generate the nodes corresponding to the lookouts
                nodes_to_loop = []

                for formated_lookout in formated_lookouts:

                    lookout_nodes = {lookout: automata_class(lookout) for lookout in formated_lookouts}

                    if not self.lookout_exists(formated_lookout):
                        self.next_states[formated_lookout] = lookout_nodes[formated_lookout]

                    nodes_to_loop.append(lookout_nodes[formated_lookout])

                # Due to infinite repetition, link all the created nodes
                for node in nodes_to_loop:
                    for target in nodes_to_loop:
                        if not node.lookout_exists(target.current_state):
                            node.next_states[target.current_state] = target

                next_states = [node for node in nodes_to_loop] + [empty_state]

            # 3) Case n to m
            if min_repeat > 0 and max_repeat <= self.max_repeat_handled:

                # Generate the nodes chain forcing at least n repetitions
                count = 1
                node_layer = [self]

                while count <= min_repeat:

                    lookout_nodes = {lookout : automata_class(lookout) for lookout in formated_lookouts}

                    for node in node_layer:

                        for formated_lookout in formated_lookouts:

                            if not node.lookout_exists(formated_lookout):
                                node.next_states[formated_lookout] = lookout_nodes[formated_lookout]

                    node_layer = lookout_nodes.values()
                    count += 1

                terminal_layers = node_layer

                # Generate the remaining chain (depth n to m) and remember them as they will be returned as terminal
                while count <= max_repeat:

                    lookout_nodes = {lookout : automata_class(lookout) for lookout in formated_lookouts}

                    for node in node_layer:

                        for formated_lookout in formated_lookouts:

                            if not node.lookout_exists(formated_lookout):
                                node.next_states[formated_lookout] = lookout_nodes[formated_lookout]

                    terminal_layers.extend(lookout_nodes.values())
                    node_layer = lookout_nodes.values()
                    count += 1

                next_states = terminal_layers

            # 4) Case 0 to m
            if min_repeat == 0 and max_repeat <= self.max_repeat_handled:

                # Add node corresponding to empty string
                if not self.lookout_exists(-1):
                    empty_state = automata_class(-1)
                    self.next_states[-1] = empty_state
                else:
                    empty_state = self.next_states[-1]

                # Generate the remaining chain (depth m) and remember them as they will be returned as terminal
                count = 1
                node_layer = [self, empty_state]
                terminal_layers = []

                while count <= max_repeat:

                    lookout_nodes = {lookout : automata_class(lookout) for lookout in formated_lookouts}

                    for node in node_layer:

                        for formated_lookout in formated_lookouts:

                            if not node.lookout_exists(formated_lookout):
                                node.next_states[formated_lookout] = lookout_nodes[formated_lookout]

                    terminal_layers.extend(lookout_nodes.values())
                    node_layer = lookout_nodes.values()
                    count += 1

                next_states = terminal_layers

        # Adding a single layer of states, no repetition involved
        else:
            formated_lookouts = get_formated_lookouts(lookout)

            for formated_lookout in formated_lookouts:
                if not self.lookout_exists(formated_lookout):
                    self.next_states[formated_lookout] = automata_class(formated_lookout)

                next_states.append(self.next_states[formated_lookout])

        return next_states

    def recover_lookout(self, lookout):
        """
        Used for the reading stage, recovers the state associated to the lookout.
        The lookout must be in the format used by sre_parse, that is ('literal', [ascii value]),
        but if a single character is given, it is converted to that format for convenience.
        Return None if there is no match.
        """

        if isinstance(lookout, str):

            if len(lookout) == 1:
                lookout = format_char_to_ascii(lookout)

            else:
                raise LookoutMustBeSingleCharacter

        elif isinstance(lookout, tuple):
            if not (lookout[0] == 'literal' and isinstance(lookout[1], int)):
                raise LookoutTupleMissFormated

        elif isinstance(lookout, int):
            pass

        else:
            raise LookoutMustBeSingleCharacter

        next_state = self.next_states.get(lookout)

        # -1 is the empty string
        if next_state is None:
            next_state = self.next_states.get(-1)

        return next_state


# ========================================================
# Set operations
# ========================================================

def value_is_in_range(value, range):
    """
    :param value: an int (x)
    :param range: a tuple of int (min, max)
    :return: True if x from min to max, False otherwise
    """
    return range[0] <= value <= range[1]


def set_to_intervals(ascii_set):
    """
    Given a set of int, return a list of intervals covering those ints exactly
    ex: the set {1,2,3,5,6,9} would be returned as [(1,3), (5,6), (9,9)]
    """

    set_size = len(ascii_set)

    if set_size == 0:
        return []

    else:

        ascii_list = list(ascii_set)
        ascii_list.sort()

        # Hack so the last interval in the list is added
        ascii_list.append(float('inf'))

        interval_list = []

        min = max = ascii_list[0]

        index = 1
        while index < set_size:
            ascii = ascii_list[index]
            if ascii == max + 1:
                max += 1
            else:
                interval_list.append((min, max))
                min = max = ascii

        return interval_list


# ========================================================
# Tokenize RegExp
# ========================================================

class RegexpTree():

    def __init__(self, node_type, *values):

        if node_type in ['single', 'union' 'kleene']:
            self.type = node_type

        else:
            raise ValueError

        if node_type == "single":
            self.min_ascii = values[0]
            self.max_ascii = values[1]
            self.next = values[2] if len(values) > 2 else None

        elif node_type == "union":
            self.fst = values[0]
            self.snd = values [1]
            self.next = values[2]

        elif node_type == 'kleene':
            self.pattern = values[0]
            self.next = values[1]

def format_regexp(regexp):
    """
    Take a regexp as string and return the equivalent RegexpTree.
    Use sre_parse to first tokenize the regexp, then translate it.
    """

    parsed_regexp = sre_parse.parse(regexp)
    return sre_to_regexp_tree(parsed_regexp)

def sre_to_regexp_tree(sre_regexp):
    """
    Take a regexp as sre_parse tokens list and return the equivalent RegexpTree.
    """

    sre_length = len(sre_regexp)

    if sre_length == 0:
        return None

    else:
        current_token = sre_regexp[0]
        token_type = current_token[0]

        if token_type == 'literal':
            pass
        elif token_type == 'in':
            pass
        elif token_type == 'range':
            pass
        elif token_type == 'max_repeat':
            pass
        elif token_type == 'branch':
            pass
        elif token_type == 'at':
            pass





def sre_list_to_interval(regexp_list):
    """
    Given a list of sre 'literal', sre 'in', sre 'range' or int, return a list of corresponding intervals of ascii
    values.
    """

    if not isinstance(regexp_list, list):
        regexp_list = [regexp_list]

    alphabet = set()

    for regexp in regexp_list:
        for token in regexp:
            type = token[0]
            value = token[1]

            if type == "literal":
                alphabet.add(value)

            elif type == "in":
                alphabet |= sre_list_to_interval([value])

            elif type == "range":
                alphabet |= set(range(value[0], value[1] + 1))

    return set_to_intervals(alphabet)


def format_char_to_sre(char):
    return 'literal', ord(char)


def format_char_to_ascii(char):
    return ord(char)


def format_literal_sre_to_ascii(sre):
    if sre[0] == "literal":
        return sre[1]
    else:
        raise UnrecognizedLookoutFormat


def parse_regexp(regexp):
    """
    Tokenize the regexp (string) using Python sre_parse module.
    This returns a SubPattern object, we will mostly be intersted in the data field
    of the returned object.
    """
    pattern = sre_parse.parse(regexp, 0)
    return pattern


def get_ascii_list_from_rse_in(rse):
    """
    Given an 'in' rse token, that is ('in', [list of rse tokens]) return a list of the corresponding asciis
    """
    formated_lookouts = set()
    for asciis in rse[1]:
        if asciis[0] == 'range':
            min = asciis[1][0]
            max = asciis[1][1]
            formated_lookouts |= set(range(min, max + 1))
        if asciis[0] == 'literal':
            formated_lookouts.add(asciis[1])

    return list(formated_lookouts)


def get_formated_lookouts(lookouts):
    """
    Format the given lookout(s) from str, int, sre token or list of those, returning a list of int corresponding to
    the ascii number of these lookouts.
    """
    formated_lookouts = []

    if isinstance(lookouts, str) and len(lookouts) == 1:
        formated_lookouts = [format_char_to_ascii(lookouts)]

    elif isinstance(lookouts, int):
        formated_lookouts = [lookouts]

    elif isinstance(lookouts, list):
        for element in lookouts:
            formated_lookouts += get_formated_lookouts(element)

    # sre formats
    elif isinstance(lookouts, tuple):

            if lookouts[0] == 'literal':
                formated_lookouts = [lookouts[1]]

            elif lookouts[0] == 'in':
                formated_lookouts = get_ascii_list_from_rse_in(lookouts)

    else:
        raise UnrecognizedLookoutFormat

    return formated_lookouts
