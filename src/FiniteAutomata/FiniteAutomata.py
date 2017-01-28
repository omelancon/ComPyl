from collections import OrderedDict
import sre_parse


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

    # Empty object to differentiate between no terminal (None) and no-instruction terminal (IGNORED)
    IGNORED = object()

    def __init__(self, current_state=None, terminal_token=None, max_repeat_handled=100):
        self.current_state = current_state
        # OrderedDict allows to keep track of the order in which states are added
        # Giving priority to a rule given first, even though this technically an error by the user
        self.next_states = OrderedDict()
        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token
        # Maximum number of repetition of a same lookout handled, anything above is considered as infinite repetition
        self.max_repeat_handled = max_repeat_handled

    def __str__(self):
        return "<State '%s'>" % (self.current_state)

    def lookout_exists(self, lookout):
        """
        Indicates if the character 'lookout' leads to a legal state
        """
        return lookout in self.next_states

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


class LexerAutomata(FiniteAutomata):
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
        Return the states corresponding to the lookout, creating them if they don't exist

        The 'lookout' is stored as int (ascii) in the nodes but can be given as string or rse token
        """

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
                pass

            # 2) Case 0 to inf
            if min_repeat == 0 and max_repeat > self.max_repeat_handled:
                nodes_to_loop = [self]

                for formated_lookout in formated_lookouts:

                    lookout_is_current_state = (self.current_state == formated_lookout)

                    if not self.lookout_exists(formated_lookout) and not lookout_is_current_state:
                        self.next_states[formated_lookout] = automata_class(formated_lookout)

                    if not lookout_is_current_state:
                        nodes_to_loop.append(self.recover_lookout(formated_lookout))

                for node in nodes_to_loop:
                    for target in nodes_to_loop:
                        if not node.lookout_exists(target.current_state):
                            node.next_states[target.current_state] = target

                next_states = nodes_to_loop

                for state in nodes_to_loop:

                    # -1 is the empty string
                    if state.lookout_exists(-1):
                        next_states.append(state.recover_lookout(-1))

                    else:
                        new_state = automata_class(-1)
                        state.next_states[-1] = new_state
                        next_states.append(new_state)

            # 3) Case n to m
            # TODO: Check
            if min_repeat > 0 and max_repeat <= self.max_repeat_handled:
                current_states = [self]

                while max_repeat >= min_repeat:

                    next_states = []

                    for state in current_states:
                        for formated_lookout in formated_lookouts:

                            if state.lookout_exists(formated_lookout):
                                next_states.append(state.recover_lookout(formated_lookout))

                            else:
                                new_state = automata_class(formated_lookout)
                                state.next_states[formated_lookout] = new_state
                                next_states.append(new_state)

                    current_states = next_states
                    max_repeat -= 1

            # 4) Case 0 to m
            if min_repeat == 0 and max_repeat <= self.max_repeat_handled:
                pass

            # Case where empty string is accepted
            if min_repeat == 0:

                next_states = []

                for state in current_states:

                    # -1 is the empty string
                    if state.lookout_exists(-1):
                        next_states.append(state.recover_lookout(-1))

                    else:
                        new_state = automata_class(-1)
                        state.next_states[-1] = new_state
                        next_states.append(new_state)

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
# Tokenize RegExp
# ========================================================


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
