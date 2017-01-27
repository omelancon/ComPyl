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


class FiniteAutomata():

    # Empty object to differentiate between no terminal (None) and no-instruction terminal (IGNORED)
    IGNORED = object()

    def __init__(self, current_state="", terminal_token=None, max_repeat_handled=100):
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
        Set the terminal token and raise an exception if it already exists
        """
        if not self.terminal_token:
            if terminal_token is None:
                self.terminal_token = self.IGNORED
            else:
                self.terminal_token = terminal_token
        else:
            #raise NodeAlreadyHasTerminalToken
            pass

    def get_terminal_token(self):
        """
        Return the terminal token and raise and exception if the node is not a terminal state
        """
        if self.terminal_token:
            return None if self.terminal_token is self.IGNORED else self.terminal_token
        else:
            raise NodeIsNotTerminalState


class LexerAutomata(FiniteAutomata):
    def add_rule(self, regexp, token):
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
        Add the child corresponding to the 'lookout' if it doesn't exist and return it.
        Otherwise return the already existing child.

        lookout is expected to have the sre format, but can handle char and int
        """

        next_states = []
        current_states = [self]

        # __class__ is used instead of FiniteAutomata for inheritance
        automata_class = self.__class__

        # Case with repetition
        if isinstance(lookout, tuple) and lookout[0] == 'max_repeat':
            min_repeat = lookout[1][0]
            max_repeat = lookout[1][1]
            lookout = lookout[1][2].data

            formated_lookouts = get_formated_lookouts(lookout)

            # Case where we allow infinite repeat (* and +), i.e. form a loop in the automata
            if max_repeat > self.max_repeat_handled:

                nodes_to_loop = []
                for formated_lookout in formated_lookouts:

                    if not self.lookout_exists(formated_lookout):
                        self.next_states[formated_lookout] = automata_class(formated_lookout)

                    nodes_to_loop.append(self.recover_lookout(formated_lookout))

                for node in nodes_to_loop:
                    for target in nodes_to_loop:
                        if not node.lookout_exists(target.current_state):
                            node.next_states[target.current_state] = target

                next_states = nodes_to_loop

            # There is a maximum amount of repetition for the lookout
            else:

                while max_repeat >= min_repeat and max_repeat != 0:

                    next_states = []

                    for state in current_states:

                        for formated_lookout in formated_lookouts:

                            if state.lookout_exists(formated_lookout):
                                next_states.append(state.recover_lookout(formated_lookout))

                            else:
                                new_state = automata_class(formated_lookout)
                                state.next_states[formated_lookout] = new_state
                                next_states.append(new_state)

                        current_state = next_states
                        max_repeat -= 1

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


def format_sre_to_ascii(sre):
    if sre[0] == "literal":
        return sre[1]
    else:
        raise NotImplemented


def parse_regexp(regexp):
    """
    Tokenize the regexp (string) using Python sre_parse module.
    This returns a SubPattern object, we will mostly be intersted in the data field
    of the returned object.
    """
    pattern = sre_parse.parse(regexp, 0)
    return pattern


def get_ascii_list_from_in(rse):
    formated_lookouts = set()
    for asciis in rse[1]:
        if asciis[0] == 'range':
            min = asciis[1][0]
            max = asciis[1][1]
            formated_lookouts |= set(range(min, max + 1))
        if asciis[0] == 'literal':
            formated_lookouts.add(asciis[1])

    return list(formated_lookouts)


def get_formated_lookouts(lookout):
    formated_lookouts = []

    if isinstance(lookout, str) and len(lookout) == 1:
        formated_lookouts = [format_char_to_ascii(lookout)]

    elif isinstance(lookout, int):
        formated_lookouts = [lookout]

    elif isinstance(lookout, tuple) and lookout[0] == 'literal':
        formated_lookouts = [lookout[1]]

    elif isinstance(lookout, tuple) and lookout[0] == 'in':
        formated_lookouts = get_ascii_list_from_in(lookout)

    elif isinstance(lookout, list):
        for element in lookout:
            formated_lookouts += get_formated_lookouts(element)

    return formated_lookouts
