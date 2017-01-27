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

    def __init__(self, current_state="", terminal_token=None):
        self.current_state = current_state
        # OrderedDict allows to keep track of the order in which states are added
        # Giving priority to a rule given first, even though this technically an error by the user
        self.next_states = OrderedDict()
        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token

    def __str__(self):
        return "<State '%s'>" % (self.current_state)

    def lookout_exists(self, lookout):
        """
        Indicates if the character 'lookout' leads to a legal state
        """
        return lookout in self.next_states

    def add_or_recover_lookout(self, lookout):
        """
        Add the child corresponding to the 'lookout' if it doesn't exist and return it.
        Otherwise return the already existing child.
        """
        if not self.lookout_exists(lookout):
            # __class__ is used instead of FiniteAutomata for inheritance
            automata_class = self.__class__
            self.next_states[lookout] = automata_class(lookout)

        return self.next_states[lookout]

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
            raise NodeAlreadyHasTerminalToken

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
        current_state = self

        for lookout in tokenized_regexp.data:
            current_state = current_state.add_or_recover_lookout(lookout)

        current_state.set_terminal_token(token)

    def build(self, rules):
        for rule, token in rules:
            self.add_rule(rule, token)

    def recover_lookout(self, lookout):
        """
        Used for the reading stage, recovers the state associated to the lookout.
        The lookout must be in the format used by sre_parse, that is ('literal', [ascii value]),
        but if a single character is given, it is converted to that format for convenience.
        Return None if there is no match.
        """

        if isinstance(lookout, str):

            if len(lookout) == 1:
                lookout = format_char_to_sre(lookout)

            else:
                raise LookoutMustBeSingleCharacter

        elif isinstance(lookout, tuple):
            if not (lookout[0] == 'literal' and isinstance(lookout[1], int)):
                raise LookoutTupleMissFormated

        else:
            raise LookoutMustBeSingleCharacter

        next_state = self.next_states.get(lookout)

        return next_state


# ========================================================
# Tokenize RegExp
# ========================================================


def format_char_to_sre(char):
    return 'literal', ord(char)


def parse_regexp(regexp):
    """
    Tokenize the regexp (string) using Python sre_parse module.
    This returns a SubPattern object, we will mostly be intersted in the data field
    of the returned object.
    """
    pattern = sre_parse.parse(regexp, 0)
    return pattern
