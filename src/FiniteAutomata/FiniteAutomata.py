from collections import OrderedDict
import sre_parse


class NodeIsNotTerminalState(Exception):
    pass


class NodeAlreadyHasTerminalToken(Exception):
    pass


class FiniteAutomata():
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
            self.next_states[lookout] = FiniteAutomata(lookout)

        return self.next_states[lookout]

    def set_terminal_token(self, terminal_token):
        """
        Set the terminal token and raise an exception if it already exists
        """
        if not self.terminal_token:
            self.terminal_token = terminal_token
        else:
            raise NodeAlreadyHasTerminalToken

    def get_terminal_token(self):
        """
        Return the terminal token and raise and exception if the node is not a terminal state
        """
        if self.terminal_token:
            return self.terminal_token
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
        for rule, token in rules.items():
            self.add_rule(rule, token)


# ========================================================
# Tokenize RegExp
# ========================================================


def parse_regexp(regexp):
    """
    Tokenize the regexp (string) using Python sre_parse module.
    This returns a SubPattern object, we will mostly be intersted in the data field
    of the returned object.
    """
    pattern = sre_parse.parse(regexp, 0)
    return pattern
