import copy
import dill

from compyl.__lexer.finite_automaton import DFA, NodeIsNotTerminalState
from compyl.__lexer.errors import LexerError, LexerSyntaxError, LexerBuildError, RegexpParsingError
from compyl.__lexer.metaclass import MetaLexer


__all__ = ['Token', 'Lexer', 'LexerError', 'LexerSyntaxError', 'LexerBuildError', 'RegexpParsingError']


# ======================================================================================================================
# Lexer main classes
# ======================================================================================================================


class Token:
    """
    Basic token built by the lexer
    """

    def __init__(self, type, value, pos, end_pos, params=None, lineno=None):
        self.type = type
        self.value = value
        self.pos = pos
        self.end_pos = end_pos
        self.lineno = lineno
        self.params = params

    def __str__(self):
        return "<Lexer Token %s line %s>" % (self.type, str(self.lineno))

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type

        else:
            return NotImplemented


class Lexer(metaclass=MetaLexer):
    """
    Tokenize a string given a set of rules by building a Deterministic Finite Automaton.

    Rules must be provided to Lexer.add_rules as a list of tuple (regex, rule) which first element is a regular
    expressions and second element is a string/function/None.
    If a rule is given as string, the Lexer returns a token with the string as token type.
    If a rule is given as function, the Lexer calls the function with a LexerController and matched pattern as arguments
    and takes return value as token type. Optionally, the function can return a second value which is stored in the
    'params' attribute of the Token object that will be created.
    See LexerController subclass to see what the function will have access to.
    If None is given as rule, the Lexer interprets the regular expression as a pattern to be ignored (by example for
    spaces, comments, etc.)

    As stated above, a function can be passed as rule to increment Lexer.lineno, although this is not the correct way
    to do line incrementation. Lexer has a line_rule attribute which can be set with Lexer.set_line_rule, this will
    automatically increment line number when the pattern is encountered. In particular, this will increment Lexer.lineno
    even if the pattern is encountered inside another pattern (a multi-line comments for say). Lexer.set_line_rule,
    takes a tuple (regex, rule) as argument.

    Rules can also be passed with a third parameter, (regexp, rule, action). Action is a string that can take the
    following values:

    'non_greedy': when the 'non_greedy' tag is added to a rule, the lexer will match the regexp non greedily. By example
        given the buffer "\.. a comment ..\ some more code", the rule "\.._*..\" would return a syntax error since the
        _* would consume greedily. Although if the 'non_greedy' tag is added to the rule, then the comment is matched

    'trigger_on_contain': when this tag is added, the second parameter of the rule is expected to be a function.
        Whenever the given regexp is found nested inside another pattern, the function will be called without creating
        a token. This can be used to count certain patterns nested inside others (by example, this is what the function
        'set_line_rule' uses to be able to count linebreaks in, say, multi-line comment).
        CAVEAT: be aware that trigger_on_contain will be triggered even if the contained patterns intersect. By example,
        if the rule 'aba' was hypothetically used for line incrementation, then the lineno would be incremented twice on
        'ababa', provided that it is inside an accepted pattern. Although, the rule cannot be triggered multiple time
        on a same character, thus '\n+' will not trigger five times on '\n\n\n' but only three times.
        Overall, it is simply better to use 'trigger_on_contain' for simple rules, not using *, + or ?, and maybe even
        consisting of a single character.

    Lexer.read appends a string to the current buffer

    Lexer.drop_old_buffer drops the part of the buffer before 'pos'

    Lexer.lex reads the Lexer.buffer and returns a Token or None if it reached the end of the buffer
    """

    class LexerController:
        """
        A class to generate intermediate objects to be passed to tokens as functions, it provides the methods to do
        basic changes to the Lexer without having access to the DFA and rules. It allows to increment lineno with
        increment_line() and pos with increment_pos() as well has having access to the buffer and params dict.

        A word on the fact that params is entirely accessible and actually points to the Lexer: we allow this instead of
        providing functions for reading/updating the params because it doesn't play a role in the inner logic of the
        lexer, thus it is ok to directly mutate it.
        """

        def __init__(self, master, init_lineno, init_pos, forced_pos=None):
            self.lineno = master.lineno

            # On trigger_on_contain actions, sub-patterns trigger actions lazily, thus the end pos is at the last
            # character and not one ahead. We use forced_pos to enforce consistency with the Lexer position usually
            # displayed when a token is returned.
            self.pos = forced_pos if forced_pos else master.pos
            self.buffer = master.buffer
            self.params = master.params
            self.init_lineno = init_lineno
            self.init_pos = init_pos

            def increment_line(*args):
                increment = args[0] if args else 1
                master.lineno += increment
                self.lineno += increment

            def increment_pos(*args):
                increment = args[0] if args else 1
                master.pos += increment
                self.pos += increment

            self.increment_line = increment_line
            self.increment_pos = increment_pos

    def __init__(self, _dfa=None):
        """

        :param _dfa: A dfa can be passed by the __copy__ or __deepcopy__ methods to avoid the costly operation of
        building another DFA.
        """

        # Line number of the pointer
        self.lineno = 1

        # Position of the pointer in the buffer
        self.pos = 0

        # Code/string to be tokenized by the lexer
        self.buffer = ""

        # The following are class attributes generated by the metaclass
        self.rules = copy.deepcopy(self.__rules__)
        self.params = copy.deepcopy(self.__params__)

        self.terminal_actions = []
        self.parse_terminal_actions(self.__terminal_actions__)

        # Build the dfa
        if _dfa is not None:
            self.dfa = _dfa
        else:
            self._build()

    def __copy__(self):
        """
        Copy the lexer, but reuse the same DFA
        """

        dup = type(self)(_dfa=self.dfa)
        dup.params = self.params

        dup.lineno = self.lineno
        dup.pos = self.pos
        dup.buffer = self.buffer

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the lexer with its rules and DFA
        """
        dup = type(self)(_dfa=copy.deepcopy(self.dfa))
        dup.params = copy.deepcopy(self.params)

        dup.lineno = self.lineno
        dup.pos = self.pos
        dup.buffer = self.buffer


        return dup

    def read(self, buffer):
        if not isinstance(buffer, str):
            raise LexerError("buffer must be a string")

        self.buffer += buffer

    def drop_old_buffer(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0

    def parse_terminal_actions(self, actions):
        for action in actions:
            if isinstance(action, tuple) and len(action) == 2:
                action_fn = action[0]
                action_trigger = action[1]

                if action_trigger == "always":
                    trigger_code = 0

                elif action_trigger == "only_ignored":
                    trigger_code = -1

                elif action_trigger == "only_tokens":
                    trigger_code = 1

                else:
                    raise LexerError(
                        "terminal actions as tuple can only have options 'always', 'only_ignored' or 'only_tokens'"
                    )
            else:
                action_fn = action
                trigger_code = 0

            if callable(action_fn):
                self.terminal_actions.append((action_fn, trigger_code))

            else:
                raise LexerError("""terminal action must be of type (function, string) tuple,
                    the string can take values 'always', 'only_ignored' or 'only_tokens'""")

    def _build(self):
        self.dfa = DFA(rules=self.rules)

    def save(self, filename="lexer.p"):
        with open(filename, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(path):

        with open(path, "rb") as file:
            lexer = dill.load(file)

        if isinstance(lexer, Lexer):
            return lexer
        else:
            raise LexerError("The unpickled object from " + path + " is not a Lexer")

    def lex(self):
        try:
            _ = self.buffer[self.pos]
        except IndexError:
            return None

        # Start at empty state of DFA
        self.dfa.reset_current_state()

        init_lineno = self.lineno
        init_pos = self.pos
        terminal_token = None

        # Step through the Finite State Automaton
        while True:
            end_of_buffer = False
            try:
                lookout = self.buffer[self.pos]
            except IndexError:
                # End of buffer
                end_of_buffer = True

            lookout_state = None if end_of_buffer else self.dfa.push(lookout)

            if lookout_state and lookout_state.has_special_action():
                special_actions = lookout_state.get_special_actions()

                for action_type, action in special_actions:

                    if action_type == DFA.TRIGGER_ON_CONTAIN:
                        controller = self.LexerController(
                            self,
                            init_lineno,
                            init_pos,
                            forced_pos=self.pos + 1
                        )
                        action(controller)

            if lookout_state is None:
                try:
                    terminal_token = self.dfa.get_current_state_terminal()
                except NodeIsNotTerminalState:
                    raise LexerSyntaxError("Syntax error at line %s" % self.lineno)

                break

            self.pos += 1

        # Exited the FSA, a terminal instruction was given

        # value is later used to increment line number if a line_rule was set, also returned in the Token
        value = self.buffer[init_pos:self.pos]
        ignore = False

        # A None terminal token represents an ignored state (usually spaces and linebreaks)
        if terminal_token is None:
            ignore = True

        elif isinstance(terminal_token, str):
            # Case where the terminal token is a string
            token = Token(
                terminal_token,
                value,
                init_pos,
                self.pos,
                lineno=init_lineno)

        else:
            # Case where the terminal token is a function to be called
            # We expect rule to be a function LexerController -> string -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, it is interpreted as an ignored sequence

            controller = self.LexerController(
                self,
                init_lineno,
                init_pos
            )

            try:
                token_return = terminal_token(controller)
            except TypeError:
                raise LexerError("Lexer rules must be string or function LexerController -> (string/None, *)")

            token_type = None
            token_params = None

            if isinstance(token_return, str):
                token_type = token_return
            elif token_return is None:
                ignore = True
            elif isinstance(token_return, tuple) and isinstance(token_return[0], str):
                token_type = token_return[0]
                token_params = token_return[1]
            else:
                raise LexerError(
                    """Lexer rules as functions must return string or None as first return value. An optional second
                    value can be returned to be stored in the token 'params' attribute."""
                )

            if not ignore:
                token = Token(token_type,
                              value,
                              init_pos,
                              self.pos,
                              params=token_params,
                              lineno=init_lineno)

        # Before returning a token, we trigger all terminal actions
        # Recall that the trigger code of the terminal action has the following meaning:
        # -1 -> trigger only on ignored match
        #  1 -> trigger only on match returning token
        #  0 -> always trigger the action the there is a match
        for action, trigger_code in self.terminal_actions:
            if trigger_code == 0:
                controller = self.LexerController(
                    self,
                    init_lineno,
                    init_pos
                )
                action(controller)

            elif trigger_code == -1 and ignore:
                controller = self.LexerController(
                    self,
                    init_lineno,
                    init_pos
                )
                action(controller)

            elif trigger_code == 1 and not ignore:
                controller = self.LexerController(
                    self,
                    init_lineno,
                    init_pos
                )
                action(controller)

        # Return if a non-ignored pattern was found, else continue lexing until a token is found
        if ignore:
            return self.lex()

        else:
            return token
