import re
import copy
import dill

from ..FiniteAutomaton.FiniteAutomaton import DFA, NodeIsNotTerminalState


class LexerError(Exception):
    pass


class Token:
    """
    Basic token built by the lexer
    """

    def __init__(self, type, value, lineno=None):
        self.type = type
        self.value = value
        self.lineno = lineno

    def __str__(self):
        return "<Token %s line %s>" % (self.type, str(self.lineno))

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type

        else:
            return NotImplemented


class Lexer:
    """
    Tokenize a string given a set of rules by building a Deterministic Finite Automaton.

    Rules must be provided to Lexer.add_rules as a list of tuple (regex, rule) which first element is a regular
    expressions and second element is a string/function/None.
    If a rule is given as string, the Lexer returns a token with the string as token type.
    If a rule is given as function, the Lexer calls the function with a LexerController and matched pattern as arguments
    and takes return value as token type. See LexerController subclass to see what the function will have access to.
    If None is given as rule, the Lexer interprets the regular expression as a pattern to be ignored (by example for
    spaces, comments, etc.)

    As stated above, a function can be passed as rule to increment Lexer.lineno, although this is not the correct way
    to do line incrementation. Lexer has a line_rule attribute which can be set with Lexer.set_line_rule, this will
    automatically increment line number when the pattern is encountered. In particular, this will increment Lexer.lineno
    even if the pattern is encountered inside another pattern (a multi-line comments for say). Lexer.set_line_rule,
    takes a tuple (regex, rule) as argument. Doing so will also pre-compute regexp intersections in later versions to
    make this option even more efficient.

    Lexer.read appends a string to the current buffer

    Lexer.drop_old_buffer drops the part of the buffer read already

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

        def __init__(self, master):
            self.lineno = master.lineno
            self.pos = master.pos
            self.buffer = master.buffer
            self.params = master.params

            def increment_line(increment=1):
                master.lineno += increment
                self.lineno += increment

            def increment_pos(increment=1):
                master.lineno += increment
                self.lineno += increment

            self.increment_line = increment_line
            self.increment_pos = increment_pos

    def __init__(self, buffer=None, rules=None, line_rule=None, params=None):

        # Line number of the pointer
        self.lineno = 1

        # Position of the pointer in the buffer
        self.pos = 0

        # Code/string to be tokenized by the lexer
        self.buffer = ""

        # List of rules
        self.rules = []

        # A special way to implement the line rule
        self.line_rule = None

        # The Non-deterministic Finite Automaton that can later be optionally be saved
        self.nfa = None

        # A dict of params that is accessible to rules as functions
        self.params = params if params else {}

        if buffer:
            self.read(buffer)

        if line_rule:
            self.set_line_rule(line_rule)

        if rules:
            self.add_rules(rules)

    def __copy__(self):
        """
        Copy the lexer, but reuse the same DFA
        """
        dup = Lexer(rules=copy.copy(self.rules),
                    line_rule=copy.copy(self.line_rule),
                    params=copy.copy(self.params)
                    )
        dup.lineno = self.lineno
        dup.pos = self.pos
        dup.buffer = self.buffer

        # The following reuse existing objects, not deep-copying
        dup.params = self.params
        dup.dfa = self.dfa

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the lexer with its rules and DFA
        """
        dup = Lexer(rules=copy.copy(self.rules),
                    line_rule=copy.copy(self.line_rule),
                    params=copy.copy(self.params)
                    )
        dup.lineno = self.lineno
        dup.pos = self.pos
        dup.buffer = self.buffer

        # The following need to copy objects
        dup.params = copy.deepcopy(self.params)
        dup.dfa = copy.deepcopy(self.dfa)

        return dup

    def read(self, buffer):
        if not isinstance(buffer, str):
            raise LexerError("buffer must be a string")

        self.buffer += buffer

    def drop_old_buffer(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0

    def set_line_rule(self, line_rule):
        self.line_rule = line_rule
        self.add_rules([(line_rule, None)])

    def add_rules(self, rules):
        for regex, rule in rules:
            self.rules.append((regex, rule))

    def build(self):
        self.dfa = DFA(self.rules)

    def save(self, filename="lexer.p"):
        file = open(filename, "wb")
        dill.dump(self, file)

    @staticmethod
    def load(path):

        file = open(path, "rb")

        try:
            lexer = dill.load(file)
        finally:
            file.close()

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

            if lookout_state is None:
                try:
                    terminal_token = self.dfa.get_current_state_terminal()
                except NodeIsNotTerminalState:
                    raise LexerError("Syntax error at line %s" % self.lineno)

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
            token = Token(terminal_token, value, lineno=self.lineno)

        else:
            # Case where the terminal token is a function to be called
            # We expect rule to be a function LexerController -> string -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, it is interpreted as an ignored sequence

            controller = self.LexerController(self)

            try:
                token_type = terminal_token(controller, value)
            except TypeError:
                raise LexerError("Lexer rules must be string or function LexerController -> string -> string/None")

            if isinstance(token_type, str):
                pass
            elif token_type is None:
                ignore = True
            else:
                raise LexerError("Lexer rules as functions must return string or None")

            if not ignore:
                token = Token(token_type, value, lineno=self.lineno)

        # Auto-increment the line number by checking if line_rule match in the match
        # TODO: This makes the lexer a 2 pass lexer, we could improve that by precomputing if linerule and current
        # TODO: rule intersect
        if self.line_rule:
            line_rule_match = re.findall(self.line_rule, value)

            self.lineno += len(line_rule_match)

        # Return if a non-ignored pattern was found, else continue lexing until a token is found
        return self.lex() if ignore else token
