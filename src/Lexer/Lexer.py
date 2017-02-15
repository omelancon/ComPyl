import re
import copy
import dill as pickle

from ..FiniteAutomata.FiniteAutomata import LexerDFA, NodeIsNotTerminalState


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
        return "<Token %s> line %s" % (self.type, str(self.lineno))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other

        elif isinstance(other, Token):
            return self.type == other.type

        else:
            return NotImplemented


class Lexer:
    """
    Tokenize a string given a set of rules by building a FiniteAutomata.

    Rules must be provided to Lexer.add_rules as a list of tuple (regex, rule) which first element is a regular
    expressions and second element is a string/function/None.
    If a rule is given as string, the Lexer returns a token with the string as token type.
    If a rule is given as function, the Lexer calls the function with Lexer and matched pattern as arguments
    and takes return value as token type, this allows mainly to increment lines manually or do change to the Lexer.
    If None is given as rule, the Lexer interprets the regular expression as a pattern to be ignored (in practice this
    is mostly for spaces, comments, etc.)

    As stated above, a function can be passed as rule to increment Lexer.lineno, although this is not the correct way
    to do line incrementation. Lexer has a line_rule attribute which can be set with Lexer.set_line_rule, this will
    automatically increment line number when the pattern is encountered. In particular, this will increment Lexer.lineno
    even if the pattern is encountered inside another pattern (a multi-line comments for say).

    Lexer.read appends a string to the current buffer

    Lexer.drop_old_buffer drops the part of the buffer read already

    Lexer.lex reads the Lexer.buffer and returns a Token or None if it reached the end of the buffer
    """

    class LexerController:
        """
        A class to generate intermediate objects to be passed to tokens as functions, it provides the functions to do
        basic changes to the Lexer without having access to the dfa and rules. It allows to increment lineno and pos
        as well has having access to the buffer.
        """

        def __init__(self, master):
            self.lineno = master.lineno
            self.pos = master.pos
            self.buffer = master.buffer

            def increment_line(increment=1):
                master.lineno += increment
                self.lineno += increment

            def increment_pos(increment=1):
                master.lineno += increment
                self.lineno += increment

            self.increment_line = increment_line
            self.increment_pos = increment_pos

    def __init__(self, buffer=None, rules=None, line_rule=None, params=None):

        self.lineno = 1
        self.pos = 0
        self.buffer = ""
        self.rules = []
        self.line_rule = None
        self.dfa = None
        self.current_state = None
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
        dup.current_state = self.current_state
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

        if self.current_state is None:
            dup.current_state = None
        else:
            dup.current_state = LexerDFA.get_state_by_id(self.current_state.id, dup.dfa)

            if dup.current_state is None:
                raise LexerError("Error when trying to copy the DFA, the current state could not be retrieved.")

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
        self.dfa = LexerDFA.build(self.rules)

    def save(self, filename="lexer.p"):
        file = open(filename, "wb")
        pickle.dump(self, file)

    @staticmethod
    def load(path):
        try:
            file = open(path, "rb")
            lexer = pickle.load(file)
        except pickle.PickleError:
            raise LexerError("The file " + path + " cannot be loaded by pickle")

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
        self.current_state = self.dfa

        init_pos = self.pos
        terminal_token = None

        # Step through the Finite State Automaton
        while True:
            try:
                lookout = self.buffer[self.pos]
                lookout_state = self.current_state.transition(lookout)
            except IndexError:
                # End of buffer
                lookout_state = None

            if lookout_state is None:
                try:
                    terminal_token = self.current_state.get_terminal_token()
                except NodeIsNotTerminalState:
                    raise LexerError("Syntax error at line %s" % self.lineno)

                break

            self.current_state = lookout_state
            self.pos += 1

        # Exited the FSA, a terminal instruction was given

        # value is later used to increment line number if a line_rule was set, also returned in the Token
        value = self.buffer[init_pos:self.pos]
        ignore = False

        # A None terminal token represents an ignored state (usually spaces and linebreaks)
        if terminal_token is None:
            ignore = True

        elif isinstance(terminal_token, str):
            # Case where the terminal token is not a function, but the token name as string
            token = Token(terminal_token, value, lineno=self.lineno)

        else:
            # Case where the terminal token is a function to be called
            # We expect rule to be a function Lexer-Parser -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, it is interpreted as an ignored sequence

            try:
                controller = self.LexerController(self)
                token_type = terminal_token(controller, value)
            except TypeError:
                raise LexerError("Lexer rules must be string or function (Lexer, value (string) as arguments)")

            if isinstance(token_type, str):
                pass
            elif token_type is None:
                ignore = True
            else:
                raise LexerError("Lexer rules functions must return string or None")

            token = Token(token_type, value, lineno=self.lineno)

        # Auto-increment the line number by checking if line_rule match in the match
        # TODO: This makes the lexer a 2 pass lexer, we could improve that by precomputing if linerule and current
        # TODO: rule intersect
        if self.line_rule:
            line_rule_match = re.findall(self.line_rule, value)

            self.lineno += len(line_rule_match)

        # Return if a non-ignored pattern was found, else continue lexing until a token is found
        return self.lex() if ignore else token
