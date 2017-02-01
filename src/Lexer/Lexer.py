import re
import pickle

from ..FiniteAutomata.FiniteAutomata import LexerAutomata, NodeIsNotTerminalState


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
    def __init__(self, buffer=None, rules=None, line_rule=None):

        self.lineno = 1
        self.pos = 0
        self.buffer = ""
        self.rules = []
        self.line_rule = None
        self.fsa = LexerAutomata()
        self.current_state = None

        if buffer:
            self.read(buffer)

        if line_rule:
            self.set_line_rule(line_rule)

        if rules:
            self.add_rules(rules)

    def __copy__(self):
        """
        Copy the lexer with its rules
        """
        return Lexer(rules=self.rules, line_rule=self.line_rule)

    def __deepcopy__(self, memo):
        """
        Copy the lexer with its rules and current state
        """
        # TODO: Copy the FSA
        dup = Lexer(buffer=self.buffer,
                    rules=self.rules,
                    line_rule=self.line_rule)
        dup.lineno = self.lineno
        dup.pos = self.pos

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
        self.fsa.build(self.rules)

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

        # Start at empty state of FSA
        self.current_state = self.fsa

        init_pos = self.pos
        terminal_token = None

        # Step through the Finite State Automaton
        while True:
            try:
                lookout = self.buffer[self.pos]
                lookout_state = self.current_state.recover_lookout(lookout)
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

        # A None terminal token represents an ignored state
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
                token_type = terminal_token(self, value)
            except TypeError:
                raise LexerError("Lexer rules must be string or function (Lexer-Parser, value (string) as arguments)")

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
