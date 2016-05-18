import re


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
    Dynamic lexer, tokenize a string given a set of rules with the re library.

    Rules must be provided to Lexer.add_rules as a list of tuple (regex, rule) which first element is a regular
    expressions and second element is a string/function/None.
    If a rule is given as string, the Lexer returns a token with the string as type.
    If a rule is given as function, the Lexer calls the function on self and takes the return value as token type, this
    allows mainly to increment lines manually or do change to the Lexer. In particular, rules are dynamic and can be
    added/removed/overwritten during the lexing.
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

    def lex(self):
        if not self.buffer[self.pos:]:
            return None

        best_end = 0
        match = None
        rule = None

        # TODO: There probably is a better way to do this, by building a table
        for (regex, match_rule) in self.rules:
            current_match = re.match(regex, self.buffer[self.pos:])
            if current_match:

                current_end = current_match.end()

                if current_end > best_end:
                    match = current_match
                    rule = match_rule
                    best_end = current_end

        if not match:
            raise LexerError("Syntax error at line %s" % self.lineno)

        value = match.group()
        ignore = False

        if rule is None:
            ignore = True

        elif isinstance(rule, str):
            token = Token(rule, value, lineno=self.lineno)

        else:
            # We expect rule to be a function Lexer-Parser -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, the function.__name__ is taken as Token type

            try:
                token_type = rule(self)
            except TypeError:
                raise LexerError("Lexer rules must be string or function (Lexer-Parser as argument)")

            if isinstance(token_type, str):
                pass
            elif token_type is None:
                ignore = True
            else:
                raise LexerError("Lexer rules functions must return string or None")

            token = Token(token_type, value, lineno=self.lineno)

        # Update the Lexer-Parser
        self.pos += match.end()

        # TODO: change that by pre computing in add_rule phase if line_rule can match current regex
        # TODO: in other words we don't want to do this if regexps have no intersection
        if self.line_rule:
            line_rule_match = re.findall(self.line_rule, value)

            self.lineno += len(line_rule_match)

        # Return if a non-ignored pattern was found, else continue
        return self.lex() if ignore else token
