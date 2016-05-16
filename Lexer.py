import re


class LexerError(Exception):
    pass


class Token:
    def __init__(self, type, value, lineno=None):
        self.type = type
        self.value = value
        self.lineno = lineno


class Lexer:
    def __init__(self, buffer=None, rules=None, line_rule=None):

        self.lineno = 1
        self.pos = 0
        self.buffer = ""
        self.rules = {}
        self.line_rule = None

        if buffer:
            self.read(buffer)

        if line_rule:
            self.add_line_rule(line_rule)

        if rules:
            self.add_rules(rules)

    def read(self, buffer, drop_buffer=False):
        if not isinstance(buffer, str):
            raise ValueError("buffer must be a string")

        if drop_buffer:
            self.buffer = buffer
            self.pos = 0
        else:
            self.buffer += buffer

    def add_line_rule(self, line_rule):
        try:
            self.line_rule = re.compile(line_rule)
        except re.error:
            raise ValueError("line_rule must be a valid regex")

    def add_rules(self, rules):

        # Use items() for Python3 compatibility
        for regex, rule in rules.items():
            self.rules[regex] = rule

    def lex(self):
        if not self.buffer[self.pos:]:
            return None

        for regex, rule in self.rules.items():
            match = re.match(regex, self.buffer[self.pos:])
            if match:
                break
        else:
            raise LexerError("Syntax error at line %s" % self.lineno)

        value = match.group()

        if isinstance(rule, str):
            token = Token(rule, value, lineno=self.lineno)

        else:
            # We expect rule to be a function Lexer -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, the function.__name__ is taken as Token type
            value = match.group()

            try:
                token_type = rule(self)
            except TypeError:
                raise LexerError("Lexer rules must be string or function")

            if isinstance(token_type, str):
                pass
            elif token_type is None:
                token_type = rule.__name__
            else:
                raise LexerError("Lexer rules functions must return string or None")

            token = Token(token_type, value, self.lineno)

        # Update the Lexer
        self.pos += match.end()

        # TODO: change that by pre computing in add_rule phase if line_rule can match current regex
        if self.line_rule:
            line_rule_match = re.findall(self.line_rule, value)

            self.lineno += len(line_rule_match)

        return token
