import re


class LexerError(Exception):
    pass


class Token:
    def __init__(self, type, value, lineno=None):
        self.type = type
        self.value = value
        self.lineno = lineno

    def __str__(self):
        return "<Token %s> line %s" % (self.type, str(self.lineno))

    def __eq__(self, other):
        return self.type == other.type


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
            self.set_line_rule(line_rule)

        if rules:
            self.add_rules(rules)

    def read(self, buffer):
        if not isinstance(buffer, str):
            raise LexerError("buffer must be a string")

        self.buffer += buffer

    def drop_buffer(self, drop_lineno=False):

        if drop_lineno:
            self.lineno = 1

        self.pos = 0
        self.buffer = ""

    def set_line_rule(self, line_rule):
        self.line_rule = line_rule
        self.add_rules({line_rule: None})

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
        ignore = False

        if rule is None:
            ignore = True

        elif isinstance(rule, str):
            token = Token(rule, value, lineno=self.lineno)

        else:
            # We expect rule to be a function Lexer -> string/None
            # if a string is returned, it is taken as the Token type
            # if None is returned, the function.__name__ is taken as Token type

            try:
                token_type = rule(self)
            except TypeError:
                raise LexerError("Lexer rules must be string or function (Lexer as argument)")

            if isinstance(token_type, str):
                pass
            elif token_type is None:
                ignore = True
            else:
                raise LexerError("Lexer rules functions must return string or None")

            token = Token(token_type, value, self.lineno)

        # Update the Lexer
        self.pos += match.end()

        # TODO: change that by pre computing in add_rule phase if line_rule can match current regex
        # TODO: in other words we don't want to do this if regexs have no intersection
        if self.line_rule:
            line_rule_match = re.findall(self.line_rule, value)

            self.lineno += len(line_rule_match)

        # Return if a non-ignored pattern was found, else continue
        return self.lex() if ignore else token
