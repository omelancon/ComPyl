class LexerError(Exception):
    pass


class LexerSyntaxError(LexerError):
    def __init__(self, *args, lineno=None, pos=None):
        self.lineno, self.pos = lineno, pos
        super().__init__(*args)


class LexerBuildError(LexerError):
    pass


class RegexpParsingError(LexerBuildError):
    pass
