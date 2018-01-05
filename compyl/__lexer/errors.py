class LexerError(Exception):
    pass


class LexerSyntaxError(LexerError):
    pass


class LexerBuildError(LexerError):
    pass


class RegexpParsingError(LexerBuildError):
    pass
