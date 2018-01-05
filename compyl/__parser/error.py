class ParserError(Exception):
    pass


class ParserBuildError(ParserError):
    pass


class ParserSyntaxError(ParserError):
    pass


class GrammarError(ParserBuildError):
    def __init__(self, conflicts=None, reduce_cycles=None):

        self.conflicts = [] if conflicts is None else conflicts

        self.reduce_cycles = [] if reduce_cycles is None else reduce_cycles

        qty_rr_conflicts = len([c for c in self.conflicts if c.type == "reduce/reduce"])
        qty_sr_conflicts = len(self.conflicts) - qty_rr_conflicts
        qty_reduce_cycles = len(self.reduce_cycles)

        message = 'Grammar errors detected' + \
                  (' | {0} reduce/reduce'.format(str(qty_rr_conflicts)) if qty_rr_conflicts else '') + \
                  (' | {0} shift/reduce'.format(str(qty_sr_conflicts)) if qty_sr_conflicts else '') + \
                  (' | {0} reduce cycle'.format(str(qty_reduce_cycles)) if qty_reduce_cycles else '') + \
                  '\n'
        message += '\n'.join(sorted([c.to_string() for c in self.conflicts + self.reduce_cycles]))

        super(GrammarError, self).__init__(message)
