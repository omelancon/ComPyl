import copy
import dill

from compyl.__parser.finite_automaton import DFA
from compyl.__parser.rule_formatter import rules_are_valid, format_rules
from compyl.__parser.error import ParserError, ParserSyntaxError, ParserBuildError, GrammarError


__all__ = ['Parser', 'ParserError', 'ParserSyntaxError', 'ParserBuildError', 'GrammarError']


# ======================================================================================================================
# Parser decorators
# ======================================================================================================================


def _require_dfa(fn):
    def wrapped_fn(self, *args, **kwargs):
        if not self.dfa:
            raise ParserError
        else:
            return fn(self, *args, **kwargs)

    return wrapped_fn


# ======================================================================================================================
# Parser Main Class
# ======================================================================================================================


class Parser:
    def __init__(self, rules=None, terminal=None):
        self.rules = {}
        self.terminals = []
        self.dfa = None

        if rules:
            self.add_rules(rules)

        if terminal:
            self.set_terminals(terminal)

    def __copy__(self):
        """
        Copy the parser reusing the same DFA
        :return:
        """
        dup = Parser(rules=self.rules, terminal=self.terminals)
        dup.dfa = self.dfa

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the parser and its DFA
        :return:
        """

        dup = Parser(rules=self.rules, terminal=self.terminals)
        dup.dfa = copy.deepcopy(self.dfa)

        return dup

    def set_terminals(self, terminals):
        self.terminals = terminals

    def add_rules(self, rules):
        if rules_are_valid(rules):
            self.rules.update(rules)
        else:
            raise ParserError("Invalid Rule Format")

    def build(self):
        formatted_rules = format_rules(self.rules)
        self.dfa = DFA(formatted_rules, self.terminals)

    def reset(self):
        self.dfa.reset()

    def save(self, filename="lexer.p"):
        with open(filename, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(path):

        file = open(path, "rb")

        try:
            parser = dill.load(file)
        finally:
            file.close()

        if isinstance(parser, Parser):
            return parser
        else:
            raise ParserError("The unpickled object from " + path + " is not a Parser")

    @_require_dfa
    def parse(self, token):
        if token:
            self.dfa.push(token)
            return None
        else:
            return self.end()

    @_require_dfa
    def end(self):
        return self.dfa.end()

