import unittest
import copy

from compyl.Parser.Parser import Parser, ParserException
from compyl.Parser.FiniteAutomaton import Token, ParserSyntaxError, ParserRulesError
from compyl.Parser.GrammarError import GrammarError


# =========================================================
# Helper tokens and functions
# =========================================================


def placeholder_reducer(*args):
    return None


class Start:
    def __init__(self, a, b, end):
        self.a = a
        self.b = b
        self.end = end

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.end == other.end


class End:
    def __init__(self, c, d):
        self.c = c
        self.d = d

    def __eq__(self, other):
        return self.c == other.c and self.d == other.d


def generate_token(type, value='placeholder'):
    return Token(type, value)


def generate_token_stream(*tokens_args):
    tokens = []

    for args in tokens_args:
        if isinstance(args, tuple):
            tokens.append(generate_token(*args))

        else:
            tokens.append(generate_token(args))

    return tokens


# =========================================================
# Tests
# =========================================================


class ParserTestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        rules = {
            'start': [
                ('TOKEN_A TOKEN_B end?', Start),
                ('TOKEN_A TOKEN_C end?', Start),
            ],
            'end': [('TOKEN_C TOKEN_D', End)]
        }

        cls.parser = Parser(rules=rules, terminal='start')
        cls.parser.build()

        cls.parser_copy = copy.deepcopy(cls.parser)

    def tearDown(self):
        self.__class__.parser = copy.deepcopy(self.parser_copy)

    def test_token_stream_one(self):

        token_stream = generate_token_stream(
            ('TOKEN_A', 'a'),
            ('TOKEN_B', 'b'),
            ('TOKEN_C', 'c'),
            ('TOKEN_D', 'd'),
        )

        for tk in token_stream:
            self.parser.parse(tk)

        result = self.parser.end()
        expected_end = End('c', 'd')
        expected = Start('a', 'b', expected_end)

        self.assertEqual(result, expected)

    def test_token_stream_two(self):

        token_stream = generate_token_stream(
            ('TOKEN_A', 'a'),
            ('TOKEN_C', 'b'),
        )

        for tk in token_stream:
            self.parser.parse(tk)

        result = self.parser.end()
        expected = Start('a', 'b', None)

        self.assertEqual(result, expected)

    def test_syntax_error(self):
        token_stream = generate_token_stream(
            'TOKEN_A',
            'TOKEN_B',
            'TOKEN_C',
        )

        for tk in token_stream:
            self.parser.parse(tk)

        tk = generate_token('TOKEN_A')

        self.assertRaises(ParserSyntaxError, self.parser.parse, tk)

    def test_uncomplete_stream(self):

        token_stream = generate_token_stream(
            'TOKEN_A',
            'TOKEN_B',
            'TOKEN_C',
        )

        for tk in token_stream:
            self.parser.parse(tk)

        self.assertRaises(ParserSyntaxError, self.parser.end)


class ParserTestConflicts(unittest.TestCase):

    def test_reduce_reduce_basic(self):

        rules = {
            'start': [('a', placeholder_reducer),
                      ('b', placeholder_reducer)],
            'a': [
                ('A B C', placeholder_reducer)
            ],
            'b': [
                ('A B C', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='start')

        try:
            parser.build()
            self.assertTrue(False, msg='Parser did not detect reduce/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_reduce_reduce())

    def test_reduce_reduce_empty_list(self):

        rules = {
            'list': [
                ('list_of_letters', placeholder_reducer),
                ('list_of_numbers', placeholder_reducer)
            ],
            'list_of_letters': [
                ('', placeholder_reducer),
                ('LETTER list_of_letters', placeholder_reducer)
            ],
            'list_of_numbers': [
                ('', placeholder_reducer),
                ('NUMBER list_of_numbers', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='list')

        try:
            parser.build()
            self.assertTrue(False, msg='Parser did not detect reduce/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_reduce_reduce())

    def test_fixed_empty_list(self):

        rules = {
            'list': [
                ('list_of_letters', placeholder_reducer),
                ('list_of_numbers', placeholder_reducer)
            ],
            'list_of_letters': [
                ('LETTER list_of_letters?', placeholder_reducer)
            ],
            'list_of_numbers': [
                ('NUMBER list_of_numbers?', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='list')

        try:
            parser.build()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_shift_reduce_dangling_else(self):

        rules = {
            'stat': [
                ('IF stat ELSE stat', placeholder_reducer),
                ('IF stat', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='stat')

        try:
            parser.build()
            self.assertTrue(False, msg='Parser did not detect shift/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_shift_reduce())

    def test_fixed_dangling_else(self):

        rules = {
            'stat': [
                ('IF stat ELSE stat SEMICOLON', placeholder_reducer),
                ('IF stat SEMICOLON', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='stat')

        try:
            parser.build()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_shift_reduce_arithmetic(self):

        rules = {
            'exp': [
                ('exp PLUS exp', placeholder_reducer),
                ('exp TIMES exp', placeholder_reducer),
                ('LEFT_PAR exp RIGHT_PAR', placeholder_reducer),
                ('NUMBER', placeholder_reducer),
                ('VARIABLE', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='exp')

        try:
            parser.build()
            self.assertTrue(False, msg='Parser did not detect shift/reduce conflict')

        except GrammarError as e:
            # The length is not deterministic
            self.assertEqual(len(e.conflicts), 8)
            for c in e.conflicts:
                self.assertTrue(c.is_shift_reduce())

    def test_fixed_arithmetic(self):

        rules = {
            'exp': [
                ('factor PLUS factor', placeholder_reducer),
                ('factor', placeholder_reducer)
            ],
            'factor': [
                ('atomic TIMES atomic', placeholder_reducer),
                ('atomic', placeholder_reducer)
            ],
            'atomic': [
                ('NUMBER', placeholder_reducer),
                ('VARIABLE', placeholder_reducer),
                ('LEFT_PAR exp RIGHT_PAR', placeholder_reducer)
            ]
        }

        parser = Parser(rules=rules, terminal='exp')

        try:
            parser.build()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_reduction_cycle(self):

        rules = {
            'A': [('B', placeholder_reducer)],
            'B': [('C', placeholder_reducer)],
            'C': [('A', placeholder_reducer)]
        }

        parser = Parser(rules=rules, terminal='A')

        try:
            parser.build()
            self.assertTrue(False, msg='Parser did not detect reduce cycle')

        except GrammarError as e:
            self.assertEqual(len(e.reduce_cycles), 1)


class ParserTestBuild(unittest.TestCase):

    def test_terminal_token_given(self):
        rules = {
            'A': [('a', placeholder_reducer)],
        }

        parser = Parser(rules=rules)
        self.assertRaises(ParserRulesError, parser.build)

    def test_terminal_in_rules(self):
        rules = {
            'A': [('a', placeholder_reducer)],
        }

        parser = Parser(rules=rules, terminal='B')
        self.assertRaises(ParserRulesError, parser.build)

    def test_rule_structure(self):
        rules = {
            'A': ('a', placeholder_reducer),
        }

        self.assertRaises(ParserException, Parser, rules=rules, terminal='A')

if __name__ == '__main__':
    unittest.main(verbosity=2)




