import unittest

import copy
from compyl.__parser.finite_automaton import Token
from compyl import ParserBuildError, ParserSyntaxError, GrammarError
from compyl import Parser as P

# =========================================================
# Helper tokens and functions
# =========================================================


def placeholder_reducer(*args):
    return None


def get_string_of_instructions_reducer(reduce_marker):
    """
    Return a reducer to make actions taken by the parser readable
    Returned reducer returns a string of seen tokens and reduction markers provided separated by white spaces
    """
    return lambda *args: ' '.join([a.value if hasattr(a, 'value') else str(a) for a in args] + [reduce_marker])


class ComparableNode:
    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

class LooseComparisonToken(Token):
    """Mimicks the Parser.Token class to only check type and value attributes
    This allows comparison of mocked Parser.Token and Lexer.Token only by their attribute and not types"""
    def __eq__(self, other):
        return self.type == other.type and self.value == other.value


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

        class Parser(P, terminal='start'):
            start =\
                ('TOKEN_A TOKEN_B end?', Start),\
                ('TOKEN_A TOKEN_C end?', Start)
            end = ('TOKEN_C TOKEN_D', End)

        cls.parser = Parser()

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

    def test_incomplete_stream(self):

        token_stream = generate_token_stream(
            'TOKEN_A',
            'TOKEN_B',
            'TOKEN_C',
        )

        for tk in token_stream:
            self.parser.parse(tk)

        self.assertRaises(ParserSyntaxError, self.parser.end)


class ParserTestAdvanced(unittest.TestCase):

    def test_advanced_grammar(self):
        class Parser(P, terminal='A'):

            A =\
                ('B A', get_string_of_instructions_reducer('A1')),\
                ('C A', get_string_of_instructions_reducer('A2')),\
                ('e', get_string_of_instructions_reducer('A3'))
            B =\
                ('a b c', get_string_of_instructions_reducer('B1')),\
                ('b c? a', get_string_of_instructions_reducer('B2'))
            C =\
                ('x Y B', get_string_of_instructions_reducer('C1')),\
                ('x y d', get_string_of_instructions_reducer('C2'))
            Y =\
                ('', get_string_of_instructions_reducer('Y1')),\
                ('y', get_string_of_instructions_reducer('Y2'))

        parser = Parser()

        tokens = generate_token_stream(
            ('x', 'x'),
            ('b', 'b'),
            ('a', 'a'),
            ('x', 'x'),
            ('y', 'y'),
            ('a', 'a'),
            ('b', 'b'),
            ('c', 'c'),
            ('e', 'e'),
        )

        expected = 'x Y1 b None a B2 C1 x y Y2 a b c B1 C1 e A3 A2 A2'

        for tk in tokens:
            parser.parse(tk)

        res = parser.end()

        self.assertEqual(res, expected)

    def test_closing_empty_rule(self):
        """
        This test is meant to make sure that closure is built correctly internally.
        When since END can be empty, after encountering c, the parser has to dig deeper to figure out that it must
        reduce C1, END1, B1 when lookahead it e.
        """

        class Parser(P, terminal='A'):

            A = 'B e', get_string_of_instructions_reducer('A1')
            B = 'C END', get_string_of_instructions_reducer('B1')
            C = 'c', get_string_of_instructions_reducer('C1')
            END =\
                ('', get_string_of_instructions_reducer('END1')),\
                ('end', get_string_of_instructions_reducer('END2'))

        parser = Parser()

        tokens = generate_token_stream(('c', 'c'), ('e', 'e'))

        for tk in tokens:
            parser.parse(tk)

        res = parser.end()
        expected = 'c C1 END1 B1 e A1'

        self.assertEqual(res, expected)

    def test_sequence_of_empty_rules(self):

        class Parser(P, terminal='start'):

            start = 'A B C', get_string_of_instructions_reducer('start1')
            A = '', get_string_of_instructions_reducer('A1')
            B = '', get_string_of_instructions_reducer('B1')
            C = 'c?', get_string_of_instructions_reducer('C1')

        parser = Parser()

        res = parser.end()
        expected = 'A1 B1 None C1 start1'

        self.assertEqual(res, expected)

        parser.reset()

        token = generate_token('c', 'c')
        parser.parse(token)

        res = parser.end()
        expected = 'A1 B1 c C1 start1'

        self.assertEqual(res, expected)

    def test_multiple_terminals(self):
        class Parser(P, terminal=('start', 'other_start')):
            start = 'a b c', get_string_of_instructions_reducer('start1')
            other_start = 'b c d', get_string_of_instructions_reducer('other_start1')

        parser = Parser()

        start_stream = generate_token_stream(
            ('a', 'a'),
            ('b', 'b'),
            ('c', 'c')
        )

        other_start_stream = generate_token_stream(
            ('b', 'b'),
            ('c', 'c'),
            ('d', 'd')
        )

        for tk in start_stream:
            parser.parse(tk)

        res = parser.end()
        expected = 'a b c start1'

        self.assertEqual(res, expected)

        parser.reset()

        for tk in other_start_stream:
            parser.parse(tk)

        res = parser.end()
        expected = 'b c d other_start1'

        self.assertEqual(res, expected)

    def test_multiple_overlapping_terminals(self):
        class Parser(P, terminal=('start', 'other_start')):

            start = 'a other_start', get_string_of_instructions_reducer('start1')
            other_start = 'b c', get_string_of_instructions_reducer('other_start1')

        parser = Parser()

        start_stream = generate_token_stream(
            ('a', 'a'),
            ('b', 'b'),
            ('c', 'c')
        )

        other_start_stream = generate_token_stream(
            ('b', 'b'),
            ('c', 'c')
        )

        for tk in start_stream:
            parser.parse(tk)

        res = parser.end()
        expected = 'a b c other_start1 start1'

        self.assertEqual(res, expected)

        parser.reset()

        for tk in other_start_stream:
            parser.parse(tk)

        res = parser.end()
        expected = 'b c other_start1'

        self.assertEqual(res, expected)


class ParserTestConflicts(unittest.TestCase):

    def test_reduce_reduce_basic(self):

        class Parser(P, terminal='start'):

            start = ('a', placeholder_reducer),\
                    ('b', placeholder_reducer)
            a = 'A B C', placeholder_reducer
            b = 'A B C', placeholder_reducer

        try:
            parser = Parser()
            self.assertTrue(False, msg='Parser did not detect reduce/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_reduce_reduce())

    def test_reduce_reduce_empty_list(self):

        class Parser(P, terminal='list'):

            list =\
                ('list_of_letters', placeholder_reducer),\
                ('list_of_numbers', placeholder_reducer)
            list_of_letters =\
                ('', placeholder_reducer),\
                ('LETTER list_of_letters', placeholder_reducer)
            list_of_numbers =\
                ('', placeholder_reducer),\
                ('NUMBER list_of_numbers', placeholder_reducer)

        try:
            parser = Parser()
            self.assertTrue(False, msg='Parser did not detect reduce/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_reduce_reduce())

    def test_fixed_empty_list(self):

        class Parser(P, terminal='list'):

            list =\
                ('list_of_letters', placeholder_reducer),\
                ('list_of_numbers', placeholder_reducer)
            list_of_letters = 'LETTER list_of_letters?', placeholder_reducer
            list_of_numbers = 'NUMBER list_of_numbers?', placeholder_reducer

        try:
            parser = Parser()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_shift_reduce_dangling_else(self):

        class Parser(P, terminal='stat'):

            stat =\
                ('IF stat ELSE stat', placeholder_reducer),\
                ('IF stat', placeholder_reducer)

        try:
            parser = Parser()
            self.assertTrue(False, msg='Parser did not detect shift/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 1)
            self.assertTrue(e.conflicts[0].is_shift_reduce())

    def test_fixed_dangling_else(self):

        class Parser(P, terminal='stat'):

            stat = \
                ('IF stat ELSE stat SEMICOLON', placeholder_reducer),\
                ('IF stat SEMICOLON', placeholder_reducer)

        try:
            parser = Parser()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_shift_reduce_arithmetic(self):

        class Parser(P, terminal='exp'):

            exp =\
                ('exp PLUS exp', placeholder_reducer),\
                ('exp TIMES exp', placeholder_reducer),\
                ('LEFT_PAR exp RIGHT_PAR', placeholder_reducer),\
                ('NUMBER', placeholder_reducer),\
                ('VARIABLE', placeholder_reducer)

        try:
            parser = Parser()
            self.assertTrue(False, msg='Parser did not detect shift/reduce conflict')

        except GrammarError as e:
            self.assertEqual(len(e.conflicts), 8)
            for c in e.conflicts:
                self.assertTrue(c.is_shift_reduce())

    def test_fixed_arithmetic(self):

        class Parser(P, terminal='exp'):
            exp =\
                ('factor PLUS factor', placeholder_reducer),\
                ('factor', placeholder_reducer)
            factor =\
                ('atomic TIMES atomic', placeholder_reducer),\
                ('atomic', placeholder_reducer)
            atomic =\
                ('NUMBER', placeholder_reducer),\
                ('VARIABLE', placeholder_reducer),\
                ('LEFT_PAR exp RIGHT_PAR', placeholder_reducer)

        try:
            parser = Parser()
        except GrammarError:
            self.assertTrue(False, msg='Parser detected wrong conflict')

    def test_reduction_cycle(self):

        class Parser(P, terminal='A'):
            A = 'B', placeholder_reducer
            B = 'C', placeholder_reducer
            C = 'A', placeholder_reducer

        try:
            parser = Parser()
            self.assertTrue(False, msg='Parser did not detect reduce cycle')

        except GrammarError as e:
            self.assertEqual(len(e.reduce_cycles), 1)


class ParserTestBuild(unittest.TestCase):

    def test_terminal_token_given(self):
        class Parser(P):
            A = 'a', placeholder_reducer

        self.assertRaises(ParserBuildError, Parser)

    def test_terminal_in_rules(self):
        class Parser(P, terminal='B'):
            A = 'a', placeholder_reducer

        self.assertRaises(ParserBuildError, Parser)


class ParserTestReset(unittest.TestCase):

    def test_reset(self):
        class Parser(P, terminal='start'):

            start = 'a b c', lambda *args: 'success'

        parser = Parser()

        partial_stream = generate_token_stream('a', 'b')
        full_stream = generate_token_stream('a', 'b', 'c')

        for tk in partial_stream:
            parser.parse(tk)

        self.assertRaises(ParserSyntaxError, parser.end)
        parser.reset()

        for tk in full_stream:
            parser.parse(tk)

        self.assertEqual('success', parser.end())


class ParserTestSave(ParserTestBasic):
    """
    Rerun the tests from LexerTestBasic but by saving and loading the created lexer before tests
    """
    parser_filename = "test_parser_save.p"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.parser.save(cls.parser_filename)

    def tearDown(self):
        self.__class__.parser = P.load(self.parser_filename)


class ParserRealExample(unittest.TestCase):
    def test_real_example(self):
        import compyl

        class Lexer(compyl.Lexer, line_rule='\n'):
            IF = r'if'
            THEN = r'then'
            ELSE = r'else'
            END = r'end'
            EQUAL = r'='
            INT = r'[0-9]+'
            VAR = r'[a-zA-Z]+'
            _ = '[ \t]'

        lexer = Lexer()

        class Code(ComparableNode):
            def __init__(self, stats):
                self.stats = stats

        class If(ComparableNode):
            def __init__(self, *args):
                self.cond = args[1]
                self.if_stat = args[3]
                self.else_stat = args[4]

        class Declaration(ComparableNode):
            def __init__(self, *args):
                self.var = args[0]
                self.value = args[2]

        import compyl

        class Parser(compyl.Parser, terminal='code'):
            code = 'stats', Code,
            stats = \
                ('stat', lambda x: [x]), \
                ('stat stats', lambda x, y: [x] + y)
            stat = \
                ('IF VAR THEN stat else_block? END', If), \
                ('VAR EQUAL INT', Declaration)
            else_block = 'ELSE stat', lambda x, y: y

        parser = Parser()

        lexer.read("""
        x = 1
        if x then y = 1 else y = 4 end
        """)

        for token in lexer:
            parser.parse(token)

        output = parser.end()  # Code object

        Token = LooseComparisonToken

        expected = Code([
            Declaration(Token('VAR', 'x'), Token('EQUAL', '='), Token('INT', '1')),
            If(Token('IF', 'if'), Token('VAR', 'x'), Token('THEN', 'then'),
               Declaration(Token('VAR', 'y'), Token('Equal', '='), Token('INT', '1')),
               Declaration(Token('VAR', 'y'), Token('Equal', '='), Token('INT', '4')),
               Token('END', 'end'))
        ])

        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)

