import unittest
import copy

from compyl.Lexer.Lexer import Lexer, LexerError
from compyl.Lexer.FiniteAutomaton import FiniteAutomatonError


def get_token_stream(lexer, buffer):
    lexer.read(buffer)
    tk_list = []

    tk = lexer.lex()

    while tk:
        tk_list.append(tk)
        tk = lexer.lex()

    return tk_list


def get_token_stream_types(lexer, buffer):
    return [tk.type for tk in get_token_stream(lexer, buffer)]


def get_token_stream_values(lexer, buffer):
    return [tk.value for tk in get_token_stream(lexer, buffer)]


class LexerTestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = [
            (r'[a-z]+', 'WORD'),
            (r'unreachable', 'UNREACHABLE'),
            (r'[0-9]+', 'NUMBER'),
            (r' ', None),
        ]

        cls.lexer = Lexer(rules=rules, line_rule='\n')
        cls.lexer.build()

        cls.lexer_copy = copy.deepcopy(cls.lexer)

    def tearDown(self):
        self.__class__.lexer = copy.deepcopy(self.lexer_copy)

    def test_token_stream(self):
        buffer = "11 words written here\n" + \
                 "unreachable should be a word\n"

        token_types = get_token_stream_types(self.lexer, buffer)
        expected_types = ['NUMBER'] + ['WORD'] * 8

        self.assertEqual(token_types, expected_types)

    def test_token_empty_stream(self):
        buffer = ""

        token_types = get_token_stream_types(self.lexer, buffer)
        expected_types = []

        self.assertEqual(token_types, expected_types)

    def test_syntax_error(self):
        buffer = "words then some forbidden token ?"
        self.assertRaises(LexerError, get_token_stream, self.lexer, buffer)


class LexerTestSpecialActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def letter_counter(t):
            t.params['letters'] += 1

        def digit_counter(t):
            t.params['digits'] += 1

        rules = [
            (r'[a-z]', letter_counter, 'trigger_on_contain'),
            (r'[0-9]', digit_counter, 'trigger_on_contain'),
            (r'[a-zA-Z0-9]+', 'TOKEN'),
            (r'[a-zA-Z0-9]+END', 'TOKEN', 'non_greedy'),
            (r' ', None)
        ]

        cls.lexer = Lexer(rules=rules, line_rule='\n', params={'letters': 0, 'digits': 0})
        cls.lexer.build()

        cls.lexer_copy = copy.deepcopy(cls.lexer)

    def tearDown(self):
        self.__class__.lexer = copy.deepcopy(self.lexer_copy)

    def test_trigger_on_contain(self):
        buffer = 'foo bar 1 test 567'
        get_token_stream(self.lexer, buffer)

        self.assertEqual(self.lexer.params['letters'], 10)
        self.assertEqual(self.lexer.params['digits'], 4)

    def test_non_greedy(self):
        buffer = 'fooENDbar'
        token_types = get_token_stream_types(self.lexer, buffer)
        expected = ['TOKEN', 'TOKEN']

        self.assertEqual(token_types, expected)

    def test_sample(self):
        buffer = ' foo bar\n' + 'bar foo '
        token_types = get_token_stream_types(self.lexer, buffer)
        expected = ['TOKEN'] * 4

        self.assertEqual(token_types, expected)


class LexerTestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def skip_next(t):
            t.increment_pos()

        def increment_line(t):
            t.increment_line()

        rules = [
            (r'[a-zA-Z]', 'LETTER'),
            (r'\n', None),
            (r'\n', increment_line, 'trigger_on_contain'),
            (r'[a-zA-Z]', skip_next, 'trigger_on_contain')
        ]

        cls.lexer = Lexer(rules=rules)
        cls.lexer.build()

        cls.lexer_copy = copy.deepcopy(cls.lexer)

    def tearDown(self):
        self.__class__.lexer = copy.deepcopy(self.lexer_copy)

    def test_increment_pos(self):
        buffer = 'A1V3K6'

        token_types = get_token_stream_types(self.lexer, buffer)
        expected = ['LETTER'] * 3

        self.assertEqual(token_types, expected)

    def test_increment_line(self):
        buffer = '\n\n'

        self.assertEqual(self.lexer.lineno, 1)

        token_types = get_token_stream(self.lexer, buffer)
        expected = []

        self.assertEqual(token_types, expected)
        self.assertEqual(self.lexer.lineno, 3)


class LexerTestLineRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = [
            (r'\w+', 'WORD'),
            (r' |\t', None)
        ]

        cls.lexer = Lexer(rules=rules, line_rule='\n')
        cls.lexer.build()

        cls.lexer_copy = copy.deepcopy(cls.lexer)

    def tearDown(self):
        self.__class__.lexer = copy.deepcopy(self.lexer_copy)

    def test_line_rule(self):
        buffer = """some code
            some more code
        """

        self.lexer.read(buffer)

        t = self.lexer.lex()
        self.lexer.lex()
        self.assertEqual(self.lexer.lineno, 1)
        self.lexer.lex()
        self.assertEqual(self.lexer.lineno, 2)

        get_token_stream(self.lexer, '')

        self.assertEqual(self.lexer.lineno, 3)


class LexerTestRegexp(unittest.TestCase):
    def test_match_dot(self):
        rules = [
            (r'.', 'DOT')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        lexer.read('a\n')

        tk = lexer.lex()
        self.assertEqual(tk.value, 'a')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_any(self):
        rules = [
            (r'_', 'ANY')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, '&F=\n')
        self.assertEqual(token_values, ['&', 'F', '=', '\n'])

    def test_match_kleene(self):
        # The x before the MANY_A rule is needed as regexp must have minimal length over 0
        rules = [
            (r'xa*', 'MANY_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'xxaaa')
        self.assertEqual(token_values, ['x', 'xaaa'])

    def test_match_plus(self):
        rules = [
            (r'a+', 'MANY_A'),
            (r'xa+', 'X_WITH_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'aaa')
        self.assertEqual(token_values, ['aaa'])

        lexer.read('x')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_optional(self):
        rules = [
            (r'xa?', 'X_MAYBE_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'xxa')
        self.assertEqual(token_values, ['x', 'xa'])

    def test_match_amount(self):
        rules = [
            (r'a{3}', 'THREE_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        lexer.read('aaaaa')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aaa')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_amount(self):
        rules = [
            (r'a{2, 3}x', 'THREE_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        lexer.read('aaaxaaxax')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aaax')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aax')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_or(self):
        rules = [
            (r'A|B', 'A_OR_B')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'AB')
        self.assertEqual(token_values, ['A', 'B'])

    def test_match_set(self):
        rules = [
            (r'[abc]', 'ABC')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'abc')
        self.assertEqual(token_values, ['a', 'b', 'c'])
        self.assertRaises(LexerError, get_token_stream, lexer, 'd')

    def test_match_set_inverse(self):
        rules = [
            (r'[^a]', 'NOT_A')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, 'bcd')
        self.assertEqual(token_values, ['b', 'c', 'd'])
        self.assertRaises(LexerError, get_token_stream, lexer, 'a')

    def test_match_escape(self):
        rules = [
            (r'\*', 'STAR')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        token_values = get_token_stream_values(lexer, '*')
        self.assertEqual(token_values, ['*'])

    def test_match_escape_space(self):
        rules = [
            (r'\s', 'SPACE')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = ' \t\n\r\f\v'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_space(self):
        rules = [
            (r'\S', 'NOT_SPACE')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = 'abc'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in ' \t\n\r\f\v':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)

    def test_match_escape_alphanum(self):
        rules = [
            (r'\w', 'ALPHNUM')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_alphanum(self):
        rules = [
            (r'\W', 'NOT_ALPHANUM')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = '.:+!@#:'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)

    def test_match_escape_num(self):
        rules = [
            (r'\d', 'NUM')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = '123456789'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_num(self):
        rules = [
            (r'\D', 'NOT_NUM')
        ]

        lexer = Lexer(rules=rules)
        lexer.build()

        buffer = '.:+!@#:ABCabcxyzXYZ'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in '123456789':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)


class LexerTestTerminalAction(unittest.TestCase):
    def test_single_terminal_action(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_matches(t):
            t.params['count'] += 1

        lexer = Lexer(rules=rules, line_rule='\s', terminal_actions=[count_all_matches], params={'count': 0})
        lexer.build()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 5)

    def test_two_terminal_action(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_matches(t):
            t.params['count'] += 1

        def count_all_matches_again(t):
            t.params['count'] += 1

        lexer = Lexer(
            rules=rules,
            line_rule='\s',
            terminal_actions=[count_all_matches, count_all_matches_again],
            params={'count': 0}
        )
        lexer.build()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 10)

    def test_terminal_action_always(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_matches(t):
            t.params['count'] += 1

        lexer = Lexer(
            rules=rules,
            line_rule='\s',
            terminal_actions=[(count_all_matches, 'always')],
            params={'count': 0}
        )
        lexer.build()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 5)

    def test_terminal_action_on_ignored(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_ignored(t):
            t.params['count'] += 1

        lexer = Lexer(
            rules=rules,
            line_rule='\s',
            terminal_actions=[(count_all_ignored, 'only_ignored')],
            params={'count': 0}
        )
        lexer.build()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 2)

    def test_terminal_action_on_tokens(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_tokens(t):
            t.params['count'] += 1

        lexer = Lexer(
            rules=rules,
            line_rule='\s',
            terminal_actions=[(count_all_tokens, 'only_tokens')],
            params={'count': 0}
        )
        lexer.build()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 3)

    def test_action_when_choices(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_tokens(t):
            pass

        self.assertRaises(LexerError, Lexer,
                          rules=rules,
                          terminal_actions=[(count_all_tokens, 'only_foo')]
                          )


class LexerTestBuild(unittest.TestCase):

    def test_regexp_minimum_length(self):
        rules = [
            ('a*', 'A')
        ]

        lexer = Lexer(rules=rules)

        self.assertRaises(FiniteAutomatonError, lexer.build)

    def test_special_action_choices(self):
        rules = [
            ('a*', 'A', 'foo')
        ]

        lexer = Lexer(rules=rules)

        self.assertRaises(FiniteAutomatonError, lexer.build)


class LexerTestSave(LexerTestBasic):
    """
    Rerun the tests from LexerTestBasic but by saving and loading the created lexer before tests
    """
    lexer_filename = "test_lexer_save.p"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.lexer.save(cls.lexer_filename)

    def tearDown(self):
        self.__class__.lexer = Lexer.load(self.lexer_filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
