import unittest

import copy
from compyl import Lexer, LexerError
from compyl.__lexer.metaclass import MetaLexer

FAIL = False


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


def test_regexp_on_buffer(regexp, buffer):
    class L(Lexer):
        placeholder = regexp

    lexer = L()

    return get_token_stream_values(lexer, buffer)


class LexerTestBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class L(Lexer):
            line_rule('\n')

            WORD = r'[a-z]+'
            UNREACHABLE = 'unreachable'
            NUMBER = r'[0-9]+'
            _ = r' '

        cls.lexer = L()

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

        class L(Lexer):
            line_rule('\n')
            params(letters=0, digits=0)

            _ = r'[a-z]', letter_counter, 'trigger_on_contain'
            _ = r'[0-9]', digit_counter, 'trigger_on_contain'
            TOKEN = r'[a-zA-Z0-9]+'
            TOKEN = r'[a-zA-Z0-9]+END', 'non_greedy'
            _ = r' '

        cls.lexer = L()

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

        class L(Lexer):
            LETTER = r'[a-zA-Z]'
            _ = r'\n'
            _ = r'\n', increment_line, 'trigger_on_contain'
            _ = r'[a-zA-Z]', skip_next, 'trigger_on_contain'

        cls.lexer = L()

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
        class L(Lexer):
            line_rule('\n')

            WORD = r'\w+'
            _ = r' |\t'

        rules = [
            (r'\w+', 'WORD'),
            (r' |\t', None)
        ]

        cls.lexer = L()

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
        class L(Lexer):
            DOT = r'.'

        lexer = L()

        lexer.read('a\n')

        tk = lexer.lex()
        self.assertEqual(tk.value, 'a')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_any(self):
        class L(Lexer):
            ANY = r'_'

        lexer = L()

        token_values = get_token_stream_values(lexer, '&F=\n')
        self.assertEqual(token_values, ['&', 'F', '=', '\n'])

    def test_match_kleene(self):
        # The x before the MANY_A rule is needed as regexp must have minimal length over 0
        class L(Lexer):
            MANY_A = r'xa*'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'xxaaa')
        self.assertEqual(token_values, ['x', 'xaaa'])

    def test_match_plus(self):
        class L(Lexer):
            MANY_A = r'a+'
            X_WITH_A = r'xa+'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'aaa')
        self.assertEqual(token_values, ['aaa'])

        lexer.read('x')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_optional(self):
        class L(Lexer):
            X_MAYBE_A = r'xa?'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'xxa')
        self.assertEqual(token_values, ['x', 'xa'])

    def test_match_amount(self):
        class L(Lexer):
            THREE_A = r'a{3}'

        lexer = L()

        lexer.read('aaaaa')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aaa')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_min_max_amount(self):
        class L(Lexer):
            THREE_A = r'a{2, 3}x'

        lexer = L()

        lexer.read('aaaxaaxax')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aaax')
        tk = lexer.lex()
        self.assertEqual(tk.value, 'aax')
        self.assertRaises(LexerError, lexer.lex)

    def test_match_or(self):
        class L(Lexer):
            A_OR_B = r'A|B'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'AB')
        self.assertEqual(token_values, ['A', 'B'])

    def test_match_set(self):
        class L(Lexer):
            ABC = r'[abc]'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'abc')
        self.assertEqual(token_values, ['a', 'b', 'c'])
        self.assertRaises(LexerError, get_token_stream, lexer, 'd')

    def test_match_set_inverse(self):
        class L(Lexer):
            NOT_A = r'[^a]'

        lexer = L()

        token_values = get_token_stream_values(lexer, 'bcd')
        self.assertEqual(token_values, ['b', 'c', 'd'])
        self.assertRaises(LexerError, get_token_stream, lexer, 'a')

    def test_match_escape(self):
        class L(Lexer):
            STAR = r'\*'

        lexer = L()

        token_values = get_token_stream_values(lexer, '*')
        self.assertEqual(token_values, ['*'])

    def test_match_escape_space(self):
        class L(Lexer):
            SPACE = r'\s'

        lexer = L()

        buffer = ' \t\n\r\f\v'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_space(self):
        class L(Lexer):
            NOT_SPACE = r'\S'

        lexer = L()

        buffer = 'abc'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in ' \t\n\r\f\v':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)

    def test_match_escape_alphanum(self):
        class L(Lexer):
            ALPHNUM = r'\w'

        lexer = L()

        buffer = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_alphanum(self):
        class L(Lexer):
            NOT_ALPHANUM = r'\W'

        lexer = L()

        buffer = '.:+!@#:'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)

    def test_match_escape_num(self):
        class L(Lexer):
            NUM = r'\d'

        lexer = L()

        buffer = '123456789'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))

    def test_match_escape_non_num(self):
        class L(Lexer):
            NOT_NUM = r'\D'

        lexer = L()

        buffer = '.:+!@#:ABCabcxyzXYZ'
        token_values = get_token_stream_values(lexer, buffer)
        self.assertEqual(token_values, list(buffer))
        for c in '123456789':
            lexer.pos = 0
            lexer.buffer = ''
            self.assertRaises(LexerError, get_token_stream, lexer, c)


class LexerTestRegexpPriority(unittest.TestCase):

    def test_priority(self):
        """
        Challenge the RegExp module on various corner case of regexp operation priority
        """

        expected_rule_buffer_outputs = [
            (r'abc|def', 'abc', ['abc']),
            (r'abc|def', 'def', ['def']),
            (r'abc|def', 'abdef', FAIL),
            (r'ab(c|d)ef', 'abcef', ['abcef']),
            (r'ab(c|d)ef', 'abc', FAIL),
            (r'ab(c|d)ef', 'def', FAIL),
            (r'a|b+', 'a', ['a']),
            (r'a|b+', 'b', ['b']),
            (r'a|b+', 'bb', ['bb']),
            (r'a|b+', 'bbb', ['bbb']),
            (r'a|b+', 'aa', ['a', 'a']),
            (r'a+|b+', 'aaa', ['aaa']),
            (r'a+|b+', 'bb', ['bb']),
            (r'a+|b+', 'aabb', ['aa', 'bb']),
            (r'(a|b)+', 'a', ['a']),
            (r'(a|b)+', 'b', ['b']),
            (r'(a|b)+', 'aba', ['aba']),
            (r'ab{1,3}', 'ab', ['ab']),
            (r'ab{1,3}', 'abb', ['abb']),
            (r'ab{1,3}', 'abbb', ['abbb']),
            (r'ab{1,3}', 'abbbb', FAIL),
            (r'(a|b){1,2}', 'ab', ['ab']),
            (r'(a|b){1,2}', 'ba', ['ba']),
            (r'(a|b){1,2}', 'b', ['b']),
            (r'(a|b){1,2}', 'aaa', ['aa', 'a']),
            (r'a{2}+', 'aa', ['aa']),
            (r'a{2}+', 'aaaa', ['aaaa']),
            (r'a+{2}', 'aa', ['aa']),
            (r'a+{2}', 'aaa', ['aaa']),
            (r'a+{2}', 'a', FAIL),
            (r'(a*b)+', 'bbabaab', ['bbabaab']),
            (r'[a-z][A-Z]+', 'xXX', ['xXX']),
            (r'[a-z][A-Z]+', 'xXxXX', ['xX', 'xXX']),
            (r'([a-z][A-Z])+', 'xXxX', ['xXxX']),
            (r'([a-z][A-Z])+', 'xXX', FAIL),
            (r'([a-z][A-Z])+', 'XxX', FAIL),
            (r'a([bc]d)*', 'abd', ['abd']),
            (r'a([bc]d)*', 'a', ['a']),
            (r'a([bc]d)*', 'abdcdbd', ['abdcdbd']),
            (r'a([bc]d)*', 'abdacdbd', ['abd', 'acdbd']),
            (r'a([bc]d)*', 'add', FAIL),
            (r'a([bc]d)*', 'abc', FAIL),
            (r'a|b|c', 'a', ['a']),
            (r'a|b|c', 'b', ['b']),
            (r'a|b|c', 'c', ['c']),
            (r'a|b|c', 'ac', ['a', 'c']),
            (r'(a|b)|c', 'a', ['a']),
            (r'(a|b)|c', 'b', ['b']),
            (r'(a|b)|c', 'c', ['c']),
            (r'(a|b)|c', 'ac', ['a', 'c']),
            (r'xa?|bc', 'x', ['x']),
            (r'xa?|bc', 'xa', ['xa']),
            (r'xa?|bc', 'bc', ['bc']),
            (r'xa?|bc', 'xbc', ['x', 'bc']),
            (r'[^\W]\++(j|l?)?', 'a+', ['a+']),
            (r'[^\W]\++(j|l?)?', 'b++j', ['b++j']),
            (r'[^\W]\++(j|l?)?', 'c+++l', ['c+++l']),
            (r'[^\W]\++(j|l?)?', 'a+a+', ['a+', 'a+']),
            (r'[^\W]\++(j|l?)?', 'a+ja+', ['a+j', 'a+']),
            (r'[^\W]\++(j|l?)?', '$+j', FAIL),
            (r'[^\W]\++(j|l?)?', 'w+jl', FAIL),
        ]

        for rules, buffer, expected in expected_rule_buffer_outputs:
            if expected is not FAIL:
                self.assertEqual(
                    test_regexp_on_buffer(rules, buffer),
                    expected
                )
            else:
                self.assertRaises(
                    LexerError,
                    test_regexp_on_buffer, rules, buffer
                )


class LexerTestTerminalAction(unittest.TestCase):
    def test_single_terminal_action(self):

        def count_all_matches(t):
            t.params['count'] += 1

        class L(Lexer):
            line_rule('\s')
            terminal_actions(count_all_matches)
            params({'count': 0})

            WORD = r'\w+'

        lexer = L()

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

        class L(Lexer):
            line_rule(r'\s')
            terminal_actions(count_all_matches, count_all_matches_again)
            params(count=0)

            WORD = r'\w+'

        lexer = L()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 10)

    def test_terminal_action_always(self):

        def count_all_matches(t):
            t.params['count'] += 1

        class L(Lexer):
            line_rule(r'\s')
            terminal_actions((count_all_matches, 'always'))
            params(count=0)

            WORD = r'\w+'

        lexer = L()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 5)

    def test_terminal_action_on_ignored(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_ignored(t):
            t.params['count'] += 1

        class L(Lexer):
            line_rule(r'\s')
            terminal_actions((count_all_ignored, 'only_ignored'))
            params({'count': 0})

            WORD = r'\w+'

        lexer = L()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 2)

    def test_terminal_action_on_tokens(self):

        def count_all_tokens(t):
            t.params['count'] += 1

        class L(Lexer):
            line_rule(r'\s')
            terminal_actions((count_all_tokens, 'only_tokens'))
            params({'count': 0})

            WORD = r'\w+'

        lexer = L()

        get_token_stream(lexer, 'word word word')
        self.assertEqual(lexer.params['count'], 3)

    def test_action_when_choices(self):
        rules = [
            (r'\w+', 'WORD'),
        ]

        def count_all_tokens(t):
            pass

        try:
            class L(Lexer):
                terminal_actions = [(count_all_tokens, 'only_foo')]

                WORD = r'\w+'

            lexer = L()

            self.assertTrue(False, 'only_foo should not be a valid terminal action tag')

        except LexerError:
            self.assertTrue(True)


class LexerTestBuild(unittest.TestCase):

    def test_regexp_minimum_length(self):

        class L(Lexer):
            A = r'a*'

        self.assertRaises(LexerError, L)

    def test_special_action_choices(self):

        try:
            class L(Lexer):
                A = r'a', 'foo'

            lexer = L()

            self.assertTrue(False, 'foo should not be a valid special action tag')

        except LexerError:
            self.assertTrue(True)


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


class LexerInstanceTest(unittest.TestCase):
    """
    Tests to make sure metaclassing keeps type tree consistent
    """

    def test_lexer_isinstance_of_Lexer(self):
        class L(Lexer):
            A = r'a'

        self.assertTrue(isinstance(L(), Lexer))

    def test_lexer_isinstance_of_L(self):
        class L(Lexer):
            A = r'a'

        self.assertTrue(isinstance(L(), L))

    def test_lexer_not_isinstance_of_metaclass(self):
        class L(Lexer):
            A = r'a'

        self.assertFalse(isinstance(L(), MetaLexer))


if __name__ == '__main__':
    unittest.main(verbosity=2)
