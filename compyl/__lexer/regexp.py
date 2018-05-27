import copy
import re

from compyl.__lexer.interval_operations import inverse_intervals_list, get_minimal_covering_intervals
from compyl.__lexer.errors import RegexpParsingError


# ======================================================================================================================
# Tokenize RegExp
# ======================================================================================================================

# In this module are the tools that take a regular expression as string and transform it to a bare-bone format.
# By bare-bone format we mean that complex regexp operators are reduced to basic regexps, that is characters (intervals
# in our case), union (or) and kleene operator (*).
#
# From this format the algorithm to build a Non-Deterministic Finite Automata is very simple.
#
# Here are some example of how regexps are reduced:
#
# a+ => aa*
# a? => ()|a
# a{2,3} => a(a|aa)     this is not exactly what happens, but this is the idea
# . => (0,9)|(11,255)   this one is a demonstration of how the interval notation works, since . means anything but \n
#                       we convert it to any ascii value that is not \n (10)

MAX_UNICODE = 0x10ffff


class _RegexpTreeException(Exception):
    pass


class RegexpTree:
    """
    A tree structure of a regexp.
    Reduce a regexp to basic regexp tokens, that is characters, unions (or) and kleene operator (*)
    Characters are treated in intervals.
    """

    def __init__(self, node_type, *values):

        if node_type in ['single', 'union', 'kleene']:
            self.type = node_type

        else:
            raise _RegexpTreeException("node type (first arg) must be 'single', 'union' or 'kleene'")

        if node_type == "single":
            self.min_ascii = values[0]
            self.max_ascii = values[1]
            self.next = values[2] if len(values) > 2 else None

        elif node_type == "union":
            self.fst = values[0]
            self.snd = values[1]
            self.next = values[2] if len(values) > 2 else None

        elif node_type == 'kleene':
            self.pattern = values[0]
            self.next = values[1] if len(values) > 1 else None

    def __str__(self):
        return "<RegexpTree '%s'>" % self.type

    def __copy__(self):
        if self.type == 'single':
            dup = RegexpTree(
                'single',
                self.min_ascii,
                self.max_ascii,
                self.next
            )

        elif self.type == 'union':
            dup = RegexpTree(
                'union',
                self.fst,
                self.snd,
                self.next
            )

        elif self.type == 'kleene':
            dup = RegexpTree(
                'kleene',
                self.pattern,
                self.next
            )

        return dup

    def __deepcopy__(self, memo):
        if id(self) in memo:
            raise _RegexpTreeException("found loop in RegexpTree while deepcopying")
        else:
            memo[id(self)] = self

        if self.type == 'single':
            dup = RegexpTree(
                'single',
                self.min_ascii,
                self.max_ascii,
                copy.deepcopy(self.next)
            )

        elif self.type == 'union':
            dup = RegexpTree(
                'union',
                copy.deepcopy(self.fst),
                copy.deepcopy(self.snd),
                copy.deepcopy(self.next)
            )

        elif self.type == 'kleene':
            dup = RegexpTree(
                'kleene',
                copy.deepcopy(self.pattern),
                copy.deepcopy(self.next)
            )

        return dup

    def extend(self, next):
        """
        Add the given RegexpTree at the end of the chain of RegexpTrees starting at self
        """
        if self.next is None:
            self.next = next
        else:
            self.next.extend(next)

    def pop(self):
        """
        Remove the last RegexTree instruction at the end of the chain starting at self
        """
        if self.next is None:
            return None

        elif self.next.next is None:
            pop = self.next
            self.next = None
            return pop

        else:
            return self.next.pop()

    def truncate(self):
        """
        Delete the upcoming part of the RegexpTree chain
        """
        self.next = None

    def length(self):
        """
        Return a tuple of int, the first element is the minimal length of the the regexp, the second is the maximal
        length of the regexp
        """

        def get_length(rg):
            # Used to handle the None
            return rg.length() if rg else (0, 0)

        if self.type == 'single':
            min_len = max_len = 1

        elif self.type == 'union':
            left = get_length(self.fst)
            right = get_length(self.snd)

            min_len = min(left[0], right[0])
            max_len = max(left[1], right[1])

        elif self.type == 'kleene':
            min_len = 0
            max_len = float('inf')

        else:
            raise _RegexpTreeException("Unrecognized RegExpTree type when calculating length")

        next_min, next_max = get_length(self.next)

        return min_len + next_min, max_len + next_max

    def print_regexp(self):
        """
        Return the corresponding regexp as string for debugging purpose
        """
        if self.type == 'single':
            if self.min_ascii == self.max_ascii:
                exp = chr(self.min_ascii)
            else:
                exp = "[%s-%s]" % (chr(self.min_ascii), chr(self.max_ascii))

        elif self.type == 'union':
            if self.fst is None:
                exp = "(%s)?" % self.snd.print_regexp()
            elif self.snd is None:
                exp = "(%s)?" % self.fst.print_regexp()
            else:
                exp = "(%s)|(%s)" % (self.fst.print_regexp(), self.snd.print_regexp())

        elif self.type == 'kleene':
            exp = "(%s)*" % self.pattern.print_regexp()

        else:
            raise _RegexpTreeException("node is of unexpected type")

        return exp if self.next is None else (exp + self.next.print_regexp())


def format_regexp(regexp):
    """
    Parse a regular expression and return it as a RegexpTree object
    """
    return Parser.parse(regexp)


# ======================================================================================================================
# RegExp Dummy Parser
# ======================================================================================================================

class Token:
    pass


class Char(Token):
    def __init__(self, value):
        self.ascii = value if isinstance(value, int) else ord(value)

    @property
    def intervals(self):
        return [(self.ascii, self.ascii)]

    def to_regexp_tree(self):
        return RegexpTree('single', self.ascii, self.ascii)


class StandardEscape(Char):
    escape = {
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v"
    }

    def __init__(self, value):
        super().__init__(self.escape[value[1]])


class CharEscape(Char):
    def __init__(self, value):
        super().__init__(value[1])


class HexEscape(Char):
    def __init__(self, value):
        super().__init__(int(value[2:], 16))


class CharSet(Token):
    def __init__(self, *intervals):
        self.intervals = get_minimal_covering_intervals(list(intervals))

    def to_regexp_tree(self):
        return self._get_intervals_union(copy.copy(self.intervals))

    @classmethod
    def _get_intervals_union(cls, intervals):
        fst = intervals.pop(0)

        if len(intervals) == 0:
            return RegexpTree('single', *fst)

        else:
            return RegexpTree(
                'union',
                RegexpTree('single', *fst),
                cls._get_intervals_union(intervals)
            )

class EscapeSequence(CharSet):

    escape = {
        's': [(9, 13), (32, 32)],
        'S': [(0, 8), (14, 31), (33, 255)],
        'w': [(48, 57), (65, 90), (95, 95), (97, 122)],
        'W': [(0, 47), (58, 64), (91, 94), (96, 96), (123, 255)],
        'd': [(48, 57)],
        'D': [(0, 47), (58, 255)],
    }

    def __init__(self, value):
        super().__init__(*self.escape[value[1]])


class Dot(CharSet):
    def __init__(self, value):
        super().__init__(*[(0, 9), (11, MAX_UNICODE)])


class Any(CharSet):
    def __init__(self, value):
        super().__init__((0, MAX_UNICODE))


class Set(CharSet):
    def __init__(self, value):
        inner = self.get_inner_set(value)
        tokens = self._parse_set(
            Tokenizer.tokenize(inner, where='set')
        )

        intervals = [interval for tk in tokens for interval in tk.intervals]

        super().__init__(*intervals)

    @staticmethod
    def get_inner_set(value):
        return value[1:-1]

    @staticmethod
    def _parse_set(tokens):
        parsed_tokens = []
        pos = 0

        while pos < len(tokens):
            tk = tokens[pos]

            if isinstance(tk, Dash):
                try:
                    last = parsed_tokens.pop()
                    next = tokens[pos + 1]
                except IndexError:
                    raise RegexpParsingError

                if isinstance(last, Char) and isinstance(next, Char) and last.ascii <= next.ascii:
                    parsed_tokens.append(
                        CharSet((last.ascii, next.ascii))
                    )
                else:
                    raise RegexpParsingError

                pos += 2
            else:
                parsed_tokens.append(tk)
                pos += 1

        return parsed_tokens


class InverseSet(Set):
    def __init__(self, value):
        super().__init__(value)
        self.intervals = inverse_intervals_list(self.intervals)

    @staticmethod
    def get_inner_set(value):
        return value[2:-1]


class Repetition(Token):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def repeat_regexptree(self, node):
        """
        Given a pattern as a RegexpTree (node), return a RegexpTree representing the pattern repeated from min to max times
        """
        min = self.min
        max = self.max

        if min > 0:
            first = copy.deepcopy(node)
            last = first
            min -= 1
            max -= 1

        elif max < float('inf'):
            last = copy.deepcopy(node)
            first = RegexpTree(
                'union',
                None,
                last
            )
            max -= 1

        else:
            return RegexpTree(
                'kleene',
                copy.deepcopy(node)
            )

        while min > 0:
            extension = copy.deepcopy(node)
            last.extend(extension)
            last = extension
            min -= 1
            max -= 1

        while 0 < max < float('inf'):
            extension_snd = copy.deepcopy(node)
            extension = RegexpTree(
                'union',
                None,
                extension_snd
            )
            last.extend(extension)
            last = extension_snd

            max -= 1

        if max == float('inf'):
            first.extend(
                RegexpTree('kleene', copy.deepcopy(node))
            )

        return first


class RepetitionExact(Repetition):
    def __init__(self, value):
        n = int(value[1:-1])
        super().__init__(n, n)


class RepetitionRange(Repetition):
    def __init__(self, value):
        min, max = [int(x) for x in value[1:-1].split(',')]
        super().__init__(min, max)


class Star(Repetition):
    def __init__(self, value):
        super().__init__(0, float('inf'))


class Plus(Repetition):
    def __init__(self, value):
        super().__init__(1, float('inf'))


class Optional(Repetition):
    def __init__(self, value):
        super().__init__(0, 1)


class Operator(Token):
    def __init__(self, value):
        pass


class LPar(Operator):
    pass


class RPar(Operator):
    pass


class Union(Operator):
    pass


class Dash(Operator):
    pass


class Tokenizer:
    global_rules = [
        (EscapeSequence, r'\\[sSwWdD]'),
        (StandardEscape, r'\\[abfnrtv]'),
        (HexEscape, r'\\x[0-9a-fA-F]{2}'),
        (CharEscape, r'\\.'),
        (Dot, r'\.'),
        (Any, r'\_'),
        (Plus, r'\+'),
        (Star, r'\*'),
        (Optional, r'\?'),
        (RepetitionRange, r'\{\s*[1-9][0-9]*\s*,\s*[1-9][0-9]*\s*\}'),
        (RepetitionExact, r'\{\s*[1-9][0-9]*\s*\}'),
        (LPar, r'\('),
        (RPar, r'\)'),
        (InverseSet, r'\[\^(?:\\.|[^]\\])*\]'),
        (Set, r'\[(?:\\.|[^]\\])*\]'),
        (Union, r'\|'),
        (Char, r'.'),
    ]

    inner_set_rules = [
        (EscapeSequence, r'\\[sSwWdD]'),
        (StandardEscape, r'\\[abfnrtv]'),
        (HexEscape, r'\\x[0-9a-fA-F]{2}'),
        (CharEscape, r'\\_'),
        (Dash, r'-'),
        (Char, r'.'),
    ]

    @classmethod
    def match(cls, regexp, where=None):
        if where == 'set':
            rules = cls.inner_set_rules
        else:
            rules = cls.global_rules

        for token_cls, r in rules:
            r = re.compile(r, re.DOTALL)
            match = r.match(regexp)
            if match:
                value = match.group()
                return token_cls(value), len(value)
        else:
            raise RegexpParsingError

    @classmethod
    def tokenize(cls, regexp, where=None):
        tokens = []
        while regexp:
            tk, length = cls.match(regexp, where)
            regexp = regexp[length:]
            tokens.append(tk)

        return tokens


class Parser:
    @classmethod
    def parse(cls, regexp):

        try:
            tokens = Tokenizer.tokenize(regexp)

            return cls._parse(tokens, top_level=True)

        except RegexpParsingError:
            raise RegexpParsingError("Syntax error in regexp {}".format(regexp))

    @classmethod
    def _concat_nodes(cls, nodes):
        start = nodes[0] if nodes else None

        for node in nodes[1:]:
            start.extend(node)

        return start

    @classmethod
    def _parse(cls, tokens, top_level=False):

        regexp_nodes = []

        while tokens:

            tk = tokens.pop(0)

            if isinstance(tk, (Char, CharSet)):
                regexp_nodes.append(
                    tk.to_regexp_tree()
                )

            elif isinstance(tk, LPar):
                regexp_nodes.append(
                    cls._parse(tokens)
                )

            elif isinstance(tk, RPar):
                if top_level:
                    raise RegexpParsingError
                else:
                    return cls._concat_nodes(regexp_nodes)

            elif isinstance(tk, Repetition):
                try:
                    repeated_node = regexp_nodes.pop()
                except IndexError:
                    raise RegexpParsingError

                regexp_nodes.append(
                    tk.repeat_regexptree(repeated_node)
                )

            elif isinstance(tk, Union):
                return RegexpTree(
                    'union',
                    cls._concat_nodes(regexp_nodes),
                    cls._parse(tokens, top_level=top_level)
                )

            else:
                raise RegexpParsingError

        # Consumed all tokens, return only if out of parentheses
        if top_level:
            return cls._concat_nodes(regexp_nodes)
        else:
            raise RegexpParsingError
