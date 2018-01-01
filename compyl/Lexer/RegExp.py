import copy
import re
from collections import OrderedDict

from compyl.Lexer.IntervalOperations import inverse_intervals_list, merge_intervals, get_minimal_covering_intervals


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


class RegexpTreeException(Exception):
    pass


class RegexpParsingException(Exception):
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
            raise RegexpTreeException("node type (first arg) must be 'single', 'union' or 'kleene'")

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
            raise RegexpTreeException("found loop in RegexpTree while deepcopying")
        else:
            memo[id(self)] = self

        if self.type == 'single':
            dup = RegexpTree(
                'single',
                self.min_ascii,
                self.max_ascii,
                self.next.__deepcopy__({}) if self.next else None
            )

        elif self.type == 'union':
            dup = RegexpTree(
                'union',
                self.fst.__deepcopy__({}),
                self.snd.__deepcopy__({}),
                self.next.__deepcopy__({}) if self.next else None
            )

        elif self.type == 'kleene':
            dup = RegexpTree(
                'kleene',
                self.pattern.__deepcopy__({}),
                self.next.__deepcopy__({}) if self.next else None
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
            raise RegexpTreeException("Unrecognized RegExpTree type when calculating length")

        next_min, next_max = get_length(self.next)

        return min_len + next_min, max_len + next_max

    def print_regexp(self):
        """
        Return the corresponding regexp as string
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
            raise RegexpTreeException("node is of unexpected type")

        return exp if self.next is None else (exp + self.next.print_regexp())


def format_regexp(regexp):
    """
    Parse a regular expression and return it as a RegexpTree object
    """
    return Parser.parse(regexp)


def regexp_to_regexp_tree_list(regexp, pos=0):
    """
    Parse a regular expression and return it as a list of RegexpTree objects
    """
    nodes_list = []
    regexp_length = len(regexp)

    while pos < regexp_length:
        new_nodes, new_pos = get_next_regexp_tree_token(regexp, pos=pos, nodes_list=nodes_list)
        pos = new_pos
        nodes_list += new_nodes

    return nodes_list


def get_next_regexp_tree_token(regexp, pos=0, nodes_list=None):
    """
    Get the next regexp element(s) and return them as well as the new position in the string.
    Note: This return the next token based on a single lookahead. By example "a*c" will only see a, two call will be
          needed to see a*, thus it is needed to pass a nodes_list to be able to mutate the last element if, by example,
          "*" or "+" are seen
    Note2: When encountering |, the method parses the rest of the regexp since it needs the following tokens to form the
           union
    :param regexp: The regexp as string
    :param pos: the position where to start
    :param nodes_list: Previous tokens processed in the regexp, if a look-behind is necessary, the last element will be
                       poped out, mutating the nodes_list and using the poped node to build a new node.
    :return: The list of new RegexpTree nodes and the next position
    """
    # Will store the node if a single node is to be returned
    node = None

    if regexp[pos] == "\\":
        intervals, pos = get_escaped_ascii(regexp, pos + 1)
        node = reduce_interval_list_to_regexp_tree_union(intervals)
        nodes = [node]

    elif regexp[pos] == "[":
        end_pos = find_next_non_escaped_char("]", regexp, beg=pos + 1)

        if end_pos:
            node = get_regexptree_union_from_set(regexp[pos + 1:end_pos])
            pos += end_pos - pos + 1
        else:
            raise RegexpParsingException("bad set syntax, expected ]")

        nodes = [node]

    elif regexp[pos] == "(":
        end_pos = find_matching_closing_parenthesis(regexp, beg=pos + 1)
        sub_regexp = regexp[pos + 1:end_pos]

        node = format_regexp(sub_regexp)
        pos += end_pos - pos + 1
        nodes = [node]

    elif regexp[pos] == "*":
        node = RegexpTree(
            'kleene',
            nodes_list.pop()
        )
        pos += 1
        nodes = [node]

    elif regexp[pos] == "+":
        first_occurence = nodes_list.pop()
        kleene_component = RegexpTree('kleene', copy.deepcopy(first_occurence))
        first_occurence.extend(kleene_component)
        pos += 1
        nodes = [first_occurence]

    elif regexp[pos] == "?":
        try:
            node_to_repeat = nodes_list.pop()
        except IndexError:
            raise RegexpParsingException("bad syntax, '?' without token")

        node = repeat_regexptree(node_to_repeat, 0, 1)
        pos += 1

        nodes = [node]

    elif regexp[pos] == "{":
        end_pos = regexp.find("}", pos + 1)
        min_max = regexp[pos + 1: end_pos].split(',')
        length = len(min_max)

        if length == 1:
            min = max = int(min_max[0])

        elif length == 2:
            min = int(min_max[0])
            max = int(min_max[1])

        else:
            raise RegexpParsingException("bad syntax for min-max repetition")

        if 0 <= min <= max and max > 0:
            try:
                node_to_repeat = nodes_list.pop()
            except IndexError:
                raise RegexpParsingException("bad syntax, repetition without token")

            node = repeat_regexptree(node_to_repeat, min, max)
            pos += end_pos - pos + 1

        else:
            RegexpParsingException("bad interval for min-max repetition")

        nodes = [node]

    elif regexp[pos] == "|":
        pos += 1

        # Since we need the next token to create the union RegexpTree, we parse the rest of the regexp and later we
        # form the union and return the whole list
        next_nodes = regexp_to_regexp_tree_list(regexp, pos=pos)

        try:
            union_fst = nodes_list.pop()
            union_snd = next_nodes.pop(0)
        except IndexError:
            raise RegexpParsingException("bad syntax, unexpected |")

        # We just parsed the whole regexp, thus we are done
        pos = len(regexp)

        node = RegexpTree(
            'union',
            union_fst,
            union_snd,
        )

        nodes = [node] + next_nodes

    elif regexp[pos] == ".":
        node = RegexpTree(
            'union',
            RegexpTree('single', 0, 9),
            RegexpTree('single', 11, 255)
        )
        pos += 1
        nodes = [node]

    elif regexp[pos] == "_":
        node = RegexpTree('single', 0, 255)
        pos += 1
        nodes = [node]

    else:
        char = ord(regexp[pos])
        node = RegexpTree('single', char, char)
        pos += 1
        nodes = [node]

    return nodes, pos


def get_escaped_ascii(regexp, pos):
    """
    Return the ascii values of an escaped char, i.e. preceded by a \
    In particular, regexp should be build using raw strings, thus we need to implement the python built-in escape
    sequences
    We also implement special sequences with the same convention as they are in the re Python module
    The function has two return values. The first is a list of intervals of ascii values, the second is the new position
    """
    escape = {"a": "\a",
              "b": "\b",
              "f": "\f",
              "n": "\n",
              "r": "\r",
              "t": "\t",
              "v": "\v"
              }

    char = regexp[pos]

    if char in escape:
        ascii = ord(escape[char])
        ascii_list = [(ascii, ascii)]
        new_pos = pos + 1

    elif char == "x":
        # Hexadecimal
        try:
            ascii = int(regexp[pos + 1: pos + 3], 16)
            ascii_list = [(ascii, ascii)]
            new_pos = pos + 3
        except (ValueError, IndexError):
            raise RegexpParsingException("bad hexadecimal syntax, must be of format \\xhh")

    elif char == "s":
        # Whitespaces
        ascii_list = [(9, 13), (32, 32)]
        new_pos = pos + 1

    elif char == "S":
        # Non-whitespaces
        ascii_list = [(0, 8), (14, 31), (33, 255)]
        new_pos = pos + 1

    elif char == "w":
        # Alphanumerical and underscore
        ascii_list = [(48, 57), (65, 90), (95, 95), (97, 122)]
        new_pos = pos + 1

    elif char == "W":
        # Non-alphanumerical-or-underscore
        ascii_list = [(0, 47), (58, 64), (91, 94), (96, 96), (123, 255)]
        new_pos = pos + 1

    elif char == "d":
        # Digits
        ascii_list = [(48, 57)]
        new_pos = pos + 1

    elif char == "D":
        # Non-digits
        ascii_list = [(0, 47), (58, 255)]
        new_pos = pos + 1

    else:
        ascii = ord(char)
        ascii_list = [(ascii, ascii)]
        new_pos = pos + 1

    return ascii_list, new_pos


def get_regexptree_union_from_set(inner_set):
    """
    Given the inner part of a set ([...]) in a regexp, return the corresponding RegexpTree object
    """
    if inner_set:
        length = len(inner_set)
        pos = 0
        intervals = []
        invert_set = False

        if inner_set[pos] == "^":
            invert_set = True
            pos += 1

        while pos < length:
            if inner_set[pos] == "\\":
                escaped_intervals, pos = get_escaped_ascii(inner_set, pos)
                intervals.extend(escaped_intervals)

            elif inner_set[pos] == "-":
                try:
                    previous = intervals.pop()
                except IndexError:
                    raise RegexpParsingException("bad syntax for range in set, unexpected -")

                if previous[0] == previous[1]:
                    min = previous[0]
                else:
                    raise RegexpParsingException("bad syntax for range in set, unexpected -")

                if inner_set[pos + 1] == "\\":
                    max = ord(inner_set[pos + 2])
                    pos += 3

                else:
                    max = ord(inner_set[pos + 1])
                    pos += 2

                if min <= max:
                    intervals.append((min, max))
                else:
                    raise RegexpParsingException("bad syntax for range x-y, x ascii exceeds y ascii")

            else:
                ascii = ord(inner_set[pos])
                intervals.append((ascii, ascii))
                pos += 1

        # Sort and merge overlapping intervals
        intervals = merge_intervals(intervals)

        if invert_set:
            intervals = inverse_intervals_list(intervals)

        return reduce_interval_list_to_regexp_tree_union(intervals)

    else:
        raise RegexpParsingException("bad set, set cannot be empty")


def find_matching_closing_parenthesis(string, beg=0):
    """
    Find the closing parenthesis starting at 'beg' in a string and return its position. Return None if there is none.
    The character at position 'beg' does not have to be the opening parenthesis.
    """
    depth = 0
    pos = beg

    while True:
        try:
            char = string[pos]
        except IndexError:
            return None

        if char == "(":
            depth += 1
            pos += 1

        elif char == ")":
            if depth == 0:
                return pos

            else:
                pos += 1
                depth -= 1

        else:
            pos += 1


def find_next_non_escaped_char(char, string, beg=0):
    """
    Find the next non escaped (not preceded with \) of the given char (string even though intended to be a single char)
    and return its position. Return None if no match
    """
    non_escaped_pattern = re.compile(r"(?<!\\)" + char)
    match = non_escaped_pattern.search(string, beg)

    if match:
        return match.start()
    else:
        return None


def repeat_regexptree(node, min, max):
    """
    Given a pattern as a RegexpTree (node), return a RegexpTree representing the pattern repeated from min to max times
    """
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


def reduce_interval_list_to_regexp_tree_union(intervals, next=None):
    """
    Given a list of intervals of ascii values, return the RegexpTree corresponding to the union of all those intervals
    """
    length = len(intervals)

    if length == 0:
        return None

    if length == 1:
        return RegexpTree(
            'single',
            intervals[0][0],
            intervals[0][1],
            next
        )

    else:
        return RegexpTree(
            'union',
            RegexpTree(
                'single',
                intervals[0][0],
                intervals[0][1]
            ),
            reduce_interval_list_to_regexp_tree_union(intervals[1:]),
            next
        )


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
        super().__init__(*[(0, 9), (11, 255)])


class Any(CharSet):
    def __init__(self, value):
        super().__init__((0, 255))


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
                    raise RegexpParsingException

                if isinstance(last, Char) and isinstance(next, Char) and last.ascii <= next.ascii:
                    parsed_tokens.append(
                        CharSet((last.ascii, next.ascii))
                    )
                else:
                    raise RegexpParsingException

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
            raise RegexpParsingException

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
        tokens = Tokenizer.tokenize(regexp)

        return cls._parse(tokens, top_level=True)

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
                    raise RegexpParsingException
                else:
                    return cls._concat_nodes(regexp_nodes)

            elif isinstance(tk, Repetition):
                try:
                    repeated_node = regexp_nodes.pop()
                except IndexError:
                    raise RegexpParsingException

                regexp_nodes.append(
                    repeat_regexptree(repeated_node, tk.min, tk.max)
                )

            elif isinstance(tk, Union):
                return RegexpTree(
                    'union',
                    cls._concat_nodes(regexp_nodes),
                    cls._parse(tokens, top_level=top_level)
                )

            else:
                raise RegexpParsingException

        # Consumed all tokens, return only if out of parentheses
        if top_level:
            return cls._concat_nodes(regexp_nodes)
        else:
            raise RegexpParsingException

