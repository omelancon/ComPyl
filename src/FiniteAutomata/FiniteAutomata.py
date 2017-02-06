import sre_parse
from itertools import count


class NodeIsNotTerminalState(Exception):
    pass


class NodeAlreadyHasTerminalToken(Exception):
    pass


class FiniteAutomata(object):
    """
    Basic skeleton for Deterministic and Non-deterministic Finite Automata.
    A FiniteAutomata object is a node of the graph representation of the automata.


    """
    # Counter for state id
    _ids = count(0)

    # Special lookout values
    EMPTY = (-1, -1)

    # Special terminal values
    IGNORED = object()

    # Maximum number of repetition of a same lookout handled, anything above is considered as infinite repetition
    max_handled_repeat = 100

    def __init__(self, terminal_token=None, max_handled_repeat=None):
        self.id = self._ids.next()

        # List of next states from current state.
        # Elements of the list are tuples (lookout, next state), lookout takes the range format (min_ascii, max_ascii)
        self.next_states = []

        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token

        # Allows to by pass the default max_repeat_handled of the class
        if max_handled_repeat:
            self.max_repeat_handled = max_handled_repeat

    def __str__(self):
        return "<State '%d'>" % (self.id)

    def add_transition_range(self, min_ascii, max_ascii, *args, **kwargs):
        """
        Add the edge corresponding to the transition when a character from min_ascii to max_ascii is seen.
        For a single ascii transition, let min_ascii = max_ascii, ex: 'a' would be (97, 97)
        Extra arguments and keyword arguments are passed to the object generator.

        REMARK: in the case of a DFA, be aware that add_transition does not check for already existing transitions.
        """
        lookout = (min_ascii, max_ascii)
        new_state = self.__class__(*args, **kwargs)
        self.next_states.append((lookout, new_state))

        return new_state

    def add_empty_transition(self, *args, **kwargs):
        """
        Same ad add_transition, but for an empty string match
        """
        new_state = self.__class__(*args, **kwargs)
        self.next_states.append((self.EMPTY, new_state))

        return new_state

    def add_transition_to_state(self, min_ascii, max_ascii, state):
        """
        Form an edge from the current state to a pre-existing state
        :param max_ascii: int
        :param min_ascii: int
        :param state: a pre-existing node
        :return: None
        """

        self.next_states.append(((min_ascii, max_ascii), state))

    def add_empty_transition_to_state(self, state):
        """
        Same as add_transition_to_state but with empty string
        :param state:
        :return:
        """
        self.next_states.append((self.EMPTY, state))

    def set_terminal_token(self, terminal_token):
        """
        Set the terminal token if it is not already set to another value
        """
        if not self.terminal_exists():
            if terminal_token is None:
                self._set_terminal_to_ignored()
            else:
                self.terminal_token = terminal_token

    def _set_terminal_to_ignored(self):
        """
        Set the terminal token value to ignored
        """
        if not self.terminal_token:
            self.terminal_token = self.IGNORED

    def terminal_is_ignored(self):
        return self.terminal_token is self.IGNORED

    def terminal_exists(self):
        if self.terminal_token:
            return True
        else:
            return False

    def delete_terminal_token(self):
        """
        Delete the terminal token
        """
        self.terminal_token = None

    def get_terminal_token(self):
        """
        Return the terminal token and raise and exception if the node is not a terminal state
        """
        if self.terminal_exists():
            return None if self.terminal_is_ignored() else self.terminal_token
        else:
            raise NodeIsNotTerminalState


class LexerNFA(FiniteAutomata):

    # Counter for state id
    _ids = count(0)

    def __init__(self, *args, **kwargs):
        super(LexerNFA, self).__init__(*args, **kwargs)
        self.terminal_priority = kwargs['terminal_priority'] if 'terminal_priority' in kwargs else None

    def set_terminal_token(self, terminal_token, priority=None):
        """
        Set the terminal token if it is not already set to another value
        """
        if not self.terminal_exists():
            if terminal_token is None:
                self._set_terminal_to_ignored()
            else:
                self.terminal_token = terminal_token

            self.terminal_priority = priority

    def add_rule(self, regexp):
        """
        Add the given rule to the NFA.
        See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html
        :param regexp: A parsed regexp formated as a RegexpTree object
        :param token: the token returned by the rule (a string or a function FiniteAutomata -> string -> string)
        :return: a tuple (first, last) where first and last are respectively the first and last nodes of the rule
        """

        if regexp is not None:

            first = self.add_empty_transition()

            if regexp.type == 'single':
                min_ascii = regexp.min_ascii
                max_ascii = regexp.max_ascii

                next = first.add_transition_range(min_ascii, max_ascii)
                _, last = next.add_rule(regexp.next)

            elif regexp.type == 'union':
                fst_branch = first.add_rule(regexp.fst)
                snd_branch = first.add_rule(regexp.snd)

                next = fst_branch[1].add_empty_transition()
                snd_branch[1].add_empty_transition_to_state(next)

                _, last = next.add_rule(regexp.next)

            elif regexp.type == 'kleene':
                # The regexp A* leads to the following NFA
                #
                # self ---> first ---> s1 -A-> s2 ---> s3 (ACCEPT)
                #             |          ^------|       ^
                #             |-------------------------|
                #
                # See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html

                s1, s2 = first.add_rule(regexp.pattern)
                s3 = s2.add_empty_transition()
                # There should be a unique empty transition at this point

                s2.add_empty_transition_to_state(s1)
                first.add_empty_transition_to_state(s3)

                last = s3

                _, last = last.add_rule(regexp.next)

            else:
                raise RegexpTreeException("RegexpTree type found does not match 'single', 'union' or 'kleene'")

            return first, last

        else:
            return self, self

    def build(self, rules, max_handled_repeat=100):

        current_rule_priority = 1

        for rule, token in rules:
            formated_rule = format_regexp(rule, max_handled_repeat=max_handled_repeat)
            _, terminal_node = self.add_rule(formated_rule)
            terminal_node.set_terminal_token(token, priority=current_rule_priority)
            current_rule_priority += 1

    def get_transition_states_for_lookout(self, lookout):
        """
        Given a lookout (ascii value as int), return all the transition states attained by this lookout
        """
        states = []

        for transition in self.next_states:
            if value_is_in_range(lookout, transition[0]):
                states.append(transition[1])

        return states

    def get_transition_for_empty_string(self):
        """
        Return transition states corresponding to the empty string
        """
        states = []

        for transition in self.next_states:
            if transition[0] == self.EMPTY:
                states.append(transition[1])

        return states

    def get_transition_states_for_interval(self, interval):
        """
        Return transitions states which can be attained from all values in the interval
        """
        states = []

        for transition in self.next_states:
            if is_proper_subinterval(interval, transition[0]):
                states.append(transition[1])

        return states

    def get_epsilon_star_group(self, group=None):
        """
        Add all nodes linked by 0 or more epsilon (empty) transition from self, including self, to group, and return it.
        """
        if group is None:
            group = []

        for node in [self] + (self.get_transition_for_empty_string()):
            if node not in group:
                group.append(node)

                node.get_epsilon_star_group(group=group)

        return group


class LexerDFA(FiniteAutomata):

    # Counter for state id
    _ids = count(0)

    @staticmethod
    def build_from_nfa(nfa):
        """
        Generate the Deterministic Finite Automata corresponding to the given NFA
        """
        nodes_queue = [nfa]
        edges_lookouts = []
        nodes_as_dict = {}

        # Recover all nodes and edges from the NFA
        while nodes_queue:
            node = nodes_queue.pop()

            if node.id not in nodes_as_dict:
                nodes_as_dict[node.id] = node

                for lookout, child in node.next_states:

                    if lookout not in edges_lookouts:
                        edges_lookouts.append(lookout)

                    nodes_queue.append(child)

        # Build the DFA table
        # A good example of what this algorithm is doing can be watched here:
        # https://www.youtube.com/watch?v=taClnxU-nao
        alphabet = get_minimal_covering_intervals(edges_lookouts)

        epsilon_star_groups = {}
        for node_id, node in nodes_as_dict.items():
            epsilon_star_groups[node_id] = [child.id for child in node.get_epsilon_star_group()]

        dfa_nodes_table = {}

        dfa_nodes_queue = [tuple([nfa.id])]

        while dfa_nodes_queue:
            dfa_node = dfa_nodes_queue.pop()

            if dfa_node not in dfa_nodes_table:
                dfa_nodes_table[dfa_node] = {}

                for lookout in alphabet:

                    epsilon_star_states = set()

                    for nfa_node_id in dfa_node:

                        nfa_node = nodes_as_dict[nfa_node_id]
                        lookout_states = [child.id for child in nfa_node.get_transition_states_for_interval(lookout)]
                        for state in lookout_states:
                            epsilon_star_states |= set(epsilon_star_groups[state])

                    new_dfa_node = tuple(epsilon_star_states) if epsilon_star_states else None
                    dfa_nodes_table[dfa_node][lookout] = new_dfa_node

                    if new_dfa_node:
                        dfa_nodes_queue.append(new_dfa_node)

        # dfa_nodes_table now contains all the information required to build the DFA as a LexerDFA object
        # We first create the LexerDFA nodes to link later

        dfa_nodes_as_dict = {sub_id: LexerDFA() for sub_id in dfa_nodes_table}

        for sub_id, node in dfa_nodes_as_dict.items():

            # Set the terminal token
            possible_terminals = [nodes_as_dict[id] for id in sub_id]
            terminal_node = get_max_priority_terminal(possible_terminals)

            if terminal_node:
                node.set_terminal_token(terminal_node.get_terminal_token())

            # Set the transition states
            for lookout, target in dfa_nodes_table[sub_id].items():
                node.add_transition_to_state(lookout[0], lookout[1], target)

        return dfa_nodes_as_dict[tuple([nfa.id])]


    @staticmethod
    def build(rules):
        nfa = LexerNFA()
        nfa.build(rules)

        return LexerDFA.build_from_nfa(nfa)


def get_max_priority_terminal(nfa_nodes_list):
    """
    Given a list of nfa nodes, return the node that has the highest priority for its rule
    If no node is a terminal node, None is returned instead
    """
    highest_priority = -float('inf')
    highest_node = None

    for node in nfa_nodes_list:
        if node.terminal_exists() and node.terminal_priority > highest_priority:
            highest_node = node

    return highest_node


# ======================================================================================================================
# Set operations
# ======================================================================================================================

def value_is_in_range(value, range):
    """
    :param value: an int (x)
    :param range: a tuple of int (min, max)
    :return: True if x from min to max, False otherwise
    """
    return range[0] <= value <= range[1]


def is_proper_subinterval(subinterval, interval):
    """
    Return True if subinterval (x_1, x_2) is a subset of interval (y_1, y_2), else False
    """
    return value_is_in_range(subinterval[0], interval) and value_is_in_range(subinterval[1], interval)


def set_to_intervals(ascii_set):
    """
    Given a set of int, return a list of intervals covering those ints exactly
    ex: the set {1,2,3,5,6,9} would be returned as [(1,3), (5,6), (9,9)]
    """

    set_size = len(ascii_set)

    if set_size == 0:
        return []

    else:

        ascii_list = list(ascii_set)
        ascii_list.sort()

        # Hack so the last interval in the list is added
        ascii_list.append(float('inf'))

        interval_list = []

        min = max = ascii_list[0]

        index = 1
        while index <= set_size:
            ascii = ascii_list[index]
            if ascii == max + 1:
                max += 1
            else:
                interval_list.append((min, max))
                min = max = ascii

            index += 1

        return interval_list


def get_minimal_covering_intervals(intervals):
    """
    Given a list of intervals (min_int, max_int) which might overlap, return a minimal covering set of intervals
    The Minimal Covering Set of Interval (MCSI) is the set of intervals that has the following properties:

    1) The MCSI forms a disjoint partition of the union of the initial set
       i.e. they represent the same overall values, but the MSCI doesn't allow overlaps anymore

    2) For each interval A in MCSI and each interval B in the initial set of intervals, then either
       i) A is a proper subset of B (A and B = A)
       ii) or A and B are disjoint

    3) The MCSI is the smallest set such that the two above rules are respected
       It doesn't mean it is unique, solely that any other such set has the same cardinality

    Ex: Given the set {(1,5), (3, 7), (9,34), (15,15)}, we would return
        {(1,2), (3,5), (6,7), (8,14), (15, 15), (16, 34)}

    This is a way here to partition the intervals of ascii values in the lookouts of an FDA to get the lookout values
    of a DFA
    """

    # dictionary order comparator
    def interval_cmp(x, y):
        if x[0] > y[0]:
            return 1
        elif x[0] < y[0]:
            return -1
        elif x[1] > y[1]:
            return 1
        elif x[1] < y[1]:
            return -1
        else:
            return 0

    def rec(intervals):
        if not intervals:
            return []

        if len(intervals) == 1:
            return intervals

        intervals.sort(cmp=interval_cmp)

        left = intervals[0][0]
        right = intervals[0][1]

        for el in intervals:
            if el[0] == left:
                continue
            elif el[0] <= right:
                right = el[0] - 1
                break
            else:
                break

        # Remove intervals below (right + 1) and truncate others such that their minimum > max
        truncated_intervals = [(right + 1, el[1]) if el[0] <= right else el for el in intervals if el[1] > right]

        return [(left, right)] + rec(truncated_intervals)

    return rec(intervals)


# ======================================================================================================================
# Tokenize RegExp
# ======================================================================================================================

# In this section are the tools that take a regular expression as string and transform it to a bare-bone format.
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


class RegexpTree():
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
            self.next = values[2]

        elif node_type == 'kleene':
            self.pattern = values[0]
            self.next = values[1]

    def __str__(self):
        return "<RegexpTree '%s'>" % self.type

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

        return exp if self.next is None else (exp + self.next.print_regexp())


def format_regexp(regexp, max_handled_repeat=100):
    """
    Take a regexp as string and return the equivalent RegexpTree.
    Use sre_parse to first tokenize the regexp, then translate it.
    """

    parsed_regexp = sre_parse.parse(regexp)
    return sre_to_regexp_tree(parsed_regexp.data, max_handled_repeat=max_handled_repeat)


def sre_to_regexp_tree(sre_regexp, max_handled_repeat=100):
    """
    Take a regexp as sre_parse tokens list and return the equivalent RegexpTree.
    """

    # Token 'branch' has SubPatterns as sub-tokens, we convert back to list for uniformity
    if isinstance(sre_regexp, sre_parse.SubPattern):
        sre_regexp = sre_regexp.data

    sre_length = len(sre_regexp)

    if sre_length == 0:
        return None

    else:

        current_token = sre_regexp[0]

        # When sre_parse returned a SubPattern, extract the data
        if isinstance(current_token, sre_parse.SubPattern):
            current_token = current_token.data

        regexp_tail = sre_regexp[1:]
        token_type = current_token[0]

        if token_type == 'literal':
            return RegexpTree('single',
                              current_token[1],
                              current_token[1],
                              sre_to_regexp_tree(regexp_tail)
                              )

        # I don't think this is ever attained
        elif token_type == 'range':
            return RegexpTree('single',
                              current_token[1][0],
                              current_token[1][1],
                              sre_to_regexp_tree(regexp_tail)
                              )

        elif token_type == 'in':
            return make_regexp_intervals_union(current_token, regexp_tail, max_handled_repeat=max_handled_repeat)

        elif token_type == 'max_repeat':
            token_repeated = current_token[1][2]

            # In the case of 'max_repeat' tokens, sre_parse always returns current_token[1][2] as SubPattern, but
            # we don't always do so, thus we need to normalize here
            if isinstance(token_repeated, sre_parse.SubPattern):
                token_repeated = token_repeated.data

            min = current_token[1][0]
            max = current_token[1][1]

            if min > 0:
                new_max = max - min if max <= max_handled_repeat else max
                extension = min * token_repeated + [('max_repeat', (0, new_max, token_repeated))] + regexp_tail
                return sre_to_regexp_tree(
                    extension,
                    max_handled_repeat=max_handled_repeat
                )
            elif 1 <= max <= max_handled_repeat:
                branch_token = ('branch',
                                (None,
                                 ([('max_repeat', (0, max - 1, token_repeated))],
                                  token_repeated + [('max_repeat', (0, max - 1, token_repeated))
                                                    ])
                                 )
                                )
                return sre_to_regexp_tree(
                    [branch_token] + regexp_tail,
                    max_handled_repeat=max_handled_repeat
                )

            elif max == 0:
                return sre_to_regexp_tree(regexp_tail, max_handled_repeat=max_handled_repeat)

            # Case where min = 0, max = inf, i.e. a Kleene operator
            else:
                return RegexpTree(
                    'kleene',
                    sre_to_regexp_tree(token_repeated, max_handled_repeat=max_handled_repeat),
                    sre_to_regexp_tree(regexp_tail))

        elif token_type == 'branch':
            union_elements = current_token[1][1]
            return make_regexp_sre_union(union_elements, regexp_tail, max_handled_repeat=max_handled_repeat)

        elif token_type == 'subpattern':

            if isinstance(current_token[1][1], sre_parse.SubPattern):
                sub_regexp = current_token[1][1].data
            else:
                sub_regexp = current_token[1][1]

            subpattern = sre_to_regexp_tree(sub_regexp + regexp_tail)
            return subpattern

        # Corresponds to '.', that is anything but a linebreak (\n only)
        elif token_type == 'any':
            return RegexpTree('union',
                              RegexpTree('single', 0, 9),
                              RegexpTree('single', 11, 255),
                              sre_to_regexp_tree(regexp_tail)
                              )

        elif token_type == 'at':
            raise NotImplemented

        raise


def make_regexp_intervals_union(intervals, next, max_handled_repeat=100, already_in_intervals=False):
    """
    Given a list of sre 'in', 'range and 'literal' tokens, return a union of those as RegexpTree
    """
    if not already_in_intervals:
        intervals = sre_list_to_interval(intervals)

    list_length = len(intervals)

    if list_length == 0:
        return None

    # Will return a 'single' if we gave a single interval
    elif list_length == 1:
        min = intervals[0][0]
        max = intervals[0][1]
        return RegexpTree('single', min, max, sre_to_regexp_tree(next, max_handled_repeat=max_handled_repeat))

    else:
        fst = intervals[0]
        return RegexpTree(
            'union',
            RegexpTree('single', fst[0], fst[1], None),
            make_regexp_intervals_union(intervals[1:], [], max_handled_repeat=max_handled_repeat,
                                        already_in_intervals=True),
            sre_to_regexp_tree(next, max_handled_repeat=max_handled_repeat)
        )


def make_regexp_sre_union(regexp_union, next, max_handled_repeat=100):
    """
    Given a list of sre parsed regexp, return a union of those as a RegexpTree.
    """
    list_length = len(regexp_union)

    if list_length == 0:
        return None

    elif list_length == 1:
        regexp_tree = sre_to_regexp_tree(regexp_union[0], max_handled_repeat=max_handled_repeat)
        return regexp_tree

    else:
        fst_exp = regexp_union[0]
        fst_branch = sre_to_regexp_tree(fst_exp, max_handled_repeat=max_handled_repeat)
        snd_branch = make_regexp_sre_union(regexp_union[1:], [], max_handled_repeat=max_handled_repeat)
        next_branch = sre_to_regexp_tree(next, max_handled_repeat=max_handled_repeat)

        # It happens that both branches were empty expressions, we then collapse the tree
        if fst_branch is None and snd_branch is None:
            return next_branch
        else:
            return RegexpTree('union', fst_branch, snd_branch, next_branch)


def sre_list_to_interval(regexp_list):
    """
    Given a list of int, sre 'literal', sre 'in' or sre 'range' return a list of corresponding intervals of ascii
    values.
    """
    return set_to_intervals(sre_list_to_set(regexp_list))


def sre_list_to_set(regexp_list):
    """
    Given a list of int, (min, max), sre 'literal', sre 'in' or sre 'range' return a set of corresponding ascii values.
    """

    if not isinstance(regexp_list, list):
        regexp_list = [regexp_list]

    alphabet = set()

    for token in regexp_list:

        if isinstance(token, int):
            alphabet.add(token)

        elif isinstance(token[0], int) and isinstance(token[1], int):
            alphabet.add(set(range(token[0], token[1] + 1)))

        else:
            type = token[0]

            if type == "literal":
                value = token[1]
                alphabet.add(value)

            elif type == "in":
                alphabet |= sre_list_to_set(token[1])

            elif type == "range":
                value = token[1]
                alphabet |= set(range(value[0], value[1] + 1))

    return alphabet


def format_char_to_sre(char):
    return 'literal', ord(char)


def format_char_to_ascii(char):
    return ord(char)


def parse_regexp(regexp):
    """
    Tokenize the regexp (string) using Python sre_parse module.
    This returns a SubPattern object, we will mostly be intersted in the data field
    of the returned object.
    """
    pattern = sre_parse.parse(regexp, 0)
    return pattern
