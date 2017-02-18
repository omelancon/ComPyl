import sre_parse
import copy
from itertools import count
import re


class FiniteAutomatonError(Exception):
    pass


class NodeIsNotTerminalState(Exception):
    pass


class NodeFiniteAutomaton(object):
    """
    Basic skeleton for graph representation of Deterministic and Non-deterministic Finite Automata nodes.
    """
    # Special lookout values
    EMPTY = (-1, -1)

    # Special terminal values reserved for a terminal state that is ignored (no returned token)
    IGNORED = -1

    def __init__(self, terminal_token=None, counter=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # List of next states from current state.
        # Elements of the list are tuples (lookout, next state), lookout takes the range format (min_ascii, max_ascii)
        self.next_states = []

        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token

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
        Same as add_transition, but for an empty string match
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
        Set the terminal token if it is not already set to another value.
        To set the corresponding state to be an accepted, but ignored, state, let terminal_token be None
        """
        if not self.terminal_exists():
            if terminal_token is None:
                self._set_terminal_to_ignored()
            elif isinstance(terminal_token, str) or callable(terminal_token):
                self.terminal_token = terminal_token
            else:
                raise FiniteAutomatonError(
                    "The terminal token must be a string, a function, or None")

    def _set_terminal_to_ignored(self):
        """
        Set the terminal token value to ignored
        """
        if not self.terminal_token:
            self.terminal_token = self.IGNORED

    def terminal_is_ignored(self):
        return self.terminal_token == self.IGNORED

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


class NodeNFA(NodeFiniteAutomaton):
    def __init__(self, *args, **kwargs):
        super(NodeNFA, self).__init__(*args, **kwargs)

        self.terminal_priority = kwargs['terminal_priority'] if 'terminal_priority' in kwargs else None
        self.epsilon_star_group = None

    def set_terminal_token(self, terminal_token, priority=None):
        """
        Extend the 'set_terminal_token' method of NodeFiniteAutomaton to set a terminal token and additionally set
        the priority of the terminal (which is required for the NFA -> DFA step).
        """
        super(NodeNFA, self).set_terminal_token(terminal_token)

        if not self.terminal_exists():
            self.terminal_priority = priority

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

    def get_epsilon_star_group(self, force_recalculate=False):
        """
        Return the epsilon* group of the state, calculate it if not done yet or if force_recalculate is True
        """
        if self.epsilon_star_group is None or force_recalculate:
            self._calculate_epsilon_star_group(force_recalculate=force_recalculate)

        return self.epsilon_star_group

    def _calculate_epsilon_star_group(self, _group=None, force_recalculate=False):
        """
        Add all nodes linked by 0 or more epsilon (empty) transition from self, including self, to _group.
        Store the new _group is self.epsilon_star_group if we are at top level (_group is None)
        Then return the _group as set
        """
        # We will write the group to self only if save is True, that is for the node that called the initial calculation
        save = False

        if _group is None:
            _group = set()
            save = True

        for node in self.get_transition_for_empty_string():
            if force_recalculate or node.epsilon_star_group is None:
                node_group = node._calculate_epsilon_star_group(_group=_group, force_recalculate=force_recalculate)
            else:
                node_group = node.epsilon_star_group

            _group |= node_group

        _group.add(self)
        if save:
            self.epsilon_star_group = list(_group)

        return _group


class NodeDFA(NodeFiniteAutomaton):
    def __copy__(self):
        """
        Copy the NodeDFA node, linking it to the next states without copying those
        """
        dup = NodeDFA(terminal_token=copy.deepcopy(self.terminal_token))
        dup.id = self.id
        dup.next_states = {lookout: state for lookout, state in self.next_states}

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the NodeDFA node, recursively copying the next_states
        """
        if id(self) in memo:
            return memo[id(self)]

        else:
            dup = NodeDFA(terminal_token=copy.deepcopy(self.terminal_token))
            dup.id = self.id
            memo[id(self)] = dup
            dup.next_states = [(lookout, state.__deepcopy__(memo)) for lookout, state in self.next_states]

            return dup

    def transition(self, ascii):
        """
        Follow the lookout and return the next state, None if no state is attained from the lookout.
        Before transitions can be executed, the state has to have its 'next_states' list sorted with sort_lookouts()
        """
        value = binary_search_on_transitions(ascii, self.next_states)

        return value[1] if value else None

    def sort_lookouts(self):
        """
        Sort self.next_states by lookouts so they can be easily searched afterward
        """
        self.next_states.sort(cmp=lambda x, y: interval_cmp(x[0], y[0]))


class DFA:
    def __init__(self, rules=None):
        self.start = None
        self.current_state = None

        if rules:
            self.build(rules)

    def __copy__(self):
        """
        Identical as deepcopy as their is no point at returning a shallow copy
        """
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        """
        Return a copy of the DFA with a deepcopy of the graph
        """
        dup = DFA()

        dup.start = copy.deepcopy(self.start)
        dup.current_state = dup.get_dfa_state_by_id(self.current_state.id)

        return dup

    def build(self, rules):
        """
        Build the DFA according to the given rules, save its starting node as self.start and initialize its
        current_state to the start
        """
        formated_rules = [(format_regexp(rule), token) for rule, token in rules]

        nfa_start = self.build_nfa_from_rules(formated_rules)

        dfa_start = self.build_dfa_from_nfa(nfa_start)

        # The states are currently labelled with Python built-in id function, for aestheticism we give a nice ordering
        DFA.relabel_states_of_dfa(dfa_start)

        self.start = self.current_state = dfa_start

    def push(self, lookout):
        """
        Make the current_state transition with the given lookout, update it and return it. Return None if the lookout
        yields no legal transition.
        """
        ascii = ord(lookout)

        transition_state = self.current_state.transition(ascii)

        # Do not update if there was no legal transition
        if transition_state:
            self.current_state = transition_state

        return transition_state

    def reset_current_state(self):
        """
        Set back the current state to start
        """
        self.current_state = self.start

    def get_current_state_terminal(self):
        """
        Return the terminal of the current state. Return None if it is an ignored terminal.
        Will raise 'NodeIsNotTerminalState' if the node is not a terminal.
        """
        return self.current_state.get_terminal_token()

    def get_dfa_state_by_id(self, id):
        """
        Return the first state found with given 'id' (should be unique), None if no such state exists
        This is used when copying to recover the copied current_state object
        """
        seen_states = set()
        todo_states = [self.start]

        while todo_states:
            state = todo_states.pop()

            if state.id == id:
                return state

            else:
                seen_states.add(state)

            for _, child_state in state.next_states:
                if child_state not in seen_states:
                    todo_states.append(child_state)

        return None

    @staticmethod
    def relabel_states_of_dfa(start, count_from=0):
        """
        Relabel the states' ids of the DFA, strating at 'start'.
        """

        def rec_relabel(state, counter, relabeled):
            state.id = counter.next()

            for _, child_state in state.next_states:
                if child_state not in relabeled:
                    relabeled.add(child_state)
                    rec_relabel(child_state, counter, relabeled)

        counter = count(count_from)

        rec_relabel(start, counter, set())

    @staticmethod
    def build_nfa_from_rules(rules):
        """
        Parse and add the rules one by one to an empty NFA, see add_rule_to_nfa for the rule-adding algorithm.
        """
        id_counter = count(0)

        nfa_start = NodeNFA()

        # The rule priority has to be written in the NFA, later when the nodes are merged to form a DFA, it resolved
        # conflicts when a string attains more than one terminal node in the NFA
        current_rule_priority = 1

        for rule, token in rules:
            _, terminal_node = DFA.add_rule_to_nfa(nfa_start, rule)
            terminal_node.set_terminal_token(token, priority=current_rule_priority)
            current_rule_priority += 1

        return nfa_start

    @staticmethod
    def add_rule_to_nfa(nfa_start, regexp):
        """
        Add the given rule to the NFA.
        See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html
        :param regexp: A parsed regexp formated as a RegexpTree object
        :param token: the token returned by the rule (a string or a function NodeFiniteAutomaton -> string -> string)
        :return: a tuple (first, last) where first and last are respectively the first and last nodes of the rule
        """

        if regexp is not None:

            first = nfa_start.add_empty_transition()

            if regexp.type == 'single':
                min_ascii = regexp.min_ascii
                max_ascii = regexp.max_ascii

                next = first.add_transition_range(min_ascii, max_ascii)
                _, last = DFA.add_rule_to_nfa(next, regexp.next)

            elif regexp.type == 'union':
                fst_branch = DFA.add_rule_to_nfa(first, regexp.fst)
                snd_branch = DFA.add_rule_to_nfa(first, regexp.snd)

                next = fst_branch[1].add_empty_transition()
                snd_branch[1].add_empty_transition_to_state(next)

                _, last = DFA.add_rule_to_nfa(next, regexp.next)

            elif regexp.type == 'kleene':
                # The regexp A* leads to the following NFA
                #
                # self ---> first ---> s1 -A-> s2 ---> s3 (ACCEPT)
                #             |          ^------|       ^
                #             |-------------------------|
                #
                # See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html

                s1, s2 = DFA.add_rule_to_nfa(first, regexp.pattern)
                s3 = s2.add_empty_transition()
                # There should be a unique empty transition at this point

                s2.add_empty_transition_to_state(s1)
                first.add_empty_transition_to_state(s3)

                last = s3

                _, last = DFA.add_rule_to_nfa(last, regexp.next)

            else:
                raise RegexpTreeException("RegexpTree type found does not match 'single', 'union' or 'kleene'")

            return first, last

        else:
            return nfa_start, nfa_start

    @staticmethod
    def build_dfa_from_nfa(nfa):
        """
        Generate the Deterministic Finite Automaton corresponding to the given NFA following these steps:
        1) Recover all nodes from the NFA as well as the alphabet used by the language
        2) Recover the epsilon star groups and merge them to generate the nodes of the DFA, this returns a table
           representation fot he DFA
        3) For each equivalence class (epsilon star group), recover the terminal node with maximum priority
        4) Minimize the DFA with Hopcroft's algorithm
        5) Optimize the lookouts by merging adjacent intervals
        6) Translate the table to a graph structure
        7) Return the starting node
        """
        # ========================================================
        # Recover all nodes and possible lookouts found in the NFA
        # ========================================================
        # This step is required to recover the lookouts which will allow to determine the alphabet of the language
        # The construction of an id -> node dictionary is not absolutely necessary, but we build it as a helper while
        # we are at it.
        nodes_as_dict, edges_lookouts = recover_nodes_and_lookouts_from_nfa(nfa)

        # ========================================================
        # Recover the alphabet of the language
        # ========================================================
        # Since our edges are intervals, we use a Minimal Covering Set of Intervals as our alphabet, see the doc string
        # of function get_minimal_covering_intervals for definition of a MCSI
        # EMPTY is not in the alphabet (epsilon is an element of A*, not in A, where A is the alphabet), so we remove it
        alphabet = get_minimal_covering_intervals(edges_lookouts)
        alphabet.remove(NodeNFA.EMPTY)

        # ========================================================
        # Build the DFA table
        # ========================================================
        # A good example of what this algorithm is doing can be watched here:
        # https://www.youtube.com/watch?v=taClnxU-nao
        # We also store the parents from a given lookout for each node, this will be used in the minimisation step
        dfa_nodes_table = {}

        initial_epsilon_group = node_list_to_sorted_tuple_of_id(nfa.get_epsilon_star_group())
        dfa_nodes_queue = [initial_epsilon_group]
        seen_nodes = set()

        # We have to add the error node because Hopcroft algorithm that will later be used to minimize the DFA
        # requires a complete DFA. We first add the error node in the table
        error_node_id = tuple()
        dfa_nodes_table[error_node_id] = {'is_terminal': False, 'terminal': None, 'transitions': {},
                                          'parents': {lookout: {error_node_id} for lookout in alphabet}}

        while dfa_nodes_queue:
            dfa_node = dfa_nodes_queue.pop()

            if dfa_node not in seen_nodes:

                if dfa_node not in dfa_nodes_table:
                    dfa_nodes_table[dfa_node] = {'is_terminal': False, 'terminal': None, 'transitions': {},
                                                 'parents': {lookout: set() for lookout in alphabet}}

                for lookout in alphabet:

                    epsilon_star_states = set()

                    for nfa_node_id in dfa_node:

                        nfa_node = nodes_as_dict[nfa_node_id]

                        for state in nfa_node.get_transition_states_for_interval(lookout):
                            epsilon_star_group = state.get_epsilon_star_group()
                            epsilon_star_states |= node_list_to_set_of_id(epsilon_star_group)

                    new_dfa_node = tuple(epsilon_star_states) if epsilon_star_states else error_node_id
                    dfa_nodes_table[dfa_node]['transitions'][lookout] = new_dfa_node

                    if new_dfa_node not in dfa_nodes_table:
                        dfa_nodes_table[new_dfa_node] = {'is_terminal': False, 'terminal': None, 'transitions': {},
                                                         'parents': {lookout: set() for lookout in alphabet}}

                    dfa_nodes_table[new_dfa_node]['parents'][lookout].add(dfa_node)

                    if new_dfa_node:
                        dfa_nodes_queue.append(new_dfa_node)

                seen_nodes.add(dfa_node)

        # ========================================================
        # Mark the terminal nodes
        # ========================================================
        # A NFA can reach multiple terminal states at once, in the case of a DFA we recover all those terminal
        # states and take the one with the highest priority, that is the one which rule was given first when
        # building the NFA
        for sub_id in dfa_nodes_table:
            possible_terminals = [nodes_as_dict[id] for id in sub_id]
            terminal_node = get_max_priority_terminal(possible_terminals)

            # Store the terminal
            # The use of the boolean is because None means the state is terminal but ignored, we cannot simply use
            # 'terminal' entry to be None to indicate that the node is not a final state
            if terminal_node:
                dfa_nodes_table[sub_id]['is_terminal'] = True
                dfa_nodes_table[sub_id]['terminal'] = terminal_node.get_terminal_token()

        # ========================================================
        # Minimize the DFA
        # ========================================================
        # dfa_node_table represents a DFA, but might not be minimal
        # We get the minimal DFA using Hopcroft's algorithm

        minimum_dfa = hopcrofts_algorithm(dfa_nodes_table, alphabet)

        # ========================================================
        # Merge adjacent lookouts
        # ========================================================
        # The generation of the alphabet with get_minimal_covering_intervals made partitioning of some lookouts too fine
        # we will merge such lookouts. By example if from State x, the intervals (97,98) and (99, 102) lead to State y,
        # we merge the lookouts so that (97, 102) leads to State y.

        for id, state in minimum_dfa.items():
            inverse_map = {}

            for lookout, target in state['transitions'].items():
                if target in inverse_map:
                    inverse_map[target].append(lookout)
                else:
                    inverse_map[target] = [lookout]

            new_transitions = {}

            for target, lookouts in inverse_map.items():
                new_lookouts = merge_intervals(lookouts)

                for lookout in new_lookouts:
                    new_transitions[lookout] = target

            state['transitions'] = new_transitions

        # ========================================================
        # Build the final data structure of the DFA
        # ========================================================
        # dfa_nodes_table now contains all the information required to build the minimal DFA as a NodeDFA object
        # We first create the NodeDFA nodes to link afterward

        dfa_nodes_as_dict = {sub_id: NodeDFA() for sub_id in minimum_dfa}

        for sub_id, node in dfa_nodes_as_dict.items():

            # Set the terminal token
            is_terminal = minimum_dfa[sub_id]['is_terminal']
            if is_terminal:
                token = minimum_dfa[sub_id]['terminal']
                node.set_terminal_token(token)

            # Set the transition states
            for lookout, target in minimum_dfa[sub_id]['transitions'].items():
                if target:
                    node.add_transition_to_state(lookout[0], lookout[1], dfa_nodes_as_dict[target])

            # We sort the lookouts for easy recovery
            node.sort_lookouts()

        # Finally we recover the starting node to return it
        initial_node_id = None
        for fset in dfa_nodes_as_dict:
            if initial_epsilon_group in fset:
                initial_node_id = fset
                break
        else:
            raise FiniteAutomatonError("For unknown reason, the build algorithm lost its own initial state. Puzzling.")

        return dfa_nodes_as_dict[initial_node_id]


def hopcrofts_algorithm(dfa_nodes_table, alphabet, error_state_id=tuple()):
    """
    Hopcroft's algorithm for minimalisation of a DFA
    :param dfa_nodes_table: A dictionary id -> node, each node is itself a dictionary with the keys 'is_terminal',
    'terminal', 'transitions', 'parents'. 'is_terminal' is a boolean stating if the state is an accepting state,
    'terminal' stores the token returned by the accepting state, 'transitions' is a dict lookout -> node_id, 'parents'
    is similar to lookout, entries point to sets of nodes id from which the current node is attainable given a certain
    lookout. Hopcroft's algorithm requires that the given DFA is complete, that is a state always leads to another state
    given a lookout, but this might be the error state.
    :param alphabet: The alphabet of the language of the DFA as a list of all possible lookouts
    :return: A dict of similar structure as the input, but representing the minimal DFA. Although the minimal DFA
    returned is not complete, the error state is not present, but implied by the absence of transition.
    WARNING: The process of reindexing the nodes leads them to have frozen sets of tuples as index (keys in the returned
    dict), this might change in the future, but is perfectly fine for now since these are hashable and meaningful.
    """
    # Partition the different terminal states since we know that if they have different terminal tokens, then they are
    # distinguishable
    terminal_states_as_dict = {}

    for id in dfa_nodes_table:
        state = dfa_nodes_table[id]

        if state['is_terminal']:
            terminal = state['terminal']

            if terminal in terminal_states_as_dict:
                terminal_states_as_dict[terminal].add(id)

            else:
                terminal_states_as_dict[terminal] = {id}

    terminal_states = {frozenset(states) for _, states in terminal_states_as_dict.items()}
    non_terminal_states = frozenset([id for id in dfa_nodes_table if not dfa_nodes_table[id]['is_terminal']])

    partition = {non_terminal_states}
    partition |= terminal_states

    # Refine the sets
    sets_to_refine = terminal_states

    while sets_to_refine:
        analyzed_set = sets_to_refine.pop()

        for lookout in alphabet:
            lookout_parents = set()

            for node in analyzed_set:
                node_lookout_parents = dfa_nodes_table[node]['parents'][lookout]
                lookout_parents |= node_lookout_parents

            # Python does not allow change to a set while iterating through it, thus we have to update it afterward
            remove_from_parition = set()
            add_to_parition = set()
            for element in partition:
                intersection = element & lookout_parents
                difference = element - lookout_parents

                if intersection and difference:
                    remove_from_parition.add(element)
                    add_to_parition.add(intersection)
                    add_to_parition.add(difference)

                    if element in sets_to_refine:
                        sets_to_refine.remove(element)
                        sets_to_refine.add(intersection)
                        sets_to_refine.add(difference)

                    else:
                        if len(intersection) <= len(difference):
                            sets_to_refine.add(intersection)
                        else:
                            sets_to_refine.add(difference)

            partition |= add_to_parition
            partition -= remove_from_parition

    # Every set of set in the partition can now be merged to a single state
    # We first build a mapping (dict) from the old nodes ids to the new nodes ids

    mapping = {}

    for states in partition:
        for state in states:
            mapping[state] = states

    # Remove the error node, represented by an empty tuple
    partition.remove(frozenset([error_state_id]))

    minimal_dfa = {}
    for states in partition:
        transitions = {}

        for state in states:
            for lookout, target in dfa_nodes_table[state]['transitions'].items():
                if target == error_state_id or lookout in transitions:
                    continue
                else:
                    transitions[lookout] = mapping[target]

        # Hack to recover an item from a set
        for any in states:
            break

        minimal_dfa[states] = {'is_terminal': dfa_nodes_table[any]['is_terminal'],
                               'terminal': dfa_nodes_table[any]['terminal'],
                               'transitions': transitions}

    return minimal_dfa


def recover_nodes_and_lookouts_from_nfa(nfa):
    """
    Given a Non-Deterministic Finite Automata, return a dict of all nodes keyed by id and a list of all lookouts value
     found in the NFA.
    """
    edges_lookouts = []
    nodes_as_dict = {}
    nodes_queue = [nfa]

    while nodes_queue:
        node = nodes_queue.pop()

        if node.id not in nodes_as_dict:
            nodes_as_dict[node.id] = node

            for lookout, child in node.next_states:

                if lookout not in edges_lookouts:
                    edges_lookouts.append(lookout)

                nodes_queue.append(child)

    return nodes_as_dict, edges_lookouts


def get_max_priority_terminal(nfa_nodes_list):
    """
    Given a list of nfa nodes, return the node that has the highest priority for its rule. A node with lowest
    terminal_priority will have priority for its rule.
    If no node is a terminal node, None is returned instead
    """
    best_priority = float('inf')
    best_node = None

    for node in nfa_nodes_list:
        if node.terminal_exists() and node.terminal_priority < best_priority:
            best_node = node
            best_priority = best_node.terminal_priority

    return best_node


def node_list_to_sorted_tuple_of_id(node_list):
    return tuple(sorted([node.id for node in node_list]))


def node_list_to_set_of_id(node_list):
    return set([node.id for node in node_list])


# ======================================================================================================================
# Set operations
# ======================================================================================================================

def interval_cmp(x, y):
    """
    Dictionary order comparator for intervals
    """
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


def binary_search_on_transitions(target, transitions):
    """
    Binary search for intervals in a sorted list using interval_cmp for comparisons
    """
    left = 0
    right = len(transitions) - 1

    while left <= right:
        index = (left + right) / 2
        min, max = transitions[index][0]

        if target < min:
            right = index - 1
        elif target > max:
            left = index + 1
        else:
            return transitions[index]

    return None


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


def merge_intervals(intervals):
    """
    Given a list of intervals, return a new list of intervals where the adjacent intervals are merged.
    Ex: [(1,3), (4,6), (10,11)] would be returned as [(1, 6), (10, 11)]
    """
    if not intervals:
        return []

    intervals.sort(cmp=interval_cmp)

    merged_intervals = []
    min = max = intervals[0][0]

    for interval in intervals:
        if interval[0] > max + 1:
            merged_intervals.append((min, max))
            min = interval[0]

        max = interval[1]

    merged_intervals.append((min, max))

    return merged_intervals


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


class RegexpParsingException(Exception):
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
    nodes_list = regexp_to_regexp_tree_list(regexp)

    regexp_tree = nodes_list.pop(0) if nodes_list else None
    last = regexp_tree

    while nodes_list:
        last.extend(nodes_list.pop(0))
        last = last.next

    return regexp_tree


def regexp_to_regexp_tree_list(regexp,  pos=0):
    """
    Parse a regular expression and return it as a RegexpTree object
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
        char = ord(regexp[pos + 1])
        node = RegexpTree('single', char, char)
        pos += 2
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
        node = RegexpTree('kleene', copy.deepcopy(nodes_list[-1]))
        pos += 1
        nodes = [node]

    elif regexp[pos] == "{":
        end_pos = regexp.find("}", beg=pos+1)
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


def get_regexptree_union_from_set(inner_set):
    """
    Given the inner part of a set ([...]) in a regexp, return the corresponding RegexpTree object
    """
    if inner_set:
        length = len(inner_set)
        pos = 0
        intervals = []

        while pos < length:
            if inner_set[pos] == "\\":
                ascii = ord(inner_set[pos + 1])
                intervals.append((ascii, ascii))
                pos += 2

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

                intervals.append((min, max))

            else:
                ascii = ord(inner_set[pos])
                intervals.append((ascii, ascii))
                pos += 1

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

    else:
        last = copy.deepcopy(node)
        first = RegexpTree(
            'union',
            None,
            last
        )
        max -= 1

    while min > 0:
        extension = copy.deepcopy(node)
        last.extend(extension)
        last = extension
        min - 1
        max -= 1

    while max > 0:
        extension_snd = copy.deepcopy(node)
        extension = RegexpTree(
            'union',
            None,
            extension_snd
        )
        last.extend(extension)
        last = extension_snd

        max -= 1

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


def format_char_to_ascii(char):
    return ord(char)
