from itertools import count
import copy
from functools import cmp_to_key

import compyl.__lexer.regexp as RegExp
import compyl.__lexer.interval_operations as IntervalOp
from compyl.__lexer.errors import LexerBuildError


# ======================================================================================================================
# Finite Automatons Classes
# ======================================================================================================================


class NodeIsNotTerminalState(LexerBuildError):
    pass


class NodeFiniteAutomaton(object):
    """
    Basic skeleton for graph representation of Deterministic and Non-deterministic Finite Automata nodes.
    """
    # Special lookout values
    EMPTY = (-1, -1)

    # Special terminal values reserved for a terminal state that is accepted but ignored (no returned token)
    IGNORED = -1

    def __init__(self, terminal_token=None, special_actions=None, counter=None):
        # A counter can be provided to give ordered unique ids for the states, otherwise we generate them
        self.id = counter.next() if counter else id(self)

        # List of next states from current state.
        # Elements of the list are tuples (lookout, next state), lookout takes the range format (min_ascii, max_ascii)
        self.next_states = []

        # Terminal token is intended to be either a string or a function returning a string
        self.terminal_token = terminal_token

        # Special action is a function returning None, but acting on the state of the lexer
        # It is returned to be called by the lexer when the DFA encounters a pattern contained into a main pattern
        self.special_actions = special_actions if special_actions else []

    def __str__(self):
        return "<State '%d'>" % self.id

    def add_transition_range(self, min_ascii, max_ascii, *args, **kwargs):
        """
        Add the edge corresponding to the transition when a character from min_ascii to max_ascii is seen.
        For a single ascii transition, let min_ascii = max_ascii, ex: 'a' would be (97, 97)
        Extra arguments and keyword arguments are passed to the object generator.

        REMARK: in the case of a DFA, be aware that add_transition does not check for already existing transitions.
        Using it incorrectly can transform a DFA back into a NFA
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
                self.__set_terminal_to_ignored()
            elif isinstance(terminal_token, str) or callable(terminal_token):
                self.terminal_token = terminal_token
            else:
                raise LexerBuildError(
                    "The terminal token must be a string, a function, or None")

    def __set_terminal_to_ignored(self):
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

    def add_special_action(self, action_type, action, priority):
        """
        Add the given action to the special actions list
        """
        self.special_actions.append((action_type, action, priority))

    def get_special_actions(self):
        """
        Return the list of special actions of the node
        """
        return self.special_actions

    def set_special_actions(self, actions):
        """
        Replace the current special actions of the node by the new list of special actions
        """
        self.special_actions = []

        for action in actions:
            self.add_special_action(*action)

    def has_special_action(self):
        """
        Returns True if the state has special actions, False otherwise
        """
        return bool(self.special_actions)

    def has_special_action_of_type(self, type):
        """
        Return True if the state has a special action of given type, else False
        """

        for action in self.get_special_actions():
            if action[0] == type:
                return True

        return False


class NodeNFA(NodeFiniteAutomaton):
    def __init__(self, *args, **kwargs):
        super(NodeNFA, self).__init__(*args, **kwargs)

        # Priority of the terminal token of the current node, allowing to choose the right rule when merging NFA nodes
        self.terminal_priority = kwargs['terminal_priority'] if 'terminal_priority' in kwargs else None

        # An epsilon star group is the set of all nodes connected to the current node by one or more empty transition
        self.epsilon_star_group = None

        # Any state with is_real_state set to False cannot lead to the creation of a DFA node by itself
        self.is_real_state = kwargs['is_real_state'] if 'is_real_state' in kwargs else True

    def set_terminal_token(self, terminal_token, priority=None):
        """
        Extend the 'set_terminal_token' method of NodeFiniteAutomaton to set a terminal token and additionally set
        the priority of the terminal (which is required for the NFA -> DFA step).
        """
        if not self.terminal_exists():
            self.terminal_priority = priority

        super(NodeNFA, self).set_terminal_token(terminal_token)

    def make_real_state(self):
        self.is_real_state = True

    def make_fake_state(self):
        self.is_real_state = False

    def get_transition_states_for_lookout(self, lookout):
        """
        Given a lookout (ascii value as int), return all the transition states attained by this lookout
        """
        states = []

        for transition in self.next_states:
            if IntervalOp.value_is_in_range(lookout, transition[0]):
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
            if IntervalOp.is_proper_subinterval(interval, transition[0]):
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
        Store the new _group in self.epsilon_star_group if we are at top level (_group is None)
        Then return the _group as set
        """
        # We will write the group to self only if save is True, that is for the node that called the initial calculation
        save = False

        if _group is None:
            _group = set()
            save = True

        _group.add(self)

        for node in self.get_transition_for_empty_string():
            if node not in _group:
                if force_recalculate or node.epsilon_star_group is None:
                    node_group = node._calculate_epsilon_star_group(_group=_group, force_recalculate=force_recalculate)
                else:
                    node_group = node.epsilon_star_group

                _group |= node_group

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
        dup.special_actions = copy.copy(self.special_actions)
        dup.next_states = {lookout: state for lookout, state in self.next_states}

        return dup

    def __deepcopy__(self, memo):
        """
        Copy the NodeDFA node, recursively copying the next_states
        """

        dup = NodeDFA(terminal_token=copy.deepcopy(self.terminal_token))
        dup.id = self.id
        memo[id(self)] = dup
        dup.special_actions = copy.copy(self.special_actions)
        dup.next_states = [(lookout, copy.deepcopy(state, memo)) for lookout, state in self.next_states]

        return dup

    def transition(self, ascii):
        """
        Follow the lookout and return the next state, None if no state is attained from the lookout.
        Before transitions can be executed, the state has to have its 'next_states' list sorted with sort_lookouts()
        """
        value = IntervalOp.binary_search_on_transitions(ascii, self.next_states)

        return value[1] if value else None

    def sort_lookouts(self):
        """
        Sort self.next_states by lookouts so they can be easily searched afterward
        """
        self.next_states.sort(key=cmp_to_key(lambda x, y: IntervalOp.interval_cmp(x[0], y[0])))

    def add_special_action(self, action_type, action):
        """
        Add the given action to the special actions list
        """
        self.special_actions.append((action_type, action))


class DFA:

    # Ids for the special actions as we do not want to store them as strings in the DFA
    TRIGGER_ON_CONTAIN = 1
    NON_GREEDY = 2

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
        formated_rules = []

        for packed_rule in rules:
            rule = RegExp.format_regexp(packed_rule[0])

            if rule is None or rule.length()[0] == 0:
                raise LexerBuildError("error with rule '%s', regexp minimum length cannot be 0" % packed_rule[0])

            token = packed_rule[1]
            try:
                if packed_rule[2] == "trigger_on_contain":
                    if not callable(token):
                        raise LexerBuildError("token of special action trigger_on_contain must be a function")

                    special_action = self.TRIGGER_ON_CONTAIN

                elif packed_rule[2] == "non_greedy":
                    if not (callable(token) or isinstance(token, str) or token is None):
                        raise LexerBuildError(
                            "token of special action non_greedy must be a function, a string or None")

                    special_action = self.NON_GREEDY

                elif packed_rule[2] is None:
                    special_action = None

                else:
                    raise LexerBuildError("special action of rule (third parameter) is unrecognized")

            except IndexError:
                special_action = None

            formated_rules.append(
                (rule, token, special_action)
            )

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
        Relabel the states' ids of the DFA, starting at 'start'.
        """

        def rec_relabel(state, counter, relabeled):
            state.id = next(counter)

            for _, child_state in state.next_states:
                if child_state not in relabeled:
                    relabeled.add(child_state)
                    rec_relabel(child_state, counter, relabeled)

        counter = count(start=count_from, step=1)

        rec_relabel(start, counter, set())

    @staticmethod
    def build_nfa_from_rules(rules):
        """
        Parse and add the rules one by one to an empty NFA, see add_rule_to_nfa for the rule-adding algorithm.
        """
        nfa_start = NodeNFA()

        # The rule priority has to be written in the NFA, later when the nodes are merged to form a DFA, it resolved
        # conflicts when a string attains more than one terminal node in the NFA
        current_rule_priority = 1

        # Special actions such as trigger_on_contain need that all states be linked to the state marking the start
        # of the action's accepted pattern with an epsilon transition, thus we collect all those states and connect
        # them once the whole NFA has been generated.
        totally_connected_states = []

        for rule, token, special_action in rules:

            if not special_action:
                _, terminal_node = DFA.add_rule_to_nfa(nfa_start, rule)
                terminal_node.set_terminal_token(token, priority=current_rule_priority)

            elif special_action == DFA.TRIGGER_ON_CONTAIN:
                action_start, action_node = DFA.add_rule_to_nfa(nfa_start, rule, is_real_state=False)
                action_node.add_special_action(special_action, token, priority=current_rule_priority)

                totally_connected_states.append(action_start)

            elif special_action == DFA.NON_GREEDY:
                action_start, action_node = DFA.add_rule_to_nfa(nfa_start, rule, is_real_state=True)
                action_node.add_special_action(special_action, token, priority=current_rule_priority)

            current_rule_priority += 1

        states_set = recover_nodes_set_from_nfa(nfa_start)

        for target in totally_connected_states:
            for state in states_set:
                if state is not target:
                    state.add_empty_transition_to_state(target)

        return nfa_start

    @staticmethod
    def add_rule_to_nfa(nfa_start, regexp, is_real_state=True):
        """
        Add the given rule to the NFA.
        See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html
        :param regexp: A parsed regexp formated as a RegexpTree object
        :param token: the token returned by the rule (a string or a function NodeFiniteAutomaton -> string -> string)
        :return: a tuple (first, last) where first and last are respectively the first and last nodes of the rule
        """
        if regexp is not None:

            first = nfa_start.add_empty_transition()

            if not is_real_state:
                first.make_fake_state()

            if regexp.type == 'single':
                min_ascii = regexp.min_ascii
                max_ascii = regexp.max_ascii

                next = first.add_transition_range(min_ascii, max_ascii)
                _, last = DFA.add_rule_to_nfa(next, regexp.next, is_real_state=is_real_state)

                if not is_real_state:
                    next.make_fake_state()

            elif regexp.type == 'union':
                fst_branch = DFA.add_rule_to_nfa(first, regexp.fst, is_real_state=is_real_state)
                snd_branch = DFA.add_rule_to_nfa(first, regexp.snd, is_real_state=is_real_state)

                next = fst_branch[1].add_empty_transition()
                snd_branch[1].add_empty_transition_to_state(next)

                _, last = DFA.add_rule_to_nfa(next, regexp.next, is_real_state=is_real_state)

                if not is_real_state:
                    next.make_fake_state()

            elif regexp.type == 'kleene':
                # The regexp A* leads to the following NFA
                #
                # self ---> first ---> s1 -A-> s2 ---> s3 (ACCEPT)
                #             |          ^------|       ^
                #             |-------------------------|
                #
                # See http://www.cs.may.ie/staff/jpower/Courses/Previous/parsing/node5.html

                s1, s2 = DFA.add_rule_to_nfa(first, regexp.pattern, is_real_state=is_real_state)
                s3 = s2.add_empty_transition()
                # There should be a unique empty transition at this point

                s2.add_empty_transition_to_state(s1)
                first.add_empty_transition_to_state(s3)

                last = s3

                _, last = DFA.add_rule_to_nfa(last, regexp.next, is_real_state=is_real_state)

                if not is_real_state:
                    last.make_fake_state()

            else:
                raise RegExp.RegexpTreeException("RegexpTree type found does not match 'single', 'union' or 'kleene'")

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
        alphabet = IntervalOp.get_minimal_covering_intervals(edges_lookouts)
        alphabet.remove(NodeNFA.EMPTY)

        # ========================================================
        # Build the DFA table
        # ========================================================
        # A good example of what this algorithm is doing can be watched here:
        # https://www.youtube.com/watch?v=taClnxU-nao
        # We also store the parents from a given lookout for each node, this will be used in the minimisation step
        #
        # Additionally, we have to check that given a DFA node (equivalence class of NFA nodes), it is not only formed
        # of fake nodes, i.e. nodes that have 'is_real_node' set to False. These nodes should be rejected as their
        # only purpose is to change the behavior of the wanted DFA, not change it. Thus they are ignored if they do
        # not intersect and equivalence class of real nodes (is_real_node = True)

        dfa_nodes_table = {}

        initial_epsilon_group = node_list_to_sorted_tuple_of_id(nfa.get_epsilon_star_group())
        dfa_nodes_queue = [initial_epsilon_group]
        seen_nodes = set()

        # We have to add the error node because Hopcroft algorithm that will later be used to minimize the DFA
        # requires a complete DFA. We first add the error node in the table
        error_node_id = tuple()
        dfa_nodes_table[error_node_id] = {'is_terminal': False,
                                          'terminal': None,
                                          'transitions': {},
                                          'special_actions': [],
                                          'parents': {lookout: {error_node_id} for lookout in alphabet}
                                          }

        while dfa_nodes_queue:
            dfa_node = dfa_nodes_queue.pop()

            if dfa_node not in seen_nodes:

                is_real_state = any([nodes_as_dict[id].is_real_state for id in dfa_node])

                if not is_real_state:
                    continue

                if dfa_node not in dfa_nodes_table:
                    dfa_nodes_table[dfa_node] = {'is_terminal': False,
                                                 'terminal': None,
                                                 'transitions': {},
                                                 'special_actions': [],
                                                 'parents': {lookout: set() for lookout in alphabet}
                                                 }

                # A node with a non_greedy special action will immediately return its token and never transition
                # Thus we interrupt the formation of transitions if any such special action is found
                # Setting the non greedy returned token is done in the next step 'Mark the terminal nodes'
                is_non_greedy = any(nodes_as_dict[id].has_special_action_of_type(DFA.NON_GREEDY) for id in dfa_node)

                for lookout in alphabet:

                    # If the rule is non-greedy, we want to interrupt, thus any thing raises an error
                    if is_non_greedy:
                        dfa_nodes_table[dfa_node]['transitions'][lookout] = error_node_id

                    else:

                        epsilon_star_states = set()

                        for nfa_node_id in dfa_node:

                            nfa_node = nodes_as_dict[nfa_node_id]

                            for state in nfa_node.get_transition_states_for_interval(lookout):
                                epsilon_star_group = state.get_epsilon_star_group()
                                epsilon_star_states |= node_list_to_set_of_id(epsilon_star_group)

                        if epsilon_star_states:
                            new_dfa_node = tuple(sorted(epsilon_star_states))
                        else:
                            new_dfa_node = error_node_id

                        new_dfa_node_is_real_state = any([nodes_as_dict[id].is_real_state for id in new_dfa_node])

                        if new_dfa_node_is_real_state or new_dfa_node is error_node_id:

                            dfa_nodes_table[dfa_node]['transitions'][lookout] = new_dfa_node

                            if new_dfa_node not in dfa_nodes_table:
                                # This is a placeholder for now, it will be filled later
                                dfa_nodes_table[new_dfa_node] = {'is_terminal': False,
                                                                 'terminal': None,
                                                                 'transitions': {},
                                                                 'special_actions': [],
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
        # In this state we also recover the special actions of states. Unlike terminal instructions, a state can have
        # multiple special actions, so we keep them all
        for sub_id in dfa_nodes_table:

            # Recover the special actions
            # Multiple special actions can be triggered on a same node
            special_actions = set()

            for id in sub_id:
                node_special_actions = nodes_as_dict[id].special_actions
                for action in node_special_actions:
                    special_actions.add(action)

            special_actions = list(special_actions)

            # Sort by priority
            special_actions = sort_special_actions_list(special_actions)

            # Remove anything after a non_greedy special action, as it is shadowed
            special_actions = truncate_special_action_list_at_non_greedy(special_actions)

            # We do not store the priority once the list is sorted
            # The NodeDFA list of special actions does not take a priority, it assumes the actions are passed in order
            special_actions = remove_priority_from_special_actions_list(special_actions)

            special_actions = list(special_actions)

            # Keep the non_greedy token, as it will override the terminal token of the state if it exists
            if special_actions and special_actions[-1][0] == DFA.NON_GREEDY:
                non_greedy_action = special_actions.pop()
                non_greedy_token = non_greedy_action[1]
                non_greedy_token_exists = True

            else:
                non_greedy_token = None
                non_greedy_token_exists = False

            dfa_nodes_table[sub_id]['special_actions'] = special_actions

            # Add the terminal node
            # If a non_greedy special action was found, its token is taken
            # Otherwise we recover the token from the maximum priority token of the NFA nodes
            if non_greedy_token_exists:
                dfa_nodes_table[sub_id]['is_terminal'] = True
                dfa_nodes_table[sub_id]['terminal'] = non_greedy_token

            else:
                # Recover the terminal with maximum priority and set it as the terminal
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
        # In the process, the algorithm removes the error state

        minimum_dfa = hopcrofts_algorithm(dfa_nodes_table, alphabet)

        # ========================================================
        # Merge adjacent lookouts
        # ========================================================
        # The generation of the alphabet with get_minimal_covering_intervals made partitioning of some lookouts too fine
        # we will merge such lookouts. By example if from State x, the intervals (97,98) and (99, 102) lead to State y,
        # we merge the lookouts so that (97, 102) leads to State y.

        merge_adjacent_dfa_lookouts(minimum_dfa)

        # ========================================================
        # Build the final data structure of the DFA
        # ========================================================
        # dfa_nodes_table now contains all the information required to build the minimal DFA as a NodeDFA object

        dfa_start = build_dfa_from_dict(minimum_dfa, initial_epsilon_group)

        return dfa_start


# ======================================================================================================================
# Finite Automatons Building Helpers
# ======================================================================================================================

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
    active_states_as_dict = {}

    # Definition of what we consider an active state
    def is_active_state(state):
        return state['is_terminal'] or state['special_actions']

    for id in dfa_nodes_table:
        state = dfa_nodes_table[id]

        if is_active_state(state):
            # Action id is an hashable representation of the actions the state can lead to (terminal token an special
            # actions together. In particular, two states with the same returning behavior will have the same action_id
            action_id = (state['is_terminal'], state['terminal'], frozenset(state['special_actions']))

            if action_id in active_states_as_dict:
                active_states_as_dict[action_id].add(id)

            else:
                active_states_as_dict[action_id] = {id}

    active_states = {frozenset(states) for _, states in active_states_as_dict.items()}
    inactive_states = frozenset([id for id in dfa_nodes_table if
                                 not is_active_state(dfa_nodes_table[id])])

    partition = {inactive_states}
    partition |= active_states

    # Refine the sets
    sets_to_refine = active_states

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
    # What is done here is we look for the set of states in the partition that contains the error state, all the states
    # in this state behave like the error state and are thus an error. As for the early versions, it should never happen
    # that the error_state_id '()' is in a set with other states, but this solution has been implemented as as fix for
    # a bug that lied somewhere else, but was kept since it may actually be the correct generalization we want later
    for error_states in partition:
        if error_state_id in error_states:
            break
    else:
        raise LexerBuildError("lost error state in Hopcroft's algorithm")
    partition.remove(error_states)

    minimal_dfa = {}
    for states in partition:
        transitions = {}

        for state in states:
            for lookout, target in dfa_nodes_table[state]['transitions'].items():
                if target in error_states or lookout in transitions:
                    continue
                else:
                    transitions[lookout] = mapping[target]

        # Hack to recover an item from a set
        # We can use any node from the merged states since they all behave the same way
        for any in states:
            break

        minimal_dfa[states] = {'is_terminal': dfa_nodes_table[any]['is_terminal'],
                               'terminal': dfa_nodes_table[any]['terminal'],
                               'special_actions': dfa_nodes_table[any]['special_actions'],
                               'transitions': transitions}

    return minimal_dfa


def merge_adjacent_dfa_lookouts(dfa_as_dict):
    """
    For each state in the dfa_as_dict, recover the lookouts for each target in the transitions and merge adjacent
    transitions. Ex: (2, 5) -> y and (6, 9) -> y will be updated to (2, 9) -> y.
    Mutate dfa_as_dict
    """
    for id, state in dfa_as_dict.items():
        inverse_map = {}

        for lookout, target in state['transitions'].items():
            if target in inverse_map:
                inverse_map[target].append(lookout)
            else:
                inverse_map[target] = [lookout]

        new_transitions = {}

        for target, lookouts in inverse_map.items():
            new_lookouts = IntervalOp.merge_intervals(lookouts)

            for lookout in new_lookouts:
                new_transitions[lookout] = target

        state['transitions'] = new_transitions


def build_dfa_from_dict(dfa_as_dict, starting_state_id):
    """
    Given a table-like dict structure of a DFA and the id of the starting node in the table, build the graph of NodeDFA
    objects and return the starting state
    """

    # We first create the NodeDFA nodes to link them afterward
    dfa_nodes_as_dict = {sub_id: NodeDFA() for sub_id in dfa_as_dict}

    for sub_id, node in dfa_nodes_as_dict.items():

        # Set the terminal token
        is_terminal = dfa_as_dict[sub_id]['is_terminal']
        if is_terminal:
            token = dfa_as_dict[sub_id]['terminal']
            node.set_terminal_token(token)

        # Set the transition states
        for lookout, target in dfa_as_dict[sub_id]['transitions'].items():
            if target:
                node.add_transition_to_state(lookout[0], lookout[1], dfa_nodes_as_dict[target])

        # Set the special actions
        node.set_special_actions(dfa_as_dict[sub_id]['special_actions'])

        # We sort the lookouts for easy recovery
        node.sort_lookouts()

    # Finally we recover the starting node to return it
    for fset in dfa_nodes_as_dict:
        if starting_state_id in fset:
            initial_node_id = fset
            break
    else:
        raise LexerBuildError("For unknown reason, the build algorithm lost its own initial state. Puzzling.")

    return dfa_nodes_as_dict[initial_node_id]


def recover_nodes_set_from_nfa(nfa):
    """
    Given a Non-Deterministic Finite Automata, return a set of all the nodes it contains
    """
    nodes_set = set()
    nodes_queue = [nfa]

    while nodes_queue:
        node = nodes_queue.pop()

        if node not in nodes_set:
            nodes_set.add(node)

            for _, child in node.next_states:
                nodes_queue.append(child)

    return nodes_set


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


def sort_special_actions_list(sa_list):
    """
    Order a list of special actions (action, token, priority) in increasing order of priority, return a sorted list
    """

    return sorted(sa_list, key=lambda x: x[2])


def remove_priority_from_special_actions_list(sa_list):
    """
    Remove all priority from a list of (action, token, priority) special actions, returning a list of (action, token)
    """
    return map(lambda sa: sa[0:2], sa_list)


def truncate_special_action_list_at_non_greedy(sa_list):
    """
    Iterate through the special action list, stopping when it encounters a non_greedy special action type. Return the
    elements seen before that point, including the non greedy special action.
    If no non_greedy special_action is found, a copy of the list is returned
    """
    index = 0
    length = len(sa_list)
    truncated_list = []

    while index < length:
        truncated_list.append(sa_list[index])

        if sa_list[index][0] == DFA.NON_GREEDY:
            break

        index += 1

    return truncated_list


def node_list_to_sorted_tuple_of_id(node_list):
    return tuple(sorted([node.id for node in node_list]))


def node_list_to_set_of_id(node_list):
    return set([node.id for node in node_list])
