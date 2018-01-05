class _ConflictError(Exception):
    pass


class ReduceCycle:
    def __init__(self, cycle):
        self.cycle = cycle

    def to_string(self):
        return 'reduce cycle: the following reduction will never terminate\n' + ' ' * len('reduce cycle  ') + \
               ' -> '.join(self.cycle)


class Conflict:
    def __init__(self, type, lookout, path, node=None):
        if type == "shift/reduce" or type == "reduce/reduce":
            self.type = type
            self.path = path
            self.node = node
            self.lookout = lookout
            self.reduce_reduce_conflict = []
            self.shift_reduce_conflict = None

        else:
            raise _ConflictError("Invalid type for Conflict: " + str(type))

    def is_shift_reduce(self):
        return self.type == 'shift/reduce'

    def is_reduce_reduce(self):
        return self.type == 'reduce/reduce'

    def add_reduce_reduce_conflict(self, reduce_reduce_conflict):
        if self.is_reduce_reduce():
            self.reduce_reduce_conflict = reduce_reduce_conflict
        else:
            raise _ConflictError("Cannot use add_reduce_reduce_conflict on shift/reduce conflict")

    def add_shift_reduce_conflict(self, shift_reduce_conflict):
        if self.is_shift_reduce():
            self.shift_reduce_conflict = shift_reduce_conflict
        else:
            raise _ConflictError("Cannot use add_shift_reduce_conflict on reduce/reduce conflict")

    def to_string(self):
        margin = '\n' + ' ' * (len(self.type) + 2)
        lookout = ['.', self.lookout] if self.lookout is not None else ['.']

        if self.type == 'reduce/reduce':
            return self.type + ': ' + ' '.join(self.path + lookout) + '  can reduce to' + margin + \
                   margin.join(
                       [' '.join((self.path[:-r['reduce_len']] if r['reduce_len'] > 0 else self.path) + [
                           r['token']] + lookout) for r in self.reduce_reduce_conflict]
                   )

        elif self.type == 'shift/reduce':
            return self.type + ': ' + ' '.join(self.path + lookout) + '  can shift or reduce to' + \
                   margin + ' '.join((self.path[:-self.shift_reduce_conflict['reduce_len']] if
                                      self.shift_reduce_conflict['reduce_len'] > 0 else self.path) + [
                                         self.shift_reduce_conflict['token']] + lookout)


def find_node_conflict(node, path):
    """
    Return a list of conflicts found at this node
    :param node: TmpNodeFiniteAutomaton
    :return: List of Conflicts
    """
    conflicts = []

    for lookout, reduce_elements in node.reduce.items():
        if len(reduce_elements) > 1:
            conflict = Conflict("reduce/reduce", lookout, path, node)
            conflict.add_reduce_reduce_conflict(reduce_elements)
            conflicts.append(conflict)

        if lookout in node.reduce and lookout in node.shifts:
            conflict = Conflict("shift/reduce", lookout, path, node)
            conflict.add_shift_reduce_conflict(reduce_elements[0])
            conflicts.append(conflict)

    return conflicts


def find_conflicts(dfa_initial_node):
    """
    Return a list of conflicts found in the DFA.
    :param dfa_initial_node: TmpNodeFiniteAutomaton at the root of the DFA
    :return: List of Conflict
    """
    conflicts = []
    seen_nodes = {}
    queue = [(dfa_initial_node, [])]

    while queue:
        node, path = queue.pop()

        for lookout, child in node.shifts.items():
            if child not in seen_nodes:
                queue.append((child, path + [lookout]))
                seen_nodes[child] = True

        node_conflicts = find_node_conflict(node, path)
        if node_conflicts:
            conflicts += node_conflicts

    return conflicts
