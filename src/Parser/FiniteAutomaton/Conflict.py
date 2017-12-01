class ConflictError(Exception):
    pass


class Conflict:
    def __init__(self, type, lookout, path):
        if type == "shift/reduce" or type == "reduce/reduce":
            self.type = type
            self.path = path
            self.lookout = lookout
            self.reduce_reduce_conflict = []
            self.shift_reduce_conflict = None

        else:
            raise ConflictError("Invalid type for Conflict: " + str(type))

    def add_reduce_reduce_conflict(self, reduce_reduce_conflict):
        if self.type == "reduce/reduce":
            self.reduce_reduce_conflict = reduce_reduce_conflict
        else:
            raise ConflictError("Cannot use add_reduce_reduce_conflict on shift/reduce conflict")

    def add_shift_reduce_conflict(self, shift_reduce_conflict):
        if self.type == "shift/reduce":
            self.shift_reduce_conflict = shift_reduce_conflict
        else:
            raise ConflictError("Cannot use add_shift_reduce_conflict on reduce/reduce conflict")

    def to_string(self):
        margin = '\n' + ' ' * (len(self.type) + 2)

        if self.type == 'reduce/reduce':
            return self.type + ': ' + ' '.join(self.path) + ' . ' + self.lookout + '  can reduce to' + margin + \
                   margin.join(
                       [' '.join(self.path[:-r['reduce_len']] + [r['token']]) + ' . ' + self.lookout for r in
                        self.reduce_reduce_conflict]
                   )

        elif self.type == 'shift/reduce':
            return self.type + ': ' + ' '.join(self.path) + ' . ' + self.lookout + '  can shift or reduce to' +\
                   margin + ' '.join(self.path[:-self.shift_reduce_conflict['reduce_len']] + [
                       self.shift_reduce_conflict['token']]) + ' . ' + self.lookout


def find_node_conflict(node, path):
    """
    Return a list of conflicts found at this node
    :param node: TmpNodeFiniteAutomaton
    :return: List of Conflicts
    """
    conflicts = []

    for lookout, reduce_elements in node.reduce.items():
        if len(reduce_elements) > 1:
            conflict = Conflict("reduce/reduce", lookout, path)
            conflict.add_reduce_reduce_conflict(reduce_elements)
            conflicts.append(conflict)

        elif lookout in node.reduce and lookout in node.shifts:
            conflict = Conflict("shift/reduce", lookout, path)
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
