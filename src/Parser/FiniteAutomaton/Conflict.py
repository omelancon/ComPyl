class ConflictError(Exception):
    pass


class Conflict:
    def __init__(self, type, lookout, path):
        if type == "shift/reduce" or type == "reduce/reduce":
            self.type = type
            self.path = path
            self.lookout = lookout
            self.reduce_reduce_targets = []
            self.shift_reduce_target = None

        else:
            raise ConflictError("Invalid type for Conflict: " + str(type))

    def add_reduce_reduce_conflict(self, reduce_reduce_targets):
        if self.type == "reduce/reduce":
            self.reduce_reduce_targets.append(reduce_reduce_targets)
        else:
            raise ConflictError("Cannot use add_reduce_reduce_conflict on shift/reduce conflict")

    def add_shift_reduce_conflict(self, shift_reduce_target):
        if self.type == "shift/reduce":
            self.shift_reduce_target = shift_reduce_target
        else:
            raise ConflictError("Cannot use add_shift_reduce_conflict on reduce/reduce conflict")

    def to_string(self):
        if self.type == 'reduce/reduce':
            return self.type + ': ' + ' '.join(self.path) + ' . ' + self.lookout
        elif self.type == 'shift/reduce':
            return self.type + ': ' + ' '.join(self.path) + ' . ' + self.lookout


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
