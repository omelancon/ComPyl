class ConflictError(Exception):
    pass


class Conflict:
    def __init__(self, type, node=None):
        self.path = None
        self.lookout = None
        self.reduce_reduce_targets = None
        self.shift_reduce_target = None

        if type == "shift/reduce" or type == "reduce/reduce":
            self.type = type
            self.node = node

        else:
            raise ConflictError("Invalid type for Conflict: " + str(type))

    def set_path(self, node):
        self.path = []
        while node.shift_parent:
            self.path = [node.shift_parent_lookout] + self.path
            node = node.shift_parent

    # TODO: The use of reducer encapsulate the token to which a stream is reduced, making us unable to store it for now

    def add_reduce_reduce_conflict(self, lookout, reduce_reduce_targets=None):
        if self.type == "reduce/reduce":
            self.reduce_reduce_targets = []
            self.lookout = lookout
            self.reduce_reduce_targets.append(reduce_reduce_targets)
        else:
            raise ConflictError("Cannot use add_reduce_reduce_conflict on shift/reduce conflict")

    def add_shift_reduce_conflict(self, lookout, shift_reduce_target=None):
        if self.type == "shift/reduce":
            self.lookout = lookout
            self.shift_reduce_target = shift_reduce_target
        else:
            raise ConflictError("Cannot use add_shift_reduce_conflict on reduce/reduce conflict")

    def to_string(self):
        return self.type + ": " + ' '.join(self.path)



def find_node_conflict(node):
    """
    Return a list of conflicts found at this node
    :param node: TmpNodeFiniteAutomaton
    :return: List of Conflicts
    """
    conflicts = []

    for lookout, targets in node.reduce.items():
        if len(targets) > 1:
            conflict = Conflict("reduce/reduce", node)
            conflict.add_reduce_reduce_conflict(lookout)
            conflict.set_path(node)
            conflicts.append(conflict)

        elif lookout in node.reduce and lookout in node.shifts:
            conflict = Conflict("shift/reduce", node)
            conflict.add_shift_reduce_conflict(lookout)
            conflict.set_path(node)
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
    queue = [dfa_initial_node]

    while queue:
        node = queue.pop()

        for lookout, child in node.shifts.items():
            if child not in seen_nodes:
                queue.append(child)

        conflict = find_node_conflict(node)
        if conflict:
            conflicts.append(conflict)

    return conflicts
