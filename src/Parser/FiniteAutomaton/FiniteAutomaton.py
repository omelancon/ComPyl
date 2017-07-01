class LrItem:
    def __init__(self, parsed, expected):
        self.parsed = tuple(parsed)
        self.expected = tuple(expected)

    def __hash__(self):
        return hash((self.parsed, self.expected))

    def is_fully_parsed(self):
        return False if self.expected else True

    def get_next_expected_token(self):
        return self.expected[0]

    def move_demarkator(self):
        return LrItem(self.parsed + [self.expected[0]], self.expected[1:])


def get_closure(initial_items, rules):
    """
    :param initial_states: List of LR_Item's
    :param rules: Parsed rules
    :return: Closure as tuple of LR_Item's
    """
    closure = set()
    pending_items = set(initial_items)

    while pending_items:
        next_pending_items = set()

        for item in pending_items:
            if not item.is_fully_parsed():
                next_token = item.get_next_expected_token()

                try:
                    next_token_rules = rules[next_token]
                except KeyError:
                    continue

                for rule, _ in next_token_rules:
                    new_item = LrItem([], rule)
                    next_pending_items.add(new_item)

        closure = closure.union(pending_items)
        pending_items = next_pending_items

    return tuple(closure)



