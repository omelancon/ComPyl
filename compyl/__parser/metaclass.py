import re
from compyl.__parser.error import ParserSyntaxError

class RuleHarvester:

    def __init__(self, *args, **kwargs):
        # self.dict contain function immediately accessible in class scope to add terminals
        self.dict = {
            'terminal': lambda *terminals: self.terminals.extend(terminals)
        }
        self.parser_rules = {}
        self.terminals = []
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if re.match(r'__\w+__', key):
            self.dict[key] = value

        else:
            self._add_rule_item(key, value)

    def __getitem__(self, key):
        return self.get_dict()[key]

    def __iter__(self):
        """
        Mimic the behaviour of __iter__ of dict and adds special values
        When unpickling, dill does not run __prepare__, thus it is necessary that the cast to a dict can be done
        with dict(), since __new__ is actually calling dict() on a dn actual dict.
        """
        for k, v in self.get_dict().items():
            yield k, v

    def get_dict(self):
        return dict(
            self.dict,
            __rules__=self.parser_rules,
            __terminals__=self.terminals,
        )

    def _add_rule_item(self, key, value):

        if isinstance(value, (list, tuple)):
            # Each rule must be a tuple (string, callable)
            # value is either a single such tuple or a list/tuple of those

            if len(value) == 2 and isinstance(value[0], str) and callable(value[1]):
                # Catch a single-rule value and reformat it as a tuple
                value = (value,)

            # Check that each rule obeys the (str, callable) format
            for rule in value:
                if len(rule) != 2 or not isinstance(rule[0], str) or not callable(rule[1]):
                    raise ParserSyntaxError('ill-formatted rule: {}'.format(key))

            self.parser_rules[key] = value

        else:
            raise ParserSyntaxError('ill-formatted rule: {}'.format(key))

class MetaParser(type):
    def __prepare__(name, bases):
        if not bases:
            return dict()

        else:
            return RuleHarvester()

    def __new__(cls, name, bases, name_space):

        # MetaParser is only meant to be used for the Parser class, so we allow the creation of the class as a
        # way to inherit the metaclass, but any other inheritance will not return a class.
        if not bases:
            return type.__new__(cls, name, bases, name_space)

        elif len(bases) == 1:
            # WARNING: When unpickling with dill, name_space is actually a dict with the keys __rules__ and
            # __terminal__ parsed already. This is fine, but we need to make sure that the cast from
            # RuleHarvester to dict will always be idempotent, i.e. dict(dict(name_space)) == dict(name_space)
            return super().__new__(cls, name, bases, dict(name_space))

        else:
            raise TypeError('Parser cannot be inherited along with other classes')
