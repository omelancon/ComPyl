import re
from compyl.__lexer.errors import LexerSyntaxError

# _Terminal is a bride between the old API which received either a string or a function as token
# Since the role of the new token of type function has changed, it no longer returns the token, both can
# be provided at once. This class mimics the behaviour of token as functions.


def get_callable_terminal_token(token, instruction):

    if instruction is None:
        return lambda *args: token

    else:
        def _call_instruction(*args):
            return token, instruction(*args)

        return _call_instruction



class RuleHarvester():
    """
    Class that mimics the behaviour of a dict but filters non-magic functions (such as __init__) to allow duplicates.
    It then returns special keys __rules__, __terminal_tokens__ and __params__
    Used in __prepare__ method of MetaLexer to gather rules and bunch them in a single parsed list
    """
    def __init__(self, *args, **kwargs):
        # The initial dict contains functions accessible in the class scope to set non-rule properties
        self.dict = {
            'terminal_actions': lambda *actions: self.terminal_actions.extend(actions),
            'params': lambda params=None, **kwargs: self.params.update(params or {}, **kwargs),
            'line_rule': lambda pattern: self.lexer_rules.extend(self._get_line_rule_item(pattern))
        }
        self.lexer_rules = []
        self.terminal_actions = []
        self.params = {}
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
            __rules__=self.lexer_rules,
            __terminal_actions__=self.terminal_actions,
            __params__=self.params
        )

    def _add_rule_item(self, token, params):
        # The following are optional and may not be provided
        instruction = None
        tag = None

        # Get the pattern
        if isinstance(params, (list, tuple)):
            pattern = params[0]

            # An instruction or a tag was provided after the pattern
            if len(params) == 2:

                # We check type to see which of an instruction or a tag was provided
                if callable(params[1]):
                    instruction = params[1]

                else:
                    tag = params[1]

            # An instruction and a tag were provided after the pattern
            elif len(params) == 3:

                # The expected syntax in that case is (pattern, instruction, tag)
                instruction = params[1]
                tag = params[2]

            else:
                raise LexerSyntaxError('Too many arguments for rule')
        else:
            pattern = params

        # Case for ignored patterns
        if re.match('_+', token):
            rule_item = [(pattern, get_callable_terminal_token(None, instruction), tag)]

        else:
            rule_item = [(pattern, get_callable_terminal_token(token, instruction), tag)]

        self.lexer_rules += rule_item

    @staticmethod
    def _get_line_rule_item(pattern):
        def line_incrementer(t): t.increment_line()

        return [
            (pattern, get_callable_terminal_token(None, None), None),
            (pattern, get_callable_terminal_token(None, line_incrementer), 'trigger_on_contain')
        ]


class MetaLexer(type):
    def __prepare__(name, bases):
        if not bases:
            return dict()

        else:
            return RuleHarvester()

    def __new__(cls, name, bases, name_space):

        # MetaLexer is only meant to be used for the Lexer class below, so we allow the creation of the class as a
        # way to inherit the metaclass, but any other inheritance will not return a class.
        if not bases:
            return type.__new__(cls, name, bases, name_space)

        elif len(bases) == 1:
            # use the rule_harvester to return a compyl.lexer.Lexer object
            # We expect compyl.lexer.Lexer to be the only class to have MetaLexer as metaclass
            # name_space is of type RuleHarvester at that point

            # WARNING: When unpickling with dill, name_space is actually a dict with the keys __params__, __rules__ and
            # __terminal_actions__ parsed already. This is fine, but we need to make sure that the cast from
            # RuleHarvester to to dict will always be idempotent, i.e. dict(dict(name_space)) == dict(name_space)
            return super().__new__(cls, name, bases, dict(name_space))

        else:
            raise TypeError('Lexer cannot be inherited along with other classes')
