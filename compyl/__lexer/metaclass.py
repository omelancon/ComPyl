import re

class RuleHarvester(dict):
    def __init__(self, *args, **kwargs):
        self.lexer_rules = []
        self.terminal_actions = None
        self.params = None
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if re.match(r'__\w+__', key):
            super().__setitem__(key, value)

        else:
            self._add_rule_item(key, value)

    def get_rules(self):
        return self.lexer_rules

    def get_dict(self):
        return dict(self)

    def _add_rule_item(self, token, params):

        # Manage special keywords that are not rules
        if token == 'terminal_actions':
            if not self.terminal_actions:
                self.terminal_actions = params
            else:
                raise ValueError('duplicate terminal_actions keyword')

        elif token == 'params':
            if not self.params:
                self.params = params
            else:
                raise ValueError('duplicate params keyword')

        else:
            # Otherwise, the attribute is a rule

            # The following are optional and may not be provided
            instruction = None
            tag = None

            # Get the pattern
            if isinstance(params, (list, tuple)):
                pattern = params[0]

                # An instruction or a tag was provided after the pattern
                if len(params) == 2:

                    # We check type to see if an instruction or a tag was provided
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
                    raise ValueError('Too many arguments for rule')
            else:
                pattern = params

            # Manage reserved keywords for rules
            if token == 'line_rule':
                if instruction or tag:
                    raise ValueError('line_rule is reserved for line incrementation rule, cannot have tag or instruction')

                rule_item = self._get_line_rule_item(pattern)

            elif re.match('_+', token):
                rule_item = [(pattern, None, instruction, tag)]

            else:
                rule_item = [(pattern, token, instruction, tag)]

            self.lexer_rules += rule_item

    @staticmethod
    def _get_line_rule_item(pattern):
        def line_incrementer(t, v): t.increment_line()

        return [
            (pattern, None, None, None),
            (pattern, None, line_incrementer, 'trigger_on_contain')
        ]


class MetaLexer(type):
    def __prepare__(*args):
        return RuleHarvester()

    def __new__(cls, name, bases, rule_harvester):

        # MetaLexer is only meant to be used for the Lexer class below, so we allow the creation of the class as a
        # way to inherit the metaclass, but any other inheritance will not return a class.
        if not bases:
            return type.__new__(cls, name, bases, {})

        else:
            # use the rule_harvester to return a compyl.lexer.Lexer object
            raise NotImplemented


class Lexer(metaclass=MetaLexer):
    pass



