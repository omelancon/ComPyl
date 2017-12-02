from src.Lexer.Lexer import Lexer
from src.Parser.Parser import Parser

lexer_rules = [
    (r'if', 'IF'),
    (r'then', 'THEN'),
    (r'[0-9]+', 'INT'),
    (r'else', 'ELSE'),
    (r'end', 'END'),
    (r'=', 'EQUAL'),
    (r'\w+', 'VAR'),
    (r' ', None)
]

lexer = Lexer(
    rules=lexer_rules,
    line_rule="\n",
)
lexer.build()

class Code:
    def __init__(self, statement, next=None):
        self.statements = [statement.value]
        if next:
            self.statements += next.value.statements

class If:
    def __init__(self, *args):
        self.cond = args[1].value
        self.then_statement = args[3].value
        self.else_statement = args[4].value

class Dec:
    def __init__(self, *args):
        self.var = args[0].value
        self.value = args[2].value

parser_rules = {
    'code': [
        ('stat code', Code),
        ('stat', Code)
    ],
    'stat': [
        ('IF VAR THEN stat else? END', If),
        ('VAR EQUAL INT', Dec)
    ],
    'else': [
        ('ELSE stat', lambda x, y: y)
    ]
}

parser = Parser(rules=parser_rules, terminal='code')
parser.build()

buffer = """
x = 0
if x then y = 2 else y = 1 end
"""

lexer.read(buffer)
t = lexer.lex()
while t:
    parser.parse(t)
    t = lexer.lex()

code = parser.end()
print(code)
