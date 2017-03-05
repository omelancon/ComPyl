from src.Lexer.Lexer import Lexer
from src.Visual import visual_lexer
import copy


def vowel_counter(t):
    t.params['vowels'] += 1


def reset_lexer_params(t):
    t.params['vowels'] = 0


def WORD(t):
    return "WORD", {'vowels': t.params['vowels']}

rules = [
    (r"[aeiouyAEIOUY]", vowel_counter, "trigger_on_contain"),
    (r"\w+", WORD),
    (r"/--_*--/", None, "non_greedy"),
    (r" ", None)
]

terminal_actions = [
    (reset_lexer_params, "always")
]

lexer = Lexer(
    rules=rules,
    line_rule="\n",
    terminal_actions=terminal_actions,
    params={'vowels': 0}
)

lexer.build()

buffer = """
i say hello and you say goodbye world
/-- Some comment --/
We Also TAKE Capitals and UNDER_scores
"""

visual_lexer.plot_dfa(lexer.dfa.start)

lexer.read(buffer)

new = copy.deepcopy(lexer)

new.save("test.p")

loaded = Lexer.load("test.p")

tk = True

while tk:
    tk = loaded.lex()
    print(tk)
    if tk and tk.type == "WORD":
        print("%s has %d vowels" % (tk.value, tk.params['vowels']))
