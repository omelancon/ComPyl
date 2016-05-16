# Brainstorm for syntax

# We probably want to keep the same idea of using dicts as in the Lexer. Something like


class Node:
    def __init__(self, *args):
        self.children = args


def A(*args):
    print "Do something here"
    return Node(args[0])


def B(*args):
    return Node(args)

rules = {
    "many_A": {
        ("A", "A", "A"): A,
        ("A", "B", "A"): Node
    },
    "many_B": {
        "B": B
    }
}

# Notice: this time we force passing functions and do not allow strings and implicitly build nodes
#   this choice imitates most other parsers and still allows to pass a class (see the Node) as it can
#   be used to call the __init__ function and actually return a node.
#   Thus the AST elements must be built by the user

# Parser will have an optional start keyword, otherwise


# We then want to generate a PDA, and have a .export() function to store it in a file
# It seems pretty obvious that we want to use an underlying dict to build the PDA
