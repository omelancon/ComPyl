import networkx
import matplotlib.pyplot as pyplot
from networkx.drawing.nx_agraph import write_dot


def plot_lexer_automata(fsa):

    # Recover nodes
    nodes = []
    todo_nodes = [fsa]

    while todo_nodes:
        node = todo_nodes.pop()
        nodes.append(node)

        for lookout, child in node.next_states.items():
            if child in nodes or child in todo_nodes:
                continue
            else:
                todo_nodes.append(child)

    # Recover edges
    edges = []

    for node in nodes:
        for key, child in node.next_states.items():
            edges.append((id(node), id(child)))

    graph = networkx.DiGraph()
    graph.add_nodes_from([id(node) for node in nodes])
    graph.add_edges_from(edges)

    relabel = {}
    unique_id = 0
    for node in nodes:
        str_unique_id = str(unique_id)

        if node.current_state == None:
            label = "START"
        elif node.current_state == -1:
            label = "EMPTY"
        else:
            label = "'" + chr(node.current_state) + "'"

        if label != "START":
            label = "N" + str_unique_id + ": " + label

            if node.terminal_token:
                if isinstance(node.terminal_token, str):
                    label += "\n" + node.terminal_token
                elif hasattr(node.terminal_token, '__call__'):
                    label += "\nfn: " + node.terminal_token.__name__
                else:
                    label += "\n" + "IGNORED"

        relabel[id(node)] = label

        unique_id += 1

    graph = networkx.relabel_nodes(graph, relabel)

    # Change color of nodes with self loops so they are visible with Networkx
    # TODO: Update the drawing to used Graphviz and have actual self loops
    loops = graph.nodes_with_selfloops()
    colors = {node: 'r' if node in loops else 'b' for node in graph.nodes()}

    color_map = [colors.get(node) for node in graph.nodes()]

    networkx.draw(graph, pos=hierarchy_pos(graph, "START"), with_labels=True, node_color=color_map)
    pyplot.savefig("visual_lexer.png")


###################
# http://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3

def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5 ):
    '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
    '''

    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None, parsed = [] ):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)
            if len(neighbors)!=0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G,neighbor, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                        parent = root, parsed = parsed)
        return pos

    return h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5)



