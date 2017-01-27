import networkx
import matplotlib.pyplot as pyplot


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
        if node.current_state == "":
            relabel[id(node)] = "State " + str(unique_id) + ": E"
        elif node.current_state == -1:
            relabel[id(node)] = "State " + str(unique_id) + ": e"
        else:
            relabel[id(node)] = "State " + str(unique_id) + ": '" + chr(node.current_state) + "'"

        unique_id += 1

    graph = networkx.relabel_nodes(graph, relabel)

    networkx.draw(graph, pos=hierarchy_pos(graph), with_labels=True)
    pyplot.savefig("visual_lexer.png")


###################

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
            if parent != None:
                neighbors.remove(parent)
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



