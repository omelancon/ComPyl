import networkx
import matplotlib.pyplot as pyplot


def plot_nfa(nfa):

    nodes = []
    edges = []
    todo_nodes = [nfa]

    edge_labels = {}
    node_labels = {}
    nodes_obj_as_dict = {}

    while todo_nodes:
        node = todo_nodes.pop()
        nodes.append(node)
        nodes_obj_as_dict[node.id] = node

        node_labels[node.id] = node.id

        for lookout, child in node.next_states:

            edges.append((node.id, child.id))

            if 0 <= lookout[0] < lookout[1] <= 256:
                if 32 < lookout[0] < 128:
                    fst = chr(lookout[0])
                else:
                    fst = '???'

                if 32 < lookout[1] < 128:
                    snd = chr(lookout[1])
                else:
                    snd = '???'

                edge_labels[(node.id, child.id)] = "[%s-%s]" % (fst, snd)

            elif -1 < lookout[0] == lookout[1]:
                if 32 < lookout[1] < 128:
                    char = chr(lookout[1])
                else:
                    char = '???'

                edge_labels[(node.id, child.id)] = "'%s'" % char

            if child in nodes or child in todo_nodes:
                continue
            else:
                todo_nodes.append(child)

    graph = networkx.DiGraph()
    graph.add_nodes_from([node.id for node in nodes])
    graph.add_edges_from(edges)

    # TODO: Update the drawing to used Graphviz and have actual self loops
    #loops = graph.nodes_with_selfloops()

    colors = {node: 'r' if nodes_obj_as_dict[node].terminal_token else 'b' for node in graph.nodes()}

    color_map = [colors.get(node) for node in graph.nodes()]

    networkx.draw(graph, pos=hierarchy_pos(graph, 0), with_labels=True, node_color=color_map)
    networkx.draw_networkx_edge_labels(graph, hierarchy_pos(graph, 0), edge_labels=edge_labels)
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



