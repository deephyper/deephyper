import networkx as nx


class NxStructure:
    """A NxStructure is a structure based on a networkx graph.
    """

    def __init__(self, *args, **kwargs):
        self.graph = nx.DiGraph()

    def draw_graphviz(self, path):
        with open(path, 'w') as f:
            try:
                nx.nx_agraph.write_dot(self.graph, f)
            except:
                print('Error: can\'t create graphviz file...')
