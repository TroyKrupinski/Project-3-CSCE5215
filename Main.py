import networkx as nx

G = nx.DiGraph()

G.add_node("ab")
print(G.nodes())

G.add_edges_from([("ab", "nc"), ("fb", "nc")])  # add edges to represent causative relationships
print(G.nodes)
print(G.edges)

G.add_node("ab", ab_y=0.1, ab_n=0.9)   # add node attributes - needed for attaching probability tables to nodes
G.add_node("fb", fb_y=0.3, fb_n=0.7)   # let's go down one level and attach a table to the "no charging node"
G.add_node("nc", ab_y_fb_y_nc=0.75, ab_y_fb_n_nc=0.4, ab_n_fb_y_nc=0.6, ab_n_fb_n_nc=0.1, ab_y_fb_y_nc_n=0.25, ab_y_fb_n_nc_n=0.6, ab_n_fb_y_nc_n=0.4, ab_n_fb_n_nc_n=0.9)
# let's go down one level and attach a probability table to the "no charging node
print(G.edges)
