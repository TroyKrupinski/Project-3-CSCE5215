import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
G = nx.DiGraph()

#Troy Krupinski
#CSCE 5215 - Machine Learning
#Project 3 - Bayesian Networks



G.add_node("battery_age", ba_y=.2, ba_n = .8)
G.add_node("alternator_broken", ab_y=.1, ab_n = .9)
G.add_node("fanbelt_broken", fb_y = .3, fb_n = .7)
G.add_node("battery_dead", ba_y_bd_y = .7, ba_y_bd_n = .3, ba_n_bd_y = .3, ba_n_bd_n = .7)

G.add_node("no_charging_table",
           
           ab_y_fb_y_nc_y = .75,
           ab_y_fb_n_nc_y = .4,
           ab_n_fb_y_nc_y = .6,
           ab_n_fb_n_nc_y = .1,
           ab_y_fb_y_nc_n= 0.25,
           ab_y_fb_n_nc_n = .6,
           ab_n_fb_y_nc_n = .4,
           ab_n_fb_n_nc_n = .9)

G.add_node("battery_flat",
    bd_y_nc_y_bf_y=0.95, bd_y_nc_n_bf_y=0.85,
    bd_n_nc_y_bf_y=0.8, bd_n_nc_n_bf_y=0.1,
    bd_y_nc_y_bf_n=0.05, bd_y_nc_n_bf_n=0.15,
    bd_n_nc_y_bf_n=0.2, bd_n_nc_n_bf_n=0.9
)

G.add_node("no_oil", no_y=0.05, no_n=0.95)
G.add_node("lights", 
    l_y_bf_y=0.9, l_n_bf_y=0.1,
    l_y_bf_n=0.3, l_n_bf_n=0.7
)
G.add_node("gas_gauge",
    bf_y_gg_y=0.1, bf_n_gg_y=0.95,
    bf_y_gg_n=0.9, bf_n_gg_n=0.05
)
G.add_node("car_wont_start",
    bf_y_no_y_cs_n=0.9, bf_y_no_n_cs_n=0.9,
    bf_n_no_y_cs_n=0.9, bf_n_no_n_cs_n=0.1,
    bf_y_no_y_cs_y=0.1, bf_y_no_n_cs_y=0.1,
    bf_n_no_y_cs_y=0.1, bf_n_no_n_cs_y=0.9
)

G.add_node("dipstick_low",
    no_y_dl_y=0.95, no_n_dl_y=0.3,
    no_y_dl_n=0.05, no_n_dl_n=0.7
)
edges = [
    ("battery_age", "battery_dead"),
    ("alternator_broken", "no_charging_table"),
    ("fanbelt_broken", "no_charging_table"),
    ("no_charging_table", "battery_flat"),
    ("battery_dead", "battery_flat"),
    ("battery_flat", "lights"),
    ("battery_flat", "gas_gauge"),
    ("battery_flat", "car_wont_start"),
    ("no_oil", "dipstick_low"),
    ("no_oil", "car_wont_start")

]

G.add_edges_from(edges)

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, k=1, iterations=50)

nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", font_color="black", edge_color="gray", linewidths=1, arrowsize=20)



plt.title("Bayesian Network - Project 3")
plt.show()


fanbelt_probs = G.nodes["fanbelt_broken"]
battery_age_probs = G.nodes["battery_age"]
alternator_probs = G.nodes["alternator_broken"]


battery_dead_probs = G.nodes["battery_dead"]
no_charging_probs = G.nodes["no_charging_table"]


battery_flat_probs = G.nodes["battery_flat"]
no_oil_probs = G.nodes["no_oil"]

lights_probs = G.nodes["lights"]
gas_gauge_probs = G.nodes["gas_gauge"]
car_wont_start_probs = G.nodes["car_wont_start"]
dipstick_low_probs = G.nodes["dipstick_low"]

def car_fanbelt(G):
    # P(+cws|+fb) = P(+cws,+fb) / P(+fb)
    # P(+cws,+fb) = P(+cws|+fb) * P(+fb)
    # P(+cws) = P(+cws,+fb) + P(+cws,-fb)
    return                



print("p(+cws|+fb)")
car_fanbelt(G)  

print(G.nodes())
