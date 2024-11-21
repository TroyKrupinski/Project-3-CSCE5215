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

'''
P(B|+j, +m) =

P(B, e, a, +j, +m)
    P(+j, +m)
    Let us take P(B, e, a, +j, +m).
        Now P(B, e, a, +j, +m) = ∑e,a P(B, e, a, +j, +m) = ∑ P(B) × P(e) × P(a|B, e) × P(+j|a) × P(+m|a) e,a
        = P(B) × P(+e) × P(+a|B, +e) × P(+j|+a) × P(+m|+a) + P(B) × P(+e) × P(−a|B, +e) × P(+j|−a) ×
        P(+m|−a) + P(B) × P(−e) × P(+a|B, −e) × P(+j|+a) × P(+m|+a) + P(B) × P(−e) × P(−a|B, −e) ×
        P(+j|−a) × P(+m|−a)
'''

def car_fanbelt(G):
    # P(+cws|+fb) = P(+cws,+fb) / P(+fb)
    # P(+cws,+fb) = P(+cws|+fb) * P(+fb)
    # P(+cws) = P(+cws,+fb) + P(+cws,-fb)
    fb_y = G.nodes["fanbelt_broken"]["fb_y"] # P(+fb)
    fb_n = G.nodes["fanbelt_broken"]["fb_n"] # P(-fb)

    ab_y_fb_y_nc_y = G.nodes["no_charging_table"]["ab_y_fb_y_nc_y"] # P(+No Charging Table | +Alternator, +Fanbelt)
    ab_y_fb_n_nc_y = G.nodes["no_charging_table"]["ab_y_fb_n_nc_y"] # P(+No Charging Table | +Alternator, -Fanbelt)
    ab_n_fb_y_nc_y = G.nodes["no_charging_table"]["ab_n_fb_y_nc_y"] # P(+No Charging Table | -Alternator, +Fanbelt)
    ab_n_fb_n_nc_y = G.nodes["no_charging_table"]["ab_n_fb_n_nc_y"] # P(+No Charging Table | -Alternator, -Fanbelt)
     
        # Compute P(nc_y | +fb)
    p_nc_y_given_fb = (
        ab_y_fb_y_nc_y * G.nodes["alternator_broken"]["ab_y"] +
        ab_n_fb_y_nc_y * G.nodes["alternator_broken"]["ab_n"]
    )

    # Compute P(+cws | +fb)
    p_cws_given_fb = (
        G.nodes["car_wont_start"]["bf_y_no_y_cs_y"] * fb_y * p_nc_y_given_fb
    )

    # Return the computed probability
    return p_cws_given_fb

    

print("p(+cws|+fb)")
R2 = car_fanbelt(G)
print(f"P(+cws | +fb): {R2}")

print(G.nodes())
