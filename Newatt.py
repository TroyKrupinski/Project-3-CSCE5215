import itertools

def calculate_prob_car_wont_start():
    # Prior probabilities
    P_battery_age = {'y': 0.2, 'n': 0.8}
    P_alternator_broken = {'y': 0.1, 'n': 0.9}
    P_fanbelt_broken = {'y': 0.3, 'n': 0.7}
    P_no_oil = {'y': 0.05, 'n': 0.95}

    # Conditional probability tables
    P_battery_dead = {
        ('y', 'y'): 0.7,  # P(BD=y|BA=y)
        ('y', 'n'): 0.3,  # P(BD=n|BA=y)
        ('n', 'y'): 0.3,  # P(BD=y|BA=n)
        ('n', 'n'): 0.7   # P(BD=n|BA=n)
    }

    P_no_charging = {
        ('y', 'y', 'y'): 0.75,  # P(NC=y|AB=y,FB=y)
        ('y', 'n', 'y'): 0.4,   # P(NC=y|AB=y,FB=n)
        ('n', 'y', 'y'): 0.6,   # P(NC=y|AB=n,FB=y)
        ('n', 'n', 'y'): 0.1,   # P(NC=y|AB=n,FB=n)
        ('y', 'y', 'n'): 0.25,  # P(NC=n|AB=y,FB=y)
        ('y', 'n', 'n'): 0.6,   # P(NC=n|AB=y,FB=n)
        ('n', 'y', 'n'): 0.4,   # P(NC=n|AB=n,FB=y)
        ('n', 'n', 'n'): 0.9    # P(NC=n|AB=n,FB=n)
    }

    P_battery_flat = {
        ('y', 'y', 'y'): 0.95,  # P(BF=y|BD=y,NC=y)
        ('y', 'n', 'y'): 0.85,  # P(BF=y|BD=y,NC=n)
        ('n', 'y', 'y'): 0.8,   # P(BF=y|BD=n,NC=y)
        ('n', 'n', 'y'): 0.1,   # P(BF=y|BD=n,NC=n)
        ('y', 'y', 'n'): 0.05,  # P(BF=n|BD=y,NC=y)
        ('y', 'n', 'n'): 0.15,  # P(BF=n|BD=y,NC=n)
        ('n', 'y', 'n'): 0.2,   # P(BF=n|BD=n,NC=y)
        ('n', 'n', 'n'): 0.9    # P(BF=n|BD=n,NC=n)
    }

    P_car_wont_start = {
        ('y', 'y', 'y'): 0.1,  # P(CWS=y|BF=y,NO=y)
        ('y', 'n', 'y'): 0.1,  # P(CWS=y|BF=y,NO=n)
        ('n', 'y', 'y'): 0.1,  # P(CWS=y|BF=n,NO=y)
        ('n', 'n', 'y'): 0.9,  # P(CWS=y|BF=n,NO=n)
        ('y', 'y', 'n'): 0.9,  # P(CWS=n|BF=y,NO=y)
        ('y', 'n', 'n'): 0.9,  # P(CWS=n|BF=y,NO=n)
        ('n', 'y', 'n'): 0.9,  # P(CWS=n|BF=n,NO=y)
        ('n', 'n', 'n'): 0.1   # P(CWS=n|BF=n,NO=n)
    }

    # Calculate P(car_won't_start_y|Fanbelt_broken_y)
    total_prob = 0
    evidence = 'y'  # Fanbelt_broken = y

    # Generate all possible combinations of other variables
    states = ['y', 'n']
    variables = ['battery_age', 'alternator_broken', 'battery_dead', 'no_charging', 'battery_flat', 'no_oil']
    
    for combo in itertools.product(states, repeat=len(variables)):
        ba, ab, bd, nc, bf, no = combo
        
        # Calculate probability for this combination
        p = (P_battery_age[ba] *
             P_alternator_broken[ab] *
             P_battery_dead[(ba, bd)] *
             P_no_charging[(ab, evidence, nc)] *
             P_battery_flat[(bd, nc, bf)] *
             P_no_oil[no] *
             P_car_wont_start[(bf, no, 'y')])  # We want car_won't_start = y
        
        total_prob += p

    return total_prob

# Calculate and print the result
result = calculate_prob_car_wont_start()
print(f"P(car_won't_start_y|Fanbelt_broken_y) = {result:.4f}")