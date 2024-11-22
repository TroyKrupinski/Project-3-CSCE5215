import numpy as np

class CarBayesianNetwork:
    def __init__(self):
        # Initialize probability tables
        self.prob_tables = {
            'battery_age': {'y': 0.2, 'n': 0.8},
            'alternator_broken': {'y': 0.1, 'n': 0.9},
            'fanbelt_broken': {'y': 0.3, 'n': 0.7},
            'no_oil': {'y': 0.05, 'n': 0.95},
            
            'battery_dead': {
                ('y', 'y'): 0.7,  # P(+dead|+age)
                ('y', 'n'): 0.3,  # P(+dead|-age)
                ('n', 'y'): 0.3,  # P(-dead|+age)
                ('n', 'n'): 0.7   # P(-dead|-age)
            },
            
            'no_charging': {
                ('y', 'y', 'y'): 0.75,  # P(+charging|+alt,+fan)
                ('y', 'y', 'n'): 0.25,  # P(-charging|+alt,+fan)
                ('y', 'n', 'y'): 0.4,   # P(+charging|+alt,-fan)
                ('y', 'n', 'n'): 0.6,   # P(-charging|+alt,-fan)
                ('n', 'y', 'y'): 0.6,   # P(+charging|-alt,+fan)
                ('n', 'y', 'n'): 0.4,   # P(-charging|-alt,+fan)
                ('n', 'n', 'y'): 0.1,   # P(+charging|-alt,-fan)
                ('n', 'n', 'n'): 0.9    # P(-charging|-alt,-fan)
            },
            
            'battery_flat': {
                ('y', 'y', 'y'): 0.95,  # P(+flat|+charging,+dead)
                ('y', 'y', 'n'): 0.05,  # P(-flat|+charging,+dead)
                ('y', 'n', 'y'): 0.85,  # P(+flat|-charging,+dead)
                ('y', 'n', 'n'): 0.15,  # P(-flat|-charging,+dead)
                ('n', 'y', 'y'): 0.8,   # P(+flat|+charging,-dead)
                ('n', 'y', 'n'): 0.2,   # P(-flat|+charging,-dead)
                ('n', 'n', 'y'): 0.1,   # P(+flat|-charging,-dead)
                ('n', 'n', 'n'): 0.9    # P(-flat|-charging,-dead)
            },
            
            'car_wont_start': {
                ('y', 'y', 'y'): 0.1,   # P(+start|+oil,+flat)
                ('y', 'y', 'n'): 0.9,   # P(-start|+oil,+flat)
                ('y', 'n', 'y'): 0.1,   # P(+start|+oil,-flat)
                ('y', 'n', 'n'): 0.9,   # P(-start|+oil,-flat)
                ('n', 'y', 'y'): 0.1,   # P(+start|-oil,+flat)
                ('n', 'y', 'n'): 0.9,   # P(-start|-oil,+flat)
                ('n', 'n', 'y'): 0.9,   # P(+start|-oil,-flat)
                ('n', 'n', 'n'): 0.1    # P(-start|-oil,-flat)
            }
        }

    def calculate_p_car_will_start_given_fanbelt(self, fanbelt_state='y'):
        total_prob = 0
        
        # Iterate through all possible states
        for alt in ['y', 'n']:
            for bat_age in ['y', 'n']:
                for no_oil in ['y', 'n']:
                    # Calculate probabilities for each path
                    p_alt = self.prob_tables['alternator_broken'][alt]
                    p_bat_age = self.prob_tables['battery_age'][bat_age]
                    p_no_oil = self.prob_tables['no_oil'][no_oil]
                    
                    # Calculate P(battery_dead|battery_age)
                    for bat_dead in ['y', 'n']:
                        p_bat_dead = self.prob_tables['battery_dead'][(bat_dead, bat_age)]
                        
                        # Calculate P(no_charging|alternator,fanbelt)
                        for no_charging in ['y', 'n']:
                            p_no_charging = self.prob_tables['no_charging'][(alt, fanbelt_state, no_charging)]
                            
                            # Calculate P(battery_flat|battery_dead,no_charging)
                            for bat_flat in ['y', 'n']:
                                p_bat_flat = self.prob_tables['battery_flat'][(bat_dead, no_charging, bat_flat)]
                                
                                # Calculate P(car_will_start|no_oil,battery_flat) = P(-car_wont_start|...)
                                p_will_start = self.prob_tables['car_wont_start'][(no_oil, bat_flat, 'n')]  # Note the 'n' here
                                
                                # Multiply all probabilities along the path
                                path_prob = (p_alt * p_bat_age * p_no_oil * p_bat_dead * 
                                           p_no_charging * p_bat_flat * p_will_start)
                                
                                total_prob += path_prob
        
        return total_prob

    def print_detailed_calculation(self, fanbelt_state='y'):
        print(f"Calculating P(-car_won't_start|fanbelt_broken={fanbelt_state})")
        print("(This is the probability that the car WILL start given a broken fan belt)")
        print("\nKey intermediate probabilities:")
        
        # P(no_charging|fanbelt)
        p_no_charging = (
            self.prob_tables['no_charging'][('y', fanbelt_state, 'y')] * 
            self.prob_tables['alternator_broken']['y'] +
            self.prob_tables['no_charging'][('n', fanbelt_state, 'y')] * 
            self.prob_tables['alternator_broken']['n']
        )
        print(f"P(no_charging|fanbelt={fanbelt_state}) = {p_no_charging:.4f}")
        
        # Calculate both probabilities
        p_wont_start = 1 - self.calculate_p_car_will_start_given_fanbelt(fanbelt_state)
        p_will_start = self.calculate_p_car_will_start_given_fanbelt(fanbelt_state)
        
        print(f"\nP(car_won't_start|fanbelt={fanbelt_state}) = {p_wont_start:.4f}")
        print(f"P(-car_won't_start|fanbelt={fanbelt_state}) = {p_will_start:.4f}")
        return p_will_start

# Create network and calculate probability
network = CarBayesianNetwork()
probability = network.print_detailed_calculation('y')