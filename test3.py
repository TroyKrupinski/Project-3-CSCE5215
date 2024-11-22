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

    def calculate_p_car_wont_start_given_battery_age(self, battery_age_state='y'):
        total_prob = 0
        
        # Iterate through all possible states
        for alt in ['y', 'n']:
            for fan in ['y', 'n']:
                for no_oil in ['y', 'n']:
                    # Calculate probabilities for each path
                    p_alt = self.prob_tables['alternator_broken'][alt]
                    p_fan = self.prob_tables['fanbelt_broken'][fan]
                    p_no_oil = self.prob_tables['no_oil'][no_oil]
                    
                    # Calculate P(battery_dead|battery_age)
                    for bat_dead in ['y', 'n']:
                        p_bat_dead = self.prob_tables['battery_dead'][(bat_dead, battery_age_state)]
                        
                        # Calculate P(no_charging|alternator,fanbelt)
                        for no_charging in ['y', 'n']:
                            p_no_charging = self.prob_tables['no_charging'][(alt, fan, no_charging)]
                            
                            # Calculate P(battery_flat|battery_dead,no_charging)
                            for bat_flat in ['y', 'n']:
                                p_bat_flat = self.prob_tables['battery_flat'][(bat_dead, no_charging, bat_flat)]
                                
                                # Calculate P(car_wont_start|no_oil,battery_flat)
                                p_wont_start = self.prob_tables['car_wont_start'][(no_oil, bat_flat, 'y')]
                                
                                # Multiply all probabilities along the path
                                path_prob = (p_alt * p_fan * p_no_oil * p_bat_dead * 
                                           p_no_charging * p_bat_flat * p_wont_start)
                                
                                total_prob += path_prob
        
        return total_prob

    def print_detailed_calculation(self, battery_age_state='y'):
        print(f"Calculating P(car_won't_start|battery_age={battery_age_state})")
        print("\nKey intermediate probabilities:")
        
        # P(battery_dead|battery_age)
        p_bat_dead_y = self.prob_tables['battery_dead'][('y', battery_age_state)]
        p_bat_dead_n = self.prob_tables['battery_dead'][('n', battery_age_state)]
        print(f"P(battery_dead=y|battery_age={battery_age_state}) = {p_bat_dead_y:.4f}")
        print(f"P(battery_dead=n|battery_age={battery_age_state}) = {p_bat_dead_n:.4f}")
        
        # Calculate final probability
        final_prob = self.calculate_p_car_wont_start_given_battery_age(battery_age_state)
        print(f"\nFinal P(car_won't_start|battery_age={battery_age_state}) = {final_prob:.4f}")
        return final_prob

# Create network and calculate probability
network = CarBayesianNetwork()
probability = network.print_detailed_calculation('y')