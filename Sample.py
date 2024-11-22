import numpy as np

class ProbabilityVerification:
    def __init__(self):
        self.cpts = self.initialize_cpts()
        
    def initialize_cpts(self):
        """Initialize the same CPTs as before"""
        return {
            'Battery_age': {'y': 0.2, 'n': 0.8},
            'Alternator_broken': {'y': 0.1, 'n': 0.9},
            'Fanbelt_broken': {'y': 0.3, 'n': 0.7},
            'Battery_dead': {
                ('y', 'y'): 0.7, ('y', 'n'): 0.3,
                ('n', 'y'): 0.3, ('n', 'n'): 0.7
            },
            'No_charging': {
                ('y', 'y', 'y'): 0.75, ('y', 'n', 'y'): 0.4,
                ('n', 'y', 'y'): 0.6,  ('n', 'n', 'y'): 0.1,
                ('y', 'y', 'n'): 0.25, ('y', 'n', 'n'): 0.6,
                ('n', 'y', 'n'): 0.4,  ('n', 'n', 'n'): 0.9
            },
            'Battery_flat': {
                ('y', 'y', 'y'): 0.95, ('y', 'n', 'y'): 0.85,
                ('n', 'y', 'y'): 0.8,  ('n', 'n', 'y'): 0.1,
                ('y', 'y', 'n'): 0.05, ('y', 'n', 'n'): 0.15,
                ('n', 'y', 'n'): 0.2,  ('n', 'n', 'n'): 0.9
            },
            'car_wont_start': {
                ('y', 'y', 'n'): 0.9, ('y', 'n', 'n'): 0.9,
                ('n', 'y', 'n'): 0.9, ('n', 'n', 'n'): 0.1,
                ('y', 'y', 'y'): 0.1, ('y', 'n', 'y'): 0.1,
                ('n', 'y', 'y'): 0.1, ('n', 'n', 'y'): 0.9
            }
        }

    def calculate_complementary_probabilities(self, evidence_var, evidence_val):
        """Calculate both P(cws|evidence) and P(not_cws|evidence)"""
        total_samples = 50000
        cws_count = 0
        not_cws_count = 0
        
        for _ in range(total_samples):
            sample = {evidence_var: evidence_val}
            
            # Sample root nodes
            root_nodes = {
                'Battery_age': self.cpts['Battery_age']['y'],
                'Alternator_broken': self.cpts['Alternator_broken']['y'],
                'Fanbelt_broken': self.cpts['Fanbelt_broken']['y'],
            }
            
            for node, prob in root_nodes.items():
                if node not in sample:
                    sample[node] = np.random.choice(['y', 'n'], p=[prob, 1-prob])
            
            # Sample intermediate nodes
            bd_prob = self.cpts['Battery_dead'][(sample['Battery_age'], 'y')]
            sample['Battery_dead'] = np.random.choice(['y', 'n'], p=[bd_prob, 1-bd_prob])
            
            nc_prob = self.cpts['No_charging'][(sample['Alternator_broken'], 
                                              sample['Fanbelt_broken'], 'y')]
            sample['No_charging'] = np.random.choice(['y', 'n'], p=[nc_prob, 1-nc_prob])
            
            bf_prob = self.cpts['Battery_flat'][(sample['Battery_dead'], 
                                               sample['No_charging'], 'y')]
            sample['Battery_flat'] = np.random.choice(['y', 'n'], p=[bf_prob, 1-bf_prob])
            
            # Count outcomes
            if sample['Battery_flat'] == 'y':
                cws_count += 1
            else:
                not_cws_count += 1
        
        p_cws = cws_count / total_samples
        p_not_cws = not_cws_count / total_samples
        
        return p_cws, p_not_cws

def main():
    verifier = ProbabilityVerification()
    
    evidence_vars = [
        ('Fanbelt_broken', 'Fan belt broken'),
        ('Battery_age', 'Battery aged'),
        ('Alternator_broken', 'Alternator broken')
    ]
    
    print("\nComplementary Probability Verification:")
    print("======================================")
    
    for var, desc in evidence_vars:
        p_cws, p_not_cws = verifier.calculate_complementary_probabilities(var, 'y')
        total = p_cws + p_not_cws
        
        print(f"\nFor {desc}:")
        print(f"P(car won't start | {desc}) ≈ {p_cws:.4f}")
        print(f"P(car will start | {desc}) ≈ {p_not_cws:.4f}")
        print(f"Sum: {total:.4f} (should be approximately 1)")

if __name__ == "__main__":
    main()