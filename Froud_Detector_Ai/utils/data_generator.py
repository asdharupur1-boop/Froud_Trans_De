import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def generate_transaction_data(self, n_samples=50000, fraud_rate=0.05):
        """Generate synthetic transaction data for training"""
        
        data = []
        user_profiles = {}
        
        # Create user profiles
        for user_id in range(1, 1001):
            user_profiles[f"user_{user_id}"] = {
                'home_lat': self.rng.uniform(35, 45),
                'home_lon': self.rng.uniform(-120, -75),
                'avg_amount': self.rng.lognormal(4, 0.8),
                'preferred_hour': self.rng.normal(15, 3),
                'txn_frequency': self.rng.poisson(5) + 1
            }
        
        for i in range(n_samples):
            user_id = f"user_{self.rng.integers(1, 1001)}"
            user_profile = user_profiles[user_id]
            
            is_fraud = self.rng.random() < fraud_rate
            
            if not is_fraud:
                # Normal transaction pattern
                amount = max(1, self.rng.normal(user_profile['avg_amount'], user_profile['avg_amount']*0.3))
                hour = max(0, min(23, int(self.rng.normal(user_profile['preferred_hour'], 3))))
                distance = self.rng.exponential(10)
                time_since_last = self.rng.exponential(3600 * 24 / user_profile['txn_frequency'])
            else:
                # Fraud transaction pattern
                amount = max(1, self.rng.normal(user_profile['avg_amount'] * 3, user_profile['avg_amount']))
                hour = self.rng.integers(0, 24)
                distance = self.rng.exponential(500)
                time_since_last = self.rng.exponential(3600)
            
            transaction = {
                'transaction_id': f'txn_{i:06d}',
                'user_id': user_id,
                'amount': float(amount),
                'hour': hour,
                'is_weekend': int(self.rng.random() < 0.3),
                'distance_from_home': float(distance),
                'time_since_last_txn': float(time_since_last),
                'num_transactions_last_24h': int(self.rng.poisson(user_profile['txn_frequency'])),
                'num_transactions_last_1h': int(self.rng.poisson(0.2)),
                'merchant_category': self.rng.choice(['retail', 'gas', 'grocery', 'online', 'travel']),
                'device_type': self.rng.choice(['mobile', 'desktop', 'physical']),
                'is_fraud': int(is_fraud)
            }
            data.append(transaction)
        
        return pd.DataFrame(data)