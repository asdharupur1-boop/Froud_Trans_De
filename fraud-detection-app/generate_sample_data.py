import pandas as pd
import numpy as np

def generate_sample_transactions(n_samples=1000):
    """Generate sample transaction data for testing"""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'txn_{i:06d}' for i in range(n_samples)],
        'amount': np.random.lognormal(4, 1.5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'distance_from_home': np.random.exponential(50, n_samples),
        'time_since_last_txn': np.random.exponential(3600, n_samples),
        'num_transactions_last_24h': np.random.poisson(3, n_samples),
        'num_transactions_last_1h': np.random.poisson(0.5, n_samples),
        'merchant_category': np.random.choice(['retail', 'gas', 'grocery', 'online', 'travel'], n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'physical'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_transactions.csv', index=False)
    print(f"Generated {n_samples} sample transactions in 'sample_transactions.csv'")

if __name__ == "__main__":
    generate_sample_transactions(1000)