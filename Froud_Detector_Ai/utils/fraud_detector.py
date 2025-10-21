import numpy as np
import pandas as pd

class FraudDetector:
    def __init__(self, supervised_model, anomaly_model, scaler, feature_columns):
        self.supervised_model = supervised_model
        self.anomaly_model = anomaly_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.user_history = {}
    
    def prepare_features(self, transaction_data):
        """Prepare features for prediction from transaction data"""
        # This is a simplified version - in production, you'd have more sophisticated feature engineering
        
        # Mock feature preparation - replace with actual logic
        features = {
            'amount': transaction_data.get('amount', 0),
            'hour': transaction_data.get('hour', 12),
            'is_weekend': 1 if transaction_data.get('hour', 12) in [0, 6] else 0,
            'distance_from_home': transaction_data.get('distance', 10),
            'time_since_last_txn': transaction_data.get('time_since_last', 3600),
            'num_transactions_last_24h': transaction_data.get('recent_txns', 3),
            'num_transactions_last_1h': min(transaction_data.get('recent_txns', 3), 5),
            'merchant_encoded': ['retail', 'gas', 'grocery', 'online', 'travel'].index(
                transaction_data.get('merchant', 'retail')
            ) if transaction_data.get('merchant', 'retail') in ['retail', 'gas', 'grocery', 'online', 'travel'] else 0,
            'device_encoded': ['mobile', 'desktop', 'physical'].index(
                transaction_data.get('device', 'mobile')
            ) if transaction_data.get('device', 'mobile') in ['mobile', 'desktop', 'physical'] else 0,
            'amount_to_avg_ratio': 2.0 if transaction_data.get('amount', 0) > 500 else 1.0,
            'hour_deviation': abs(transaction_data.get('hour', 12) - 14),
            'is_night': 1 if 0 <= transaction_data.get('hour', 12) <= 5 else 0,
            'is_unusual_time': 1 if 1 <= transaction_data.get('hour', 12) <= 4 else 0,
            'txn_velocity_1h': min(transaction_data.get('recent_txns', 3), 5),
            'txn_velocity_24h': transaction_data.get('recent_txns', 3),
            'high_velocity_flag': 1 if transaction_data.get('recent_txns', 3) > 3 else 0,
            'distance_risk': np.log1p(transaction_data.get('distance', 10)),
            'amount_risk': np.log1p(transaction_data.get('amount', 0)),
            'time_risk': 1 if transaction_data.get('time_since_last', 3600) < 300 else 0
        }
        
        return features
    
    def predict(self, transaction_data):
        """Predict fraud probability for a transaction"""
        try:
            # Prepare features
            features_dict = self.prepare_features(transaction_data)
            
            # Convert to array in correct feature order
            feature_array = np.array([[features_dict[col] for col in self.feature_columns]])
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Get predictions from both models
            supervised_score = self.supervised_model.predict_proba(feature_array_scaled)[0, 1]
            anomaly_score = self.anomaly_model.decision_function(feature_array_scaled)[0]
            
            # Convert anomaly score to probability-like score
            anomaly_prob = 1 / (1 + np.exp(-anomaly_score * 10))
            
            # Combined score (weighted average)
            combined_score = 0.7 * supervised_score + 0.3 * anomaly_prob
            
            # Decision logic
            if combined_score > 0.8:
                decision = "🚨 BLOCK"
                risk_level = "High"
            elif combined_score > 0.6:
                decision = "⚠️ REVIEW"
                risk_level = "Medium"
            else:
                decision = "✅ ALLOW"
                risk_level = "Low"
            
            return {
                'decision': decision,
                'risk_level': risk_level,
                'supervised_score': float(supervised_score),
                'anomaly_score': float(anomaly_prob),
                'combined_score': float(combined_score)
            }
            
        except Exception as e:
            return {
                'decision': '❌ ERROR',
                'risk_level': 'Unknown',
                'supervised_score': 0.0,
                'anomaly_score': 0.0,
                'combined_score': 0.0,
                'error': str(e)
            }