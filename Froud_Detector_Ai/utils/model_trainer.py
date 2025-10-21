import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_features(self, df):
        """Create advanced features for the model"""
        df_features = df.copy()
        
        # Encode categorical variables
        self.label_encoders['merchant'] = LabelEncoder()
        self.label_encoders['device'] = LabelEncoder()
        
        df_features['merchant_encoded'] = self.label_encoders['merchant'].fit_transform(df_features['merchant_category'])
        df_features['device_encoded'] = self.label_encoders['device'].fit_transform(df_features['device_type'])
        
        # Create advanced features
        user_avg_amount = df_features.groupby('user_id')['amount'].transform('mean')
        df_features['amount_to_avg_ratio'] = df_features['amount'] / user_avg_amount.replace(0, 1)
        
        user_median_hour = df_features.groupby('user_id')['hour'].transform('median')
        df_features['hour_deviation'] = abs(df_features['hour'] - user_median_hour)
        
        df_features['is_night'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 5)).astype(int)
        df_features['is_unusual_time'] = ((df_features['hour'] >= 1) & (df_features['hour'] <= 4)).astype(int)
        
        df_features['txn_velocity_1h'] = df_features['num_transactions_last_1h']
        df_features['txn_velocity_24h'] = df_features['num_transactions_last_24h']
        df_features['high_velocity_flag'] = (df_features['txn_velocity_1h'] > 3).astype(int)
        
        df_features['distance_risk'] = np.log1p(df_features['distance_from_home'])
        df_features['amount_risk'] = np.log1p(df_features['amount'])
        df_features['time_risk'] = (df_features['time_since_last_txn'] < 300).astype(int)
        
        return df_features
    
    def train_model(self, data):
        """Train the fraud detection model"""
        # Create features
        df_features = self.create_features(data)
        
        # Define feature columns
        feature_columns = [
            'amount', 'hour', 'is_weekend', 'distance_from_home', 
            'time_since_last_txn', 'num_transactions_last_24h', 'num_transactions_last_1h',
            'merchant_encoded', 'device_encoded', 'amount_to_avg_ratio', 
            'hour_deviation', 'is_night', 'is_unusual_time',
            'txn_velocity_1h', 'txn_velocity_24h', 'high_velocity_flag',
            'distance_risk', 'amount_risk', 'time_risk'
        ]
        
        X = df_features[feature_columns]
        y = df_features['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train supervised model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train anomaly detection model
        iso_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        iso_model.fit(X_train_scaled)
        
        # Calculate metrics
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        # Create fraud detector
        fraud_detector = FraudDetector(rf_model, iso_model, self.scaler, feature_columns)
        
        return fraud_detector, metrics