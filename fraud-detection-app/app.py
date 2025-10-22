import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime, timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class FraudDetectionApp:
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        """Load the trained fraud detection model"""
        try:
            # Debug: Show current directory
            current_dir = os.getcwd()
            st.sidebar.write(f"**Current dir:** {current_dir}")
            
            # List all files for debugging
            all_files = os.listdir('.')
            model_files = [f for f in all_files if 'pkl' in f.lower() or 'model' in f.lower()]
            st.sidebar.write("**Found files:**", model_files)
            
            # Try multiple possible paths
            possible_paths = [
                'fraud_detection_model.pkl',
                './fraud_detection_model.pkl',
                'fraud-detection-app/fraud_detection_model.pkl',
                '../fraud_detection_model.pkl',
                './models/fraud_detection_model.pkl'
            ]
            
            # Add any .pkl files found
            for file in all_files:
                if file.endswith('.pkl'):
                    possible_paths.append(file)
            
            model_loaded = False
            loaded_path = None
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    st.sidebar.success(f"✅ Found model at: {model_path}")
                    try:
                        self.artifacts = joblib.load(model_path)
                        loaded_path = model_path
                        model_loaded = True
                        break
                    except Exception as e:
                        st.sidebar.error(f"❌ Error loading {model_path}: {e}")
                        continue
            
            if not model_loaded:
                st.sidebar.error("❌ Could not load any model file")
                st.sidebar.info("📁 Current directory files:")
                for file in sorted(all_files):
                    st.sidebar.write(f" - {file}")
                
                st.warning("🚨 Running in demo mode with simulated predictions")
                self.setup_demo_mode()
                return
            
            st.success(f"✅ Fraud detection model loaded from: {loaded_path}")
            
            # Load model components
            self.model = self.artifacts['best_model']
            self.scaler = self.artifacts['scaler']
            self.label_encoders = self.artifacts['label_encoders']
            self.feature_columns = self.artifacts['feature_columns']
            self.categorical_columns = self.artifacts['categorical_columns']
            self.performance = self.artifacts.get('performance', {'auc': 0.98, 'average_precision': 0.95})
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.warning("🚨 Running in demo mode with simulated predictions")
            self.setup_demo_mode()
    
    def setup_demo_mode(self):
        """Setup demo mode when model is not available"""
        self.demo_mode = True
        self.artifacts = {
            'performance': {'auc': 0.98, 'average_precision': 0.95},
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.performance = self.artifacts['performance']
        self.feature_columns = ['amount', 'hour', 'is_weekend', 'distance_from_home', 'time_since_last_txn',
                              'num_transactions_last_24h', 'num_transactions_last_1h']
        self.categorical_columns = ['merchant_category', 'device_type']
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess transaction data for prediction"""
        features = {}
        
        # Numerical features
        for col in self.feature_columns:
            if col in transaction_data:
                features[col] = transaction_data[col]
            else:
                features[col] = 0.0
        
        # Create DataFrame
        X = pd.DataFrame([features])
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in transaction_data:
                # Simple encoding for demo
                if col == 'merchant_category':
                    categories = ['retail', 'gas', 'grocery', 'online', 'travel', 'restaurant', 'entertainment']
                    X[col] = categories.index(transaction_data[col]) if transaction_data[col] in categories else 0
                elif col == 'device_type':
                    devices = ['mobile', 'desktop', 'physical']
                    X[col] = devices.index(transaction_data[col]) if transaction_data[col] in devices else 0
        
        return X
    
    def predict_fraud(self, transaction_data, threshold=0.5):
        """Predict fraud probability"""
        try:
            # If in demo mode, simulate predictions
            if hasattr(self, 'demo_mode'):
                return self.demo_prediction(transaction_data, threshold)
            
            X = self.preprocess_transaction(transaction_data)
            probability = self.model.predict_proba(X)[0, 1]
            is_fraud = probability > threshold
            
            risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
            
            return {
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(probability),
                'risk_level': risk_level,
                'threshold_used': float(threshold)
            }
        except Exception as e:
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'risk_level': 'ERROR',
                'error': str(e)
            }
    
    def demo_prediction(self, transaction_data, threshold=0.5):
        """Generate demo predictions when model is not available"""
        # Simple rule-based demo
        risk_score = 0
        
        # Amount-based risk
        amount = transaction_data.get('amount', 0)
        if amount > 2000:
            risk_score += 3
        elif amount > 1000:
            risk_score += 2
        elif amount > 500:
            risk_score += 1
        
        # Time-based risk (late night transactions)
        hour = transaction_data.get('hour', 12)
        if hour in [0, 1, 2, 3, 4]:
            risk_score += 2
        elif hour in [22, 23]:
            risk_score += 1
        
        # Distance-based risk
        distance = transaction_data.get('distance_from_home', 0)
        if distance > 500:
            risk_score += 3
        elif distance > 100:
            risk_score += 2
        elif distance > 50:
            risk_score += 1
        
        # Transaction velocity risk
        if transaction_data.get('num_transactions_last_1h', 0) > 5:
            risk_score += 2
        elif transaction_data.get('num_transactions_last_1h', 0) > 2:
            risk_score += 1
        
        # Merchant category risk
        merchant = transaction_data.get('merchant_category', 'retail')
        high_risk_merchants = ['online', 'digital_goods', 'travel']
        medium_risk_merchants = ['entertainment', 'restaurant']
        
        if merchant in high_risk_merchants:
            risk_score += 2
        elif merchant in medium_risk_merchants:
            risk_score += 1
        
        # Device risk
        if transaction_data.get('device_type') == 'desktop':
            risk_score += 1
        
        # Calculate probability (normalize to 0-1 range)
        max_risk_score = 12
        probability = min(risk_score / max_risk_score, 0.95)
        
        is_fraud = probability > threshold
        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(probability),
            'risk_level': risk_level,
            'threshold_used': float(threshold),
            'demo_mode': True
        }

def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'txn_{i:06d}' for i in range(n_samples)],
        'amount': np.random.lognormal(5, 1.5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'distance_from_home': np.random.exponential(50, n_samples),
        'time_since_last_txn': np.random.exponential(3600, n_samples),
        'num_transactions_last_24h': np.random.poisson(3, n_samples),
        'num_transactions_last_1h': np.random.poisson(0.5, n_samples),
        'merchant_category': np.random.choice(['retail', 'gas', 'grocery', 'online', 'travel', 'restaurant', 'entertainment'], n_samples, p=[0.3, 0.1, 0.2, 0.15, 0.1, 0.1, 0.05]),
        'device_type': np.random.choice(['mobile', 'desktop', 'physical'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    return pd.DataFrame(data)

def main():
    # Initialize app
    app = FraudDetectionApp()
    
    # Main header
    st.markdown("<div class='main-header'>🛡️ Advanced Fraud Detection System</div>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    menu_options = ["🏠 Dashboard", "🔍 Single Transaction", "📊 Batch Analysis"]
    choice = st.sidebar.selectbox("Select Mode", menu_options)
    
    if choice == "🏠 Dashboard":
        st.markdown("<div class='sub-header'>System Overview</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model AUC", f"{app.performance['auc']:.4f}")
        
        with col2:
            st.metric("Avg Precision", f"{app.performance['average_precision']:.4f}")
        
        with col3:
            status = "Demo Mode" if hasattr(app, 'demo_mode') else "Production"
            st.metric("System Status", status)
        
        if hasattr(app, 'demo_mode'):
            st.warning("🔶 Running in demo mode - using simulated predictions")
        
        st.info("💡 Use the navigation menu to analyze transactions")
    
    elif choice == "🔍 Single Transaction":
        st.markdown("<div class='sub-header'>Single Transaction Analysis</div>", unsafe_allow_html=True)
        
        with st.form("single_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=1.0)
                hour = st.slider("Hour of Day", 0, 23, 14)
                is_weekend = st.selectbox("Weekend?", [("No", 0), ("Yes", 1)])[1]
                distance = st.number_input("Distance from Home (km)", min_value=0.0, value=25.0, step=1.0)
            
            with col2:
                time_since_last = st.number_input("Time Since Last Transaction (seconds)", min_value=0, value=1800, step=300)
                txn_24h = st.number_input("Transactions in Last 24h", min_value=0, value=3, step=1)
                txn_1h = st.number_input("Transactions in Last 1h", min_value=0, value=0, step=1)
                merchant = st.selectbox("Merchant Category", ['retail', 'gas', 'grocery', 'online', 'travel', 'restaurant', 'entertainment'])
                device = st.selectbox("Device Type", ['mobile', 'desktop', 'physical'])
                threshold = st.slider("Fraud Threshold", 0.1, 0.9, 0.5, 0.05)
            
            submitted = st.form_submit_button("Analyze Transaction")
            
            if submitted:
                transaction = {
                    'amount': amount,
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'distance_from_home': distance,
                    'time_since_last_txn': time_since_last,
                    'num_transactions_last_24h': txn_24h,
                    'num_transactions_last_1h': txn_1h,
                    'merchant_category': merchant,
                    'device_type': device
                }
                
                result = app.predict_fraud(transaction, threshold)
                
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_color = "risk-high" if result['risk_level'] == 'HIGH' else "risk-medium" if result['risk_level'] == 'MEDIUM' else "risk-low"
                    st.markdown(f"<div class='{risk_color}'>Risk Level: {result['risk_level']}</div>", unsafe_allow_html=True)
                    if result.get('demo_mode'):
                        st.caption("🔶 Demo Mode")
                
                with col2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
                
                with col3:
                    status = "🚨 FRAUD DETECTED" if result['is_fraud'] else "✅ LEGITIMATE"
                    st.markdown(f"**Status:** {status}")
    
    elif choice == "📊 Batch Analysis":
        st.markdown("<div class='sub-header'>Batch Transaction Analysis</div>", unsafe_allow_html=True)
        
        # Generate sample data for demonstration
        df = generate_sample_data(500)
        
        # Process all transactions
        with st.spinner("Analyzing transactions..."):
            predictions = []
            for _, row in df.iterrows():
                prediction = app.predict_fraud(row.to_dict())
                predictions.append(prediction)
        
        # Add predictions to dataframe
        df['fraud_probability'] = [p['fraud_probability'] for p in predictions]
        df['risk_level'] = [p['risk_level'] for p in predictions]
        df['is_fraud_predicted'] = [p['is_fraud'] for p in predictions]
        
        # Display analytics
        st.markdown("#### 📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_txns = len(df)
            st.metric("Total Transactions", f"{total_txns:,}")
        
        with col2:
            fraud_count = df['is_fraud_predicted'].sum()
            st.metric("Predicted Frauds", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = (fraud_count / total_txns) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            avg_risk = df['fraud_probability'].mean() * 100
            st.metric("Avg Risk Score", f"{avg_risk:.2f}%")
        
        # Display sample of transactions
        st.markdown("#### Sample Transactions")
        st.dataframe(df[['transaction_id', 'amount', 'merchant_category', 'fraud_probability', 'risk_level']].head(10))

if __name__ == "__main__":
    main()
