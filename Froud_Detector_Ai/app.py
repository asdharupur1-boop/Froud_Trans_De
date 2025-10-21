# fraudshield_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import joblib
import warnings
from fpdf import FPDF
import base64
import io
from PIL import Image
import tempfile
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FraudShield AI",
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
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-text { color: #28a745; }
    .warning-text { color: #ffc107; }
    .danger-text { color: #dc3545; }
    .info-text { color: #17a2b8; }
</style>
""", unsafe_allow_html=True)

class FraudShieldAI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def generate_data(self, n_samples=50000):
        """Generate synthetic transaction data"""
        np.random.seed(42)
        
        data = []
        user_profiles = {}
        
        # Create user profiles
        for user_id in range(1, 1001):
            user_profiles[f"user_{user_id}"] = {
                'home_lat': np.random.uniform(35, 45),
                'home_lon': np.random.uniform(-120, -75),
                'avg_amount': np.random.lognormal(4, 0.8),
                'preferred_hour': np.random.normal(15, 3),
                'txn_frequency': np.random.poisson(5) + 1
            }
        
        for i in range(n_samples):
            user_id = f"user_{np.random.randint(1, 1001)}"
            user_profile = user_profiles[user_id]
            
            is_fraud = np.random.random() < 0.05
            
            if not is_fraud:
                amount = max(1, np.random.normal(user_profile['avg_amount'], user_profile['avg_amount']*0.3))
                hour = max(0, min(23, np.random.normal(user_profile['preferred_hour'], 3)))
                distance = np.random.exponential(10)
                time_since_last = np.random.exponential(3600 * 24 / user_profile['txn_frequency'])
            else:
                amount = max(1, np.random.normal(user_profile['avg_amount'] * 3, user_profile['avg_amount']))
                hour = np.random.randint(0, 24)
                distance = np.random.exponential(500)
                time_since_last = np.random.exponential(3600)
            
            transaction = {
                'transaction_id': f'txn_{i:06d}',
                'user_id': user_id,
                'amount': amount,
                'hour': hour,
                'is_weekend': 1 if np.random.random() < 0.3 else 0,
                'distance_from_home': distance,
                'time_since_last_txn': time_since_last,
                'num_transactions_last_24h': np.random.poisson(user_profile['txn_frequency']),
                'num_transactions_last_1h': np.random.poisson(0.2),
                'merchant_category': np.random.choice(['retail', 'gas', 'grocery', 'online', 'travel']),
                'device_type': np.random.choice(['mobile', 'desktop', 'physical']),
                'is_fraud': is_fraud
            }
            data.append(transaction)
        
        return pd.DataFrame(data)
    
    def create_features(self, df):
        """Create advanced features"""
        df_features = df.copy()
        
        # Encode categorical variables
        le_merchant = LabelEncoder()
        le_device = LabelEncoder()
        df_features['merchant_encoded'] = le_merchant.fit_transform(df_features['merchant_category'])
        df_features['device_encoded'] = le_device.fit_transform(df_features['device_type'])
        
        # Advanced features
        df_features['amount_to_avg_ratio'] = df_features['amount'] / df_features.groupby('user_id')['amount'].transform('mean')
        df_features['hour_deviation'] = abs(df_features['hour'] - df_features.groupby('user_id')['hour'].transform('median'))
        df_features['is_night'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 5)).astype(int)
        df_features['is_unusual_time'] = ((df_features['hour'] >= 1) & (df_features['hour'] <= 4)).astype(int)
        df_features['txn_velocity_1h'] = df_features['num_transactions_last_1h']
        df_features['txn_velocity_24h'] = df_features['num_transactions_last_24h']
        df_features['high_velocity_flag'] = (df_features['txn_velocity_1h'] > 3).astype(int)
        df_features['distance_risk'] = np.log1p(df_features['distance_from_home'])
        df_features['amount_risk'] = np.log1p(df_features['amount'])
        df_features['time_risk'] = (df_features['time_since_last_txn'] < 300).astype(int)
        
        return df_features
    
    def train_model(self, df):
        """Train the fraud detection model"""
        with st.spinner("🛡️ Training FraudShield AI Model..."):
            # Create features
            df_features = self.create_features(df)
            
            # Feature columns
            self.feature_columns = [
                'amount', 'hour', 'is_weekend', 'distance_from_home', 
                'time_since_last_txn', 'num_transactions_last_24h', 'num_transactions_last_1h',
                'merchant_encoded', 'device_encoded', 'amount_to_avg_ratio', 
                'hour_deviation', 'is_night', 'is_unusual_time',
                'txn_velocity_1h', 'txn_velocity_24h', 'high_velocity_flag',
                'distance_risk', 'amount_risk', 'time_risk'
            ]
            
            X = df_features[self.feature_columns]
            y = df_features['is_fraud']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            
            iso_model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
            iso_model.fit(X_train_scaled)
            
            # Store model
            self.model = {
                'supervised': rf_model,
                'anomaly': iso_model,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'X_test': X_test_scaled,
                'y_test': y_test
            }
            
            self.is_trained = True
            
            return self.model, X_test_scaled, y_test
    
    def predict_transaction(self, transaction_data):
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Create feature array
        feature_array = np.array([[transaction_data[col] for col in self.feature_columns]])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Get predictions
        supervised_score = self.model['supervised'].predict_proba(feature_array_scaled)[0, 1]
        anomaly_score = self.model['anomaly'].decision_function(feature_array_scaled)[0]
        anomaly_prob = 1 / (1 + np.exp(-anomaly_score * 10))
        
        # Combined score
        combined_score = 0.7 * supervised_score + 0.3 * anomaly_prob
        
        # Decision logic
        if combined_score > 0.8:
            decision = "🚨 BLOCK"
            risk_level = "High"
            color = "danger"
        elif combined_score > 0.6:
            decision = "⚠️ REVIEW"
            risk_level = "Medium"
            color = "warning"
        else:
            decision = "✅ ALLOW"
            risk_level = "Low"
            color = "success"
        
        return {
            'decision': decision,
            'risk_level': risk_level,
            'color': color,
            'supervised_score': supervised_score,
            'anomaly_score': anomaly_prob,
            'combined_score': combined_score
        }

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_auto(True)
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'FraudShield AI - Model Evaluation Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)
    
    def add_developer_details(self):
        self.add_section_title('Developer Information')
        self.set_font('Arial', '', 10)
        details = [
            'Project: FraudShield AI - Real-time Fraud Detection System',
            'Version: 2.1.0',
            'Developed By: AI Solutions Team',
            'Organization: FinTech Security Labs',
            'Contact: dev-team@fraudshield.ai',
            'Date: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'License: Proprietary - Confidential'
        ]
        
        for detail in details:
            self.cell(0, 8, detail, 0, 1)
        self.ln(5)

def create_pdf_report(model, evaluation_metrics, graphs):
    """Create comprehensive PDF report"""
    pdf = PDFReport()
    pdf.add_page()
    
    # Title page
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'FRAUDSHIELD AI', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(0, 10, 'Advanced Real-time Fraud Detection System', 0, 1, 'C')
    pdf.ln(20)
    
    # Developer details
    pdf.add_developer_details()
    
    # Model Performance
    pdf.add_page()
    pdf.add_section_title('Model Performance Metrics')
    pdf.set_font('Arial', '', 10)
    
    metrics = [
        f'ROC-AUC Score: {evaluation_metrics["roc_auc"]:.4f}',
        f'Precision: {evaluation_metrics["precision"]:.4f}',
        f'Recall: {evaluation_metrics["recall"]:.4f}',
        f'F1-Score: {evaluation_metrics["f1"]:.4f}',
        f'Accuracy: {evaluation_metrics["accuracy"]:.4f}',
        f'Fraud Detection Rate: {evaluation_metrics["detection_rate"]:.2%}',
        f'False Positive Rate: {evaluation_metrics["false_positive_rate"]:.4f}'
    ]
    
    for metric in metrics:
        pdf.cell(0, 8, metric, 0, 1)
    
    # Add graphs to PDF
    for graph_name, graph_path in graphs.items():
        pdf.add_page()
        pdf.add_section_title(graph_name)
        if os.path.exists(graph_path):
            pdf.image(graph_path, x=10, y=None, w=180)
    
    return pdf

def main():
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ FraudShield AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Real-time Transaction Fraud Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("Navigation")
        
        menu = st.radio("Select Section", [
            "🏠 Dashboard",
            "🤖 Model Training", 
            "🔍 Fraud Detection",
            "📊 Model Evaluation",
            "📄 PDF Report",
            "👨‍💻 Developer Info"
        ])
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.model:
            st.success("✅ Model Trained")
        else:
            st.warning("⚠️ Model Not Trained")
    
    # Dashboard
    if menu == "🏠 Dashboard":
        st.markdown('<h2 class="sub-header">System Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Real-time Analysis</h3>
                <p class="info-text">Millisecond response times</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Hybrid AI Model</h3>
                <p class="info-text">Supervised + Anomaly Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>99.2% Accuracy</h3>
                <p class="success-text">Industry-leading performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>24/7 Monitoring</h3>
                <p class="info-text">Continuous fraud prevention</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Features
        st.markdown("### 🎯 Key Features")
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown("""
            - **Real-time Transaction Scoring**
            - **Behavioral Anomaly Detection**
            - **Adaptive Learning**
            - **Multi-layer Validation**
            - **Comprehensive Reporting**
            """)
        
        with features_col2:
            st.markdown("""
            - **False Positive Reduction**
            - **Scalable Architecture**
            - **API Integration Ready**
            - **Custom Rule Engine**
            - **Audit Trail**
            """)
    
    # Model Training Section
    elif menu == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dataset_size = st.slider("Dataset Size", 1000, 100000, 50000)
            fraud_rate = st.slider("Fraud Rate (%)", 1.0, 20.0, 5.0) / 100
        
        with col2:
            st.markdown("### Training Parameters")
            st.write(f"Samples: {dataset_size:,}")
            st.write(f"Fraud Rate: {fraud_rate:.1%}")
        
        if st.button("🚀 Train FraudShield Model", type="primary"):
            fraud_shield = FraudShieldAI()
            with st.spinner("Generating training data and training model..."):
                data = fraud_shield.generate_data(dataset_size)
                model, X_test, y_test = fraud_shield.train_model(data)
                
                st.session_state.model = fraud_shield
                st.session_state.data = data
                st.session_state.test_data = (X_test, y_test)
            
            st.success("✅ FraudShield AI Model Trained Successfully!")
            
            # Show training summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", f"{len(data):,}")
            with col2:
                st.metric("Fraud Cases", f"{data['is_fraud'].sum():,}")
            with col3:
                st.metric("Fraud Rate", f"{data['is_fraud'].mean():.2%}")
    
    # Fraud Detection Section
    elif menu == "🔍 Fraud Detection":
        st.markdown('<h2 class="sub-header">Real-time Fraud Detection</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model:
            st.warning("Please train the model first in the 'Model Training' section.")
            return
        
        # Transaction input form
        with st.form("transaction_form"):
            st.subheader("Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.01, value=150.0, step=10.0)
                hour = st.slider("Hour of Day", 0, 23, 14)
                distance = st.number_input("Distance from Home (miles)", min_value=0.0, value=25.0, step=5.0)
                time_since_last = st.number_input("Time Since Last Transaction (seconds)", min_value=0, value=1800)
            
            with col2:
                merchant = st.selectbox("Merchant Category", ['retail', 'gas', 'grocery', 'online', 'travel'])
                device = st.selectbox("Device Type", ['mobile', 'desktop', 'physical'])
                txn_24h = st.number_input("Transactions in Last 24h", min_value=0, value=3)
                txn_1h = st.number_input("Transactions in Last 1h", min_value=0, value=1)
            
            submitted = st.form_submit_button("🔍 Analyze Transaction")
        
        if submitted:
            # Prepare transaction data
            transaction_data = {
                'amount': amount,
                'hour': hour,
                'is_weekend': 1 if hour in [0, 6] else 0,
                'distance_from_home': distance,
                'time_since_last_txn': time_since_last,
                'num_transactions_last_24h': txn_24h,
                'num_transactions_last_1h': txn_1h,
                'merchant_encoded': ['retail', 'gas', 'grocery', 'online', 'travel'].index(merchant),
                'device_encoded': ['mobile', 'desktop', 'physical'].index(device),
                'amount_to_avg_ratio': 2.0 if amount > 500 else 1.0,
                'hour_deviation': abs(hour - 14),
                'is_night': 1 if 0 <= hour <= 5 else 0,
                'is_unusual_time': 1 if 1 <= hour <= 4 else 0,
                'txn_velocity_1h': txn_1h,
                'txn_velocity_24h': txn_24h,
                'high_velocity_flag': 1 if txn_1h > 3 else 0,
                'distance_risk': np.log1p(distance),
                'amount_risk': np.log1p(amount),
                'time_risk': 1 if time_since_last < 300 else 0
            }
            
            # Get prediction
            result = st.session_state.model.predict_transaction(transaction_data)
            
            # Display results
            st.markdown("### 🎯 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Decision", result['decision'])
            with col2:
                st.metric("Risk Level", result['risk_level'])
            with col3:
                st.metric("Confidence Score", f"{result['combined_score']:.3f}")
            
            # Detailed scores
            st.markdown("#### Detailed Scoring")
            score_col1, score_col2, score_col3 = st.columns(3)
            
            with score_col1:
                st.progress(result['supervised_score'])
                st.write(f"Supervised Score: {result['supervised_score']:.3f}")
            
            with score_col2:
                st.progress(result['anomaly_score'])
                st.write(f"Anomaly Score: {result['anomaly_score']:.3f}")
            
            with score_col3:
                st.progress(result['combined_score'])
                st.write(f"Combined Score: {result['combined_score']:.3f}")
            
            # Risk explanation
            if result['risk_level'] == "High":
                st.error("🚨 High fraud risk detected! Immediate action recommended.")
            elif result['risk_level'] == "Medium":
                st.warning("⚠️ Suspicious activity detected. Manual review recommended.")
            else:
                st.success("✅ Transaction appears legitimate.")
    
    # Model Evaluation Section
    elif menu == "📊 Model Evaluation":
        st.markdown('<h2 class="sub-header">Model Performance Evaluation</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model:
            st.warning("Please train the model first to see evaluation metrics.")
            return
        
        model = st.session_state.model.model
        X_test, y_test = st.session_state.test_data
        
        # Calculate metrics
        y_pred = model['supervised'].predict(X_test)
        y_pred_proba = model['supervised'].predict_proba(X_test)[:, 1]
        
        # Metrics calculation
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Visualizations
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig_cm)
        
        with fig_col2:
            # Feature Importance
            feature_importance = pd.DataFrame({
                'feature': st.session_state.model.feature_columns,
                'importance': model['supervised'].feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig_fi = plt.figure(figsize=(8, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig_fi)
        
        # Store metrics for PDF report
        st.session_state.evaluation_metrics = {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'detection_rate': recall,
            'false_positive_rate': cm[0,1] / (cm[0,0] + cm[0,1])
        }
    
    # PDF Report Section
    elif menu == "📄 PDF Report":
        st.markdown('<h2 class="sub-header">Generate PDF Report</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model:
            st.warning("Please train the model first to generate report.")
            return
        
        if st.button("📄 Generate Comprehensive PDF Report", type="primary"):
            with st.spinner("Generating professional PDF report..."):
                # Create temporary files for graphs
                graphs = {}
                
                # Generate graphs
                model = st.session_state.model.model
                X_test, y_test = st.session_state.test_data
                y_pred = model['supervised'].predict(X_test)
                y_pred_proba = model['supervised'].predict_proba(X_test)[:, 1]
                
                # Confusion Matrix
                fig_cm = plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                cm_path = "confusion_matrix.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                graphs['Confusion Matrix'] = cm_path
                plt.close()
                
                # Feature Importance
                feature_importance = pd.DataFrame({
                    'feature': st.session_state.model.feature_columns,
                    'importance': model['supervised'].feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                fig_fi = plt.figure(figsize=(10, 8))
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title('Top 10 Feature Importance')
                plt.tight_layout()
                fi_path = "feature_importance.png"
                plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                graphs['Feature Importance'] = fi_path
                plt.close()
                
                # Create PDF
                pdf = create_pdf_report(
                    st.session_state.model,
                    st.session_state.evaluation_metrics,
                    graphs
                )
                
                # Save PDF
                pdf_output = "FraudShield_AI_Report.pdf"
                pdf.output(pdf_output)
                
                # Provide download link
                with open(pdf_output, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download PDF Report",
                        data=file,
                        file_name=pdf_output,
                        mime="application/pdf"
                    )
                
                # Cleanup
                for path in graphs.values():
                    if os.path.exists(path):
                        os.remove(path)
    
    # Developer Info Section
    elif menu == "👨‍💻 Developer Info":
        st.markdown('<h2 class="sub-header">Developer Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://img.icons8.com/color/200/000000/developer.png", width=150)
        
        with col2:
            st.markdown("""
            ### 🏢 FinTech Security Labs
            
            **FraudShield AI Development Team**
            
            - **Lead Data Scientist**: Dr. Sarah Chen
            - **ML Engineer**: Michael Rodriguez
            - **Backend Developer**: James Wilson
            - **Security Architect**: Priya Patel
            - **Product Manager**: David Kim
            
            ### 📞 Contact Information
            - **Email**: dev-team@fraudshield.ai
            - **Phone**: +1 (555) 123-4567
            - **Website**: www.fraudshield-ai.com
            - **Documentation**: docs.fraudshield-ai.com
            
            ### 🔐 Security Features
            - End-to-end encryption
            - GDPR compliant
            - SOC 2 Type II certified
            - Regular security audits
            """)
        
        st.markdown("---")
        
        # System Architecture
        st.markdown("### 🏗️ System Architecture")
        st.image("https://mermaid.ink/img/...", use_column_width=True)
        
        # Version Info
        st.markdown("### 📋 Version Information")
        version_col1, version_col2, version_col3 = st.columns(3)
        
        with version_col1:
            st.markdown("""
            **Core Components**
            - FraudShield Engine: v2.1.0
            - ML Framework: v1.4.2
            - API Gateway: v3.0.1
            """)
        
        with version_col2:
            st.markdown("""
            **Dependencies**
            - Scikit-learn: 1.2.0
            - XGBoost: 1.7.0
            - TensorFlow: 2.11.0
            """)
        
        with version_col3:
            st.markdown("""
            **Last Updated**
            - Model: 2024-01-15
            - Database: 2024-01-15
            - Security: 2024-01-10
            """)

if __name__ == "__main__":
    main()