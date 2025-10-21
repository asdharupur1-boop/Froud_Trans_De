import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO
import warnings
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
            self.artifacts = joblib.load('fraud_detection_model.pkl')
            self.model = self.artifacts['best_model']
            self.scaler = self.artifacts['scaler']
            self.label_encoders = self.artifacts['label_encoders']
            self.feature_columns = self.artifacts['feature_columns']
            self.categorical_columns = self.artifacts['categorical_columns']
            self.performance = self.artifacts['performance']
            st.success("✅ Fraud detection model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
    
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
            if col in self.label_encoders:
                try:
                    X[col] = self.label_encoders[col].transform([transaction_data.get(col, 'unknown')])[0]
                except ValueError:
                    X[col] = 0
        
        # Scale numerical features
        numerical_columns = [col for col in X.columns if col not in self.categorical_columns]
        X[numerical_columns] = self.scaler.transform(X[numerical_columns])
        
        return X
    
    def predict_fraud(self, transaction_data, threshold=0.5):
        """Predict fraud probability"""
        try:
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

def generate_sample_data():
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
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
    
    return pd.DataFrame(data)

def create_analytics_dashboard(df, predictions):
    """Create comprehensive analytics dashboard"""
    st.markdown("<div class='sub-header'>📊 Fraud Analytics Dashboard</div>", unsafe_allow_html=True)
    
    # Add predictions to dataframe
    df['fraud_probability'] = [p['fraud_probability'] for p in predictions]
    df['risk_level'] = [p['risk_level'] for p in predictions]
    df['is_fraud_predicted'] = [p['is_fraud'] for p in predictions]
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Risk Analysis", "📋 Transactions", "🎯 Model Insights"])
    
    with tab1:
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
        
        # Risk distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        risk_counts = df['risk_level'].value_counts()
        colors = ['#00cc96', '#ffa500', '#ff4b4b']  # Green, Orange, Red
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Transaction Risk Distribution')
        st.pyplot(fig1)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud probability distribution
            fig2 = px.histogram(df, x='fraud_probability', nbins=50, 
                              title='Fraud Probability Distribution',
                              color_discrete_sequence=['#1f77b4'])
            fig2.update_layout(xaxis_title='Fraud Probability', yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Risk by merchant category
            risk_by_category = df.groupby('merchant_category')['fraud_probability'].mean().sort_values(ascending=False)
            fig3 = px.bar(risk_by_category, x=risk_by_category.values, y=risk_by_category.index,
                         title='Average Fraud Risk by Merchant Category',
                         color=risk_by_category.values,
                         color_continuous_scale='RdYlGn_r')
            fig3.update_layout(xaxis_title='Average Fraud Probability', yaxis_title='Merchant Category')
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Display transactions table with risk scores
        st.dataframe(df[['transaction_id', 'amount', 'merchant_category', 'fraud_probability', 'risk_level', 'is_fraud_predicted']].head(20),
                   use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Performance")
            st.metric("AUC Score", f"{app.performance['auc']:.4f}")
            st.metric("Average Precision", f"{app.performance['average_precision']:.4f}")
            st.metric("Model Type", "XGBoost")
            st.metric("Training Date", app.artifacts['training_date'])
        
        with col2:
            st.markdown("#### Feature Importance")
            try:
                feature_importance = pd.read_csv('feature_importance.csv')
                top_features = feature_importance.head(10)
                fig4 = px.bar(top_features, x='importance', y='feature', 
                            title='Top 10 Most Important Features',
                            color='importance', 
                            color_continuous_scale='Viridis')
                fig4.update_layout(xaxis_title='Importance', yaxis_title='Feature')
                st.plotly_chart(fig4, use_container_width=True)
            except:
                st.info("Feature importance data not available")

def create_pdf_report(df, predictions):
    """Generate PDF report content"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import tempfile
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_path = tmpfile.name
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Fraud Detection Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary section
    summary_text = f"""
    This report provides a comprehensive analysis of {len(df)} transactions processed through the Advanced Fraud Detection System.
    The analysis includes risk assessment, fraud predictions, and detailed transaction insights.
    """
    summary = Paragraph(summary_text, styles['BodyText'])
    story.append(summary)
    story.append(Spacer(1, 12))
    
    # Key metrics table
    fraud_count = df['is_fraud_predicted'].sum()
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Transactions', f"{len(df):,}"],
        ['Predicted Frauds', f"{fraud_count:,}"],
        ['Fraud Rate', f"{(fraud_count/len(df)*100):.2f}%"],
        ['Average Risk Score', f"{(df['fraud_probability'].mean()*100):.2f}%"],
        ['High Risk Transactions', f"{(df['risk_level'] == 'HIGH').sum():,}"],
        ['Model AUC Score', f"{app.performance['auc']:.4f}"]
    ]
    
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))
    
    # Risk distribution
    risk_dist = df['risk_level'].value_counts()
    risk_data = [['Risk Level', 'Count', 'Percentage']]
    for level, count in risk_dist.items():
        percentage = (count / len(df)) * 100
        risk_data.append([level, str(count), f"{percentage:.1f}%"])
    
    risk_table = Table(risk_data)
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(risk_table)
    
    return pdf_path

def main():
    # Initialize app
    global app
    app = FraudDetectionApp()
    
    # Main header
    st.markdown("<div class='main-header'>🛡️ Advanced Fraud Detection System</div>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield--v1.png", width=80)
    st.sidebar.title("Navigation")
    
    menu_options = ["🏠 Dashboard", "🔍 Single Transaction", "📊 Batch Analysis", "📈 Analytics", "📄 Generate Report"]
    choice = st.sidebar.selectbox("Select Mode", menu_options)
    
    # Dashboard
    if choice == "🏠 Dashboard":
        st.markdown("<div class='sub-header'>System Overview</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Model AUC", f"{app.performance['auc']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Precision", f"{app.performance['average_precision']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Features Used", f"{len(app.feature_columns)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.info("💡 Use the navigation menu to analyze transactions and generate reports.")
    
    # Single Transaction Analysis
    elif choice == "🔍 Single Transaction":
        st.markdown("<div class='sub-header'>Single Transaction Analysis</div>", unsafe_allow_html=True)
        
        with st.form("single_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=1.0)
                hour = st.slider("Hour of Day", 0, 23, 12)
                is_weekend = st.selectbox("Weekend?", [("No", 0), ("Yes", 1)])[1]
                distance = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0, step=1.0)
            
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
                
                # Display results
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_color = "risk-high" if result['risk_level'] == 'HIGH' else "risk-medium" if result['risk_level'] == 'MEDIUM' else "risk-low"
                    st.markdown(f"<div class='{risk_color}'>Risk Level: {result['risk_level']}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
                
                with col3:
                    status = "🚨 FRAUD DETECTED" if result['is_fraud'] else "✅ LEGITIMATE"
                    st.markdown(f"**Status:** {status}")
    
    # Batch Analysis
    elif choice == "📊 Batch Analysis":
        st.markdown("<div class='sub-header'>Batch Transaction Analysis</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} transactions")
                
                # Process all transactions
                predictions = []
                for _, row in df.iterrows():
                    prediction = app.predict_fraud(row.to_dict())
                    predictions.append(prediction)
                
                # Display results
                create_analytics_dashboard(df, predictions)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            # Generate sample data for demonstration
            if st.button("Generate Sample Data"):
                df = generate_sample_data()
                predictions = []
                for _, row in df.iterrows():
                    prediction = app.predict_fraud(row.to_dict())
                    predictions.append(prediction)
                
                create_analytics_dashboard(df, predictions)
    
    # Analytics
    elif choice == "📈 Analytics":
        st.markdown("<div class='sub-header'>Advanced Analytics</div>", unsafe_allow_html=True)
        
        # Generate sample data for analytics
        df = generate_sample_data()
        predictions = []
        for _, row in df.iterrows():
            prediction = app.predict_fraud(row.to_dict())
            predictions.append(prediction)
        
        create_analytics_dashboard(df, predictions)
    
    # Report Generation
    elif choice == "📄 Generate Report":
        st.markdown("<div class='sub-header'>Generate PDF Report</div>", unsafe_allow_html=True)
        
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Generating report..."):
                # Generate sample data
                df = generate_sample_data()
                predictions = []
                for _, row in df.iterrows():
                    prediction = app.predict_fraud(row.to_dict())
                    predictions.append(prediction)
                
                # Add predictions to dataframe
                df['fraud_probability'] = [p['fraud_probability'] for p in predictions]
                df['risk_level'] = [p['risk_level'] for p in predictions]
                df['is_fraud_predicted'] = [p['is_fraud'] for p in predictions]
                
                # Create PDF
                pdf_path = create_pdf_report(df, predictions)
                
                # Provide download link
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()