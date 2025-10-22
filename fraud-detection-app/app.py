# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from fpdf import FPDF
import base64
from datetime import datetime
import io

# Set page config
st.set_page_config(
    page_title="Fraud Detection Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .developer-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Fraud Detection Model Analysis Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} - Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
    
    def add_paragraph(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 8, text)
        self.ln(5)

def generate_pdf_report(metrics, feature_importance, dataset_info, model_info):
    pdf = FraudDetectionPDF()
    pdf.add_page()
    
    # Title and Introduction
    pdf.add_section_title("Executive Summary")
    pdf.add_paragraph(
        f"This report provides a comprehensive analysis of the Fraud Detection Model performance. "
        f"The model was trained on {dataset_info['total_samples']:,} transactions with a fraud rate of {dataset_info['fraud_rate']:.2%}. "
        f"The current model achieves an overall accuracy of {metrics['accuracy']:.2%} with a precision of {metrics['precision']:.2%} in detecting fraudulent transactions."
    )
    
    # Model Performance Metrics
    pdf.add_section_title("Model Performance Metrics")
    metrics_text = f"""
    Accuracy: {metrics['accuracy']:.2%}
    Precision: {metrics['precision']:.2%}
    Recall: {metrics['recall']:.2%}
    F1-Score: {metrics['f1']:.2%}
    ROC AUC: {metrics['roc_auc']:.2%}
    Average Precision: {metrics['avg_precision']:.2%}
    
    Confusion Matrix:
    - True Negatives: {metrics['tn']:,}
    - False Positives: {metrics['fp']:,}
    - False Negatives: {metrics['fn']:,}
    - True Positives: {metrics['tp']:,}
    """
    pdf.add_paragraph(metrics_text)
    
    # Feature Importance
    pdf.add_section_title("Top Feature Importances")
    feature_text = "The following features were most important for fraud detection:\n"
    for i, (feature, importance) in enumerate(feature_importance.items()):
        if i < 10:  # Top 10 features
            feature_text += f"{i+1}. {feature}: {importance:.4f}\n"
    pdf.add_paragraph(feature_text)
    
    # Model Information
    pdf.add_section_title("Model Configuration")
    pdf.add_paragraph(f"Model Type: {model_info['model_type']}")
    pdf.add_paragraph(f"Training Date: {model_info['training_date']}")
    pdf.add_paragraph(f"Number of Features: {model_info['num_features']}")
    pdf.add_paragraph(f"Cross-Validation Score: {model_info['cv_score']:.2%}")
    
    # Recommendations
    pdf.add_section_title("Recommendations")
    recommendations = """
    1. Monitor model performance weekly for concept drift
    2. Implement feedback loop for false positives/negatives
    3. Consider retraining with new fraud patterns quarterly
    4. Maintain feature monitoring for data quality issues
    5. Set up alerts for performance metric degradation
    """
    pdf.add_paragraph(recommendations)
    
    return pdf

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fraud Detection Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section", [
        "📊 Model Performance Overview",
        "📈 Detailed Analysis",
        "🔍 Feature Importance",
        "🎯 Live Predictions", 
        "📄 Report & Documentation"
    ])
    
    # Load model and data (you'll need to replace these with your actual data)
    try:
        # For demo purposes, creating sample data
        # Replace this with your actual model loading
        # model = joblib.load('fraud_model.pkl')
        # X_test = pd.read_csv('X_test.csv')
        # y_test = pd.read_csv('y_test.csv')
        
        # Sample data for demonstration
        np.random.seed(42)
        n_samples = 5000
        X_test = pd.DataFrame({
            'amount': np.random.exponential(100, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'distance_from_home': np.random.exponential(50, n_samples),
            'time_since_last_txn': np.random.exponential(3600, n_samples),
            'num_transactions_last_24h': np.random.poisson(5, n_samples),
            'amount_log': np.random.normal(4, 1, n_samples),
            'amount_to_avg_ratio': np.random.normal(1, 0.5, n_samples),
            'txn_velocity_1h': np.random.poisson(0.5, n_samples)
        })
        
        # Generate synthetic predictions
        y_test = np.random.binomial(1, 0.05, n_samples)
        y_pred = np.random.binomial(1, 0.05, n_samples)
        y_pred_proba = np.random.uniform(0, 1, n_samples)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'roc_auc': roc_auc, 'avg_precision': avg_precision,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
        # Feature importance (sample)
        feature_importance = {
            'amount_to_avg_ratio': 0.15,
            'distance_from_home': 0.12,
            'txn_velocity_1h': 0.11,
            'time_since_last_txn': 0.10,
            'amount_log': 0.09,
            'hour': 0.08,
            'num_transactions_last_24h': 0.07,
            'amount': 0.06
        }
        
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        return

    if page == "📊 Model Performance Overview":
        st.header("Model Performance Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Primary Metric")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}", "Fraud Detection Accuracy")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}", "Fraud Capture Rate")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.2%}", "Balance Metric")
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}", "Overall Performance")
        with col6:
            st.metric("Avg Precision", f"{metrics['avg_precision']:.2%}", "Fraud Focus")
        with col7:
            st.metric("True Positives", f"{metrics['tp']:,}", "Frauds Caught")
        with col8:
            st.metric("False Positives", f"{metrics['fp']:,}", "False Alarms")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix Heatmap')
            st.pyplot(fig)
            
            # Performance Interpretation
            st.info(f"""
            **Performance Analysis:**
            - The model correctly identifies {metrics['tp']} out of {metrics['tp'] + metrics['fn']} fraudulent transactions
            - {metrics['fp']} legitimate transactions are incorrectly flagged as fraud
            - Overall fraud detection rate: {metrics['recall']:.2%}
            - Precision in fraud alerts: {metrics['precision']:.2%}
            """)
        
        with col_right:
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash', color='red')))
            fig_roc.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=500,
                height=400
            )
            st.plotly_chart(fig_roc)
            
            # Precision-Recall Curve
            st.subheader("Precision-Recall Curve")
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', name=f'PR Curve (AP = {avg_precision:.3f})', line=dict(width=3)))
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=500,
                height=400
            )
            st.plotly_chart(fig_pr)

    elif page == "📈 Detailed Analysis":
        st.header("Detailed Model Analysis")
        
        # Data Distribution
        st.subheader("Transaction Amount Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(X_test, x='amount', nbins=50, 
                             title='Distribution of Transaction Amounts',
                             labels={'amount': 'Transaction Amount'})
            st.plotly_chart(fig)
        
        with col2:
            # Fraud by hour
            fraud_by_hour = pd.DataFrame({
                'hour': X_test['hour'],
                'is_fraud': y_test
            }).groupby('hour')['is_fraud'].mean().reset_index()
            
            fig = px.line(fraud_by_hour, x='hour', y='is_fraud',
                         title='Fraud Rate by Hour of Day',
                         labels={'is_fraud': 'Fraud Rate', 'hour': 'Hour'})
            st.plotly_chart(fig)
        
        # Performance by segments
        st.subheader("Performance Across Segments")
        
        # Create segments based on transaction amount
        X_test_segmented = X_test.copy()
        X_test_segmented['is_fraud'] = y_test
        X_test_segmented['predicted_fraud'] = y_pred
        X_test_segmented['amount_segment'] = pd.cut(X_test_segmented['amount'], 
                                                   bins=[0, 50, 200, 500, np.inf],
                                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        segment_performance = X_test_segmented.groupby('amount_segment').apply(
            lambda x: pd.Series({
                'fraud_rate': x['is_fraud'].mean(),
                'detection_rate': (x['is_fraud'] & x['predicted_fraud']).sum() / max(1, x['is_fraud'].sum()),
                'false_positive_rate': ((~x['is_fraud']) & x['predicted_fraud']).sum() / max(1, (~x['is_fraud']).sum())
            })
        ).reset_index()
        
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=('Fraud Rate by Amount', 'Detection Rate', 'False Positive Rate'))
        
        fig.add_trace(go.Bar(x=segment_performance['amount_segment'], 
                            y=segment_performance['fraud_rate'], name='Fraud Rate'), 1, 1)
        fig.add_trace(go.Bar(x=segment_performance['amount_segment'], 
                            y=segment_performance['detection_rate'], name='Detection Rate'), 1, 2)
        fig.add_trace(go.Bar(x=segment_performance['amount_segment'], 
                            y=segment_performance['false_positive_rate'], name='FP Rate'), 1, 3)
        
        fig.update_layout(height=400, title_text="Performance Across Transaction Amount Segments")
        st.plotly_chart(fig)
        
        # Detailed classification report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'))

    elif page == "🔍 Feature Importance":
        st.header("Feature Importance Analysis")
        
        # Feature importance chart
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title='Feature Importance for Fraud Detection',
                    labels={'importance': 'Importance Score', 'feature': 'Features'})
        st.plotly_chart(fig)
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            'amount_to_avg_ratio': "Ratio of transaction amount to user's average transaction amount",
            'distance_from_home': "Geographic distance from user's home location",
            'txn_velocity_1h': "Number of transactions in the last hour",
            'time_since_last_txn': "Time elapsed since last transaction",
            'amount_log': "Logarithm of transaction amount",
            'hour': "Hour of day when transaction occurred",
            'num_transactions_last_24h': "Transaction count in last 24 hours",
            'amount': "Original transaction amount"
        }
        
        for feature, description in feature_descriptions.items():
            with st.expander(f"📊 {feature}"):
                st.write(description)
                if feature in feature_importance:
                    st.write(f"**Importance Score:** {feature_importance[feature]:.4f}")

    elif page == "🎯 Live Predictions":
        st.header("Live Transaction Prediction")
        
        # Input form for new transaction
        with st.form("prediction_form"):
            st.subheader("Enter Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
                hour = st.slider("Hour of Day", 0, 23, 12)
                distance = st.number_input("Distance from Home (km)", min_value=0.0, value=25.0)
                time_since_last = st.number_input("Time Since Last Transaction (seconds)", min_value=0, value=1800)
            
            with col2:
                txn_24h = st.number_input("Transactions in Last 24h", min_value=0, value=5)
                txn_1h = st.number_input("Transactions in Last 1h", min_value=0, value=1)
                amount_ratio = st.number_input("Amount to Avg Ratio", min_value=0.0, value=1.0)
                device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Physical"])
            
            submitted = st.form_submit_button("Predict Fraud Risk")
            
            if submitted:
                # Create feature vector (simplified)
                features = {
                    'amount': amount,
                    'hour': hour,
                    'distance_from_home': distance,
                    'time_since_last_txn': time_since_last,
                    'num_transactions_last_24h': txn_24h,
                    'txn_velocity_1h': txn_1h,
                    'amount_to_avg_ratio': amount_ratio,
                    'amount_log': np.log(amount + 1)
                }
                
                # Mock prediction (replace with actual model prediction)
                risk_score = min(0.95, (amount_ratio * 0.3 + (distance > 100) * 0.2 + (txn_1h > 3) * 0.2 + np.random.uniform(0, 0.3)))
                is_fraud = risk_score > 0.7
                
                # Display results
                st.subheader("Prediction Results")
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    if is_fraud:
                        st.error(f"🚨 HIGH FRAUD RISK: {risk_score:.1%}")
                    else:
                        st.success(f"✅ LOW FRAUD RISK: {risk_score:.1%}")
                
                with col_result2:
                    st.metric("Confidence Score", f"{risk_score:.1%}")
                
                # Risk factors
                st.subheader("Risk Factors Analysis")
                risk_factors = []
                if amount_ratio > 2:
                    risk_factors.append(f"Amount is {amount_ratio:.1f}x higher than user average")
                if distance > 100:
                    risk_factors.append(f"Transaction {distance}km from home location")
                if txn_1h > 3:
                    risk_factors.append(f"High transaction velocity: {txn_1h} in last hour")
                if hour in [0, 1, 2, 3]:
                    risk_factors.append("Unusual transaction time (early morning)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.info("✅ No significant risk factors detected")

    elif page == "📄 Report & Documentation":
        st.header("Report & Documentation")
        
        # Generate PDF Report
        st.subheader("Generate Analysis Report")
        
        dataset_info = {
            'total_samples': n_samples,
            'fraud_rate': y_test.mean(),
            'feature_count': X_test.shape[1]
        }
        
        model_info = {
            'model_type': 'Ensemble (XGBoost + Random Forest)',
            'training_date': '2024-01-15',
            'num_features': X_test.shape[1],
            'cv_score': 0.945
        }
        
        if st.button("📥 Generate Comprehensive PDF Report"):
            pdf = generate_pdf_report(metrics, feature_importance, dataset_info, model_info)
            
            # Save PDF to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_b64 = base64.b64encode(pdf_bytes).decode()
            
            # Download link
            href = f'<a href="data:application/octet-stream;base64,{pdf_b64}" download="fraud_detection_report.pdf">Download Full Report PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success("✅ PDF report generated successfully!")
        
        # Developer Information
        st.subheader("👨‍💻 Developer Information")
        
        # Replace these with your actual links
        linkedin_url = "YOUR_LINKEDIN_URL_HERE"
        github_url = "YOUR_GITHUB_URL_HERE"
        portfolio_url = "YOUR_PORTFOLIO_URL_HERE"
        
        col_dev1, col_dev2 = st.columns([1, 2])
        
        with col_dev1:
            st.image("https://cdn-icons-png.flaticon.com/512/174/174857.png", width=100)
        
        with col_dev2:
            st.markdown("""
            **Data Scientist & ML Engineer**  
            Specialized in Fraud Detection Systems  
            
            🔗 **Connect with me:**
            - [LinkedIn](""" + linkedin_url + """)
            - [GitHub](""" + github_url + """)  
            - [Portfolio](""" + portfolio_url + """)
            
            **Skills:** Machine Learning, Deep Learning, Fraud Analytics,  
            Model Deployment, Data Visualization, Cloud Computing
            """)
        
        # Model Documentation
        st.subheader("📚 Model Documentation")
        
        with st.expander("Model Architecture"):
            st.markdown("""
            **Ensemble Model Architecture:**
            - **XGBoost Classifier**: Primary model for non-linear patterns
            - **Random Forest**: Robust feature importance and stability
            - **Voting Classifier**: Combines predictions for better accuracy
            
            **Feature Engineering:**
            - Transaction velocity features
            - Behavioral pattern analysis
            - Geographic and temporal features
            - Amount-based risk scoring
            """)
        
        with st.expander("Deployment Architecture"):
            st.markdown("""
            **Production Deployment:**
            - **API**: FastAPI for real-time predictions
            - **Database**: PostgreSQL for transaction storage
            - **Monitoring**: MLflow for model performance tracking
            - **Scaling**: Docker containers with Kubernetes
            - **CI/CD**: Automated testing and deployment pipeline
            """)
        
        with st.expander("Performance Monitoring"):
            st.markdown("""
            **Key Monitoring Metrics:**
            - Model drift detection
            - Feature distribution changes
            - Prediction latency
            - False positive/negative rates
            - Business impact metrics
            
            **Alert Thresholds:**
            - Accuracy drop > 5%
            - Precision drop > 10%
            - Response time > 500ms
            - Feature correlation changes
            """)

if __name__ == "__main__":
    # Import required scikit-learn functions
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    
    main()
