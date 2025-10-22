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
import base64
from datetime import datetime, timedelta
import io
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Fraud Detection Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .developer-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e6ed;
    }
    .risk-high { background-color: #ff6b6b; color: white; }
    .risk-medium { background-color: #ffd93d; color: black; }
    .risk-low { background-color: #6bcf7f; color: white; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
    }
    .feature-importance-bar {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFraudAnalyzer:
    def __init__(self):
        self.metrics_history = []
    
    def generate_comprehensive_report(self, metrics, feature_importance, dataset_info, model_info, transaction_insights):
        """Generate an advanced comprehensive report"""
        report = f"""
🤖 ADVANCED FRAUD DETECTION ANALYTICS REPORT
{'=' * 60}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Report ID: FD-{datetime.now().strftime("%Y%m%d%H%M%S")}

📊 EXECUTIVE SUMMARY
{'=' * 60}
Overall Model Performance: {'EXCELLENT' if metrics['f1'] > 0.9 else 'GOOD' if metrics['f1'] > 0.8 else 'MODERATE'}
Business Impact: ${transaction_insights['potential_savings']:,.2f} potential monthly savings
Risk Coverage: {metrics['recall']:.1%} of fraudulent transactions detected

Dataset Overview:
• Total Transactions: {dataset_info['total_samples']:,}
• Fraud Rate: {dataset_info['fraud_rate']:.2%}
• Analysis Period: {dataset_info['time_period']}
• Data Quality Score: {dataset_info['data_quality']:.1%}

🎯 MODEL PERFORMANCE METRICS
{'=' * 60}
Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})
Precision:      {metrics['precision']:.3f} ({metrics['precision']:.1%})
Recall:         {metrics['recall']:.3f} ({metrics['recall']:.1%})
F1-Score:       {metrics['f1']:.3f} ({metrics['f1']:.1%})
ROC AUC:        {metrics['roc_auc']:.3f}
Avg Precision:  {metrics['avg_precision']:.3f}

Confusion Matrix Analysis:
• True Positives:  {metrics['tp']:,} (Frauds Correctly Identified)
• False Positives: {metrics['fp']:,} (False Alarms)
• True Negatives:  {metrics['tn']:,} (Legitimate Approved)
• False Negatives: {metrics['fn']:,} (Missed Frauds)

📈 FEATURE IMPORTANCE ANALYSIS
{'=' * 60}
"""
        
        # Feature importance with visual indicators
        for i, (feature, importance) in enumerate(feature_importance.items()):
            bar_width = int(importance * 100)
            report += f"{i+1:2d}. {feature:<25} {importance:.4f} [{'█' * (bar_width//2)}{' ' * (50 - bar_width//2)}]\n"
        
        report += f"""
🔧 MODEL CONFIGURATION
{'=' * 60}
Model Type: {model_info['model_type']}
Training Date: {model_info['training_date']}
Feature Count: {model_info['num_features']}
CV Score: {model_info['cv_score']:.3f}
Hyperparameters: {model_info.get('hyperparams', 'Optimized')}

💡 TRANSACTION INSIGHTS
{'=' * 60}
Peak Fraud Hours: {', '.join(map(str, transaction_insights['peak_fraud_hours']))}
High Risk Amount Range: ${transaction_insights['high_risk_amount_range'][0]:.2f} - ${transaction_insights['high_risk_amount_range'][1]:.2f}
Geographic Risk Zones: {transaction_insights['geo_risk_zones']}
Behavioral Patterns: {transaction_insights['behavioral_patterns']}

🚨 RISK ASSESSMENT & RECOMMENDATIONS
{'=' * 60}
"""
        
        recommendations = self._generate_recommendations(metrics, feature_importance)
        report += recommendations
        
        report += f"""
📞 SUPPORT & MAINTENANCE
{'=' * 60}
Next Model Review: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
Monitoring Frequency: Real-time + Daily batch
Alert Thresholds: Precision < 85% | Recall < 80%
Contact: ML Engineering Team

---
Generated by Advanced Fraud Analytics System v2.1
        """
        
        return report
    
    def _generate_recommendations(self, metrics, feature_importance):
        """Generate intelligent recommendations based on metrics"""
        recommendations = ""
        
        if metrics['precision'] < 0.85:
            recommendations += "🔴 PRIORITY: High false positive rate detected. Consider:\n"
            recommendations += "   • Adjusting classification threshold\n"
            recommendations += "   • Adding post-processing rules\n"
            recommendations += "   • Reviewing feature engineering\n\n"
        
        if metrics['recall'] < 0.80:
            recommendations += "🟡 WARNING: Fraud detection rate below target. Consider:\n"
            recommendations += "   • Additional feature engineering\n"
            recommendations += "   • Ensemble with anomaly detection\n"
            recommendations += "   • Reviewing recent fraud patterns\n\n"
        
        # Feature-specific recommendations
        top_features = list(feature_importance.keys())[:3]
        recommendations += f"🔄 FEATURE OPTIMIZATION:\n"
        recommendations += f"   • Focus on: {', '.join(top_features)}\n"
        recommendations += "   • Monitor feature drift weekly\n"
        recommendations += "   • Consider real-time feature updates\n\n"
        
        recommendations += "📊 PERFORMANCE OPTIMIZATION:\n"
        recommendations += "   • Implement model retraining pipeline\n"
        recommendations += "   • Set up A/B testing framework\n"
        recommendations += "   • Deploy shadow mode for new features\n"
        
        return recommendations

def create_sample_advanced_data():
    """Create sophisticated sample data for demonstration"""
    np.random.seed(42)
    n_samples = 10000
    
    # Generate realistic transaction data
    base_date = datetime.now() - timedelta(days=30)
    timestamps = [base_date + timedelta(hours=i) for i in range(n_samples)]
    
    X_test = pd.DataFrame({
        'timestamp': timestamps,
        'amount': np.random.exponential(150, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'distance_from_home': np.random.exponential(75, n_samples),
        'time_since_last_txn': np.random.exponential(3000, n_samples),
        'num_transactions_last_24h': np.random.poisson(8, n_samples),
        'num_transactions_last_1h': np.random.poisson(2, n_samples),
        'amount_log': np.random.normal(4.5, 1.2, n_samples),
        'amount_to_avg_ratio': np.random.normal(1.2, 0.8, n_samples),
        'txn_velocity_1h': np.random.poisson(1, n_samples),
        'is_weekend': np.random.binomial(1, 0.3, n_samples),
        'merchant_risk_score': np.random.uniform(0, 1, n_samples),
        'user_behavior_score': np.random.uniform(0.7, 1, n_samples)
    })
    
    # Generate more realistic fraud patterns
    fraud_probability = (
        (X_test['amount_to_avg_ratio'] > 2) * 0.4 +
        (X_test['distance_from_home'] > 200) * 0.3 +
        (X_test['txn_velocity_1h'] > 5) * 0.2 +
        (X_test['hour'].isin([0, 1, 2, 3])) * 0.1 +
        np.random.uniform(0, 0.2, n_samples)
    )
    
    y_test = (fraud_probability > 0.6).astype(int)
    y_pred_proba = np.clip(fraud_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return X_test, y_test, y_pred, y_pred_proba

def create_animated_metric(value, previous_value, title, delta_text):
    """Create animated metric cards"""
    delta = value - previous_value if previous_value else 0
    return st.metric(
        title,
        f"{value:.2%}",
        delta=f"{delta:+.2%}",
        delta_color="normal" if delta >= 0 else "inverse"
    )

def main():
    # Initialize analyzer
    analyzer = AdvancedFraudAnalyzer()
    
    # Header with animated elements
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header1:
        st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Analytics</h1>', unsafe_allow_html=True)
        st.markdown("### Real-time Transaction Monitoring & Risk Analysis")
    
    with col_header2:
        st.info("🔄 Live Monitoring Active")
    
    with col_header3:
        st.success(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Sidebar with advanced controls
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2173/2173475.png", width=80)
        st.title("Control Panel")
        
        # Model selection
        model_version = st.selectbox(
            "Model Version",
            ["Production v2.1", "Staging v2.2", "Experimental v3.0"],
            index=0
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Adjust model sensitivity"
        )
        
        # Date range filter
        date_range = st.date_input(
            "Analysis Period",
            value=(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
        )
        
        # Real-time updates
        real_time_updates = st.checkbox("Enable Real-time Updates", value=True)
        auto_refresh = st.checkbox("Auto-refresh Dashboard", value=False)
        
        if auto_refresh:
            st.warning("Auto-refresh every 30 seconds")
            time.sleep(30)
            st.experimental_rerun()
    
    # Load advanced sample data
    try:
        X_test, y_test, y_pred, y_pred_proba = create_sample_advanced_data()
        
        # Calculate advanced metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
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
        
        # Advanced feature importance
        feature_importance = {
            'amount_to_avg_ratio': 0.186,
            'merchant_risk_score': 0.152,
            'distance_from_home': 0.134,
            'txn_velocity_1h': 0.121,
            'time_since_last_txn': 0.098,
            'user_behavior_score': 0.087,
            'amount_log': 0.076,
            'hour': 0.065,
            'num_transactions_last_24h': 0.054,
            'amount': 0.043,
            'is_weekend': 0.032,
            'num_transactions_last_1h': 0.028
        }
        
    except Exception as e:
        st.error(f"🚨 Error loading data: {e}")
        return
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Dashboard", 
        "📈 Advanced Analytics",
        "🔍 Feature Intelligence", 
        "🎯 Live Predictions",
        "🚨 Alerts & Monitoring",
        "👨‍💻 Developer Hub"
    ])
    
    with tab1:
        st.header("🏢 Executive Dashboard")
        
        # KPI Overview
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_animated_metric(metrics['accuracy'], 0.92, "Accuracy", "vs Target")
        with col2:
            create_animated_metric(metrics['precision'], 0.88, "Precision", "Fraud Accuracy")
        with col3:
            create_animated_metric(metrics['recall'], 0.85, "Recall", "Detection Rate")
        with col4:
            create_animated_metric(metrics['f1'], 0.89, "F1-Score", "Balance Metric")
        
        # Business Impact Metrics
        st.subheader("💰 Business Impact")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            potential_savings = metrics['tp'] * 250  # Average fraud amount
            st.metric("Monthly Savings", f"${potential_savings:,.0f}", "From fraud prevention")
        with col6:
            false_positive_cost = metrics['fp'] * 5  # Cost per false alarm
            st.metric("FP Cost", f"${false_positive_cost:,.0f}", "Customer service impact")
        with col7:
            detection_rate = metrics['tp'] / (metrics['tp'] + metrics['fn'])
            st.metric("Detection Rate", f"{detection_rate:.1%}", "Frauds caught")
        with col8:
            efficiency_score = (metrics['precision'] + metrics['recall']) / 2
            st.metric("Efficiency Score", f"{efficiency_score:.1%}", "Overall performance")
        
        # Real-time Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Performance Trends")
            
            # Simulated trend data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            trend_data = pd.DataFrame({
                'date': dates,
                'accuracy': np.random.normal(0.94, 0.02, 30),
                'precision': np.random.normal(0.89, 0.03, 30),
                'recall': np.random.normal(0.86, 0.04, 30)
            })
            
            fig_trend = px.line(trend_data, x='date', y=['accuracy', 'precision', 'recall'],
                              title="Model Performance Trends (30 Days)",
                              labels={'value': 'Score', 'variable': 'Metric'})
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col_chart2:
            st.subheader("🕒 Fraud Distribution by Hour")
            
            fraud_by_hour = pd.DataFrame({
                'hour': X_test['hour'],
                'is_fraud': y_test
            }).groupby('hour')['is_fraud'].mean().reset_index()
            
            fig_hour = px.bar(fraud_by_hour, x='hour', y='is_fraud',
                            title="Fraud Rate by Hour of Day",
                            labels={'is_fraud': 'Fraud Rate'})
            fig_hour.update_traces(marker_color='#ff6b6b')
            st.plotly_chart(fig_hour, use_container_width=True)
    
    with tab2:
        st.header("📈 Advanced Analytics")
        
        col_analytics1, col_analytics2 = st.columns(2)
        
        with col_analytics1:
            # Advanced Confusion Matrix
            st.subheader("🎯 Enhanced Confusion Matrix")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Legit', 'Predicted Fraud'],
                y=['Actual Legit', 'Actual Fraud'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues'
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix with Proportions",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                width=500,
                height=400
            )
            st.plotly_chart(fig_cm)
            
            # Performance Interpretation
            st.info(f"""
            **📊 Performance Insights:**
            - **Detection Power**: {metrics['recall']:.1%} of frauds identified
            - **Alert Accuracy**: {metrics['precision']:.1%} of alerts are actual fraud
            - **Missed Opportunities**: {metrics['fn']} frauds missed
            - **False Alarms**: {metrics['fp']} legitimate transactions flagged
            """)
        
        with col_analytics2:
            # ROC and Precision-Recall Curves
            st.subheader("📊 Model Curves")
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            
            fig_curves = make_subplots(
                rows=1, cols=2,
                subplot_titles=('ROC Curve', 'Precision-Recall Curve'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_curves.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(width=3)),
                row=1, col=1
            )
            fig_curves.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='red')),
                row=1, col=1
            )
            
            fig_curves.add_trace(
                go.Scatter(x=recall_curve, y=precision_curve, mode='lines', 
                          name=f'PR (AP = {avg_precision:.3f})', line=dict(width=3)),
                row=1, col=2
            )
            
            fig_curves.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_curves, use_container_width=True)
        
        # Advanced Segmentation Analysis
        st.subheader("🎪 Transaction Segmentation Analysis")
        
        X_test_segmented = X_test.copy()
        X_test_segmented['is_fraud'] = y_test
        X_test_segmented['predicted_fraud'] = y_pred
        X_test_segmented['amount_segment'] = pd.cut(X_test_segmented['amount'], 
                                                   bins=[0, 50, 200, 500, 1000, np.inf],
                                                   labels=['Micro', 'Small', 'Medium', 'Large', 'XL'])
        
        segment_analysis = X_test_segmented.groupby('amount_segment').agg({
            'is_fraud': ['count', 'mean'],
            'predicted_fraud': 'mean'
        }).round(4)
        
        segment_analysis.columns = ['Total_Transactions', 'Actual_Fraud_Rate', 'Predicted_Fraud_Rate']
        segment_analysis['Detection_Rate'] = X_test_segmented.groupby('amount_segment').apply(
            lambda x: (x['is_fraud'] & x['predicted_fraud']).sum() / max(1, x['is_fraud'].sum())
        )
        
        st.dataframe(segment_analysis.style.format({
            'Actual_Fraud_Rate': '{:.2%}',
            'Predicted_Fraud_Rate': '{:.2%}',
            'Detection_Rate': '{:.2%}'
        }).background_gradient(cmap='YlOrRd'))
    
    with tab3:
        st.header("🔍 Feature Intelligence")
        
        # Interactive Feature Importance
        st.subheader("🎯 Feature Importance Analysis")
        
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=True)
        
        fig_features = px.bar(importance_df, x='importance', y='feature', orientation='h',
                             title='Feature Importance for Fraud Detection',
                             labels={'importance': 'Importance Score', 'feature': 'Features'},
                             color='importance',
                             color_continuous_scale='Viridis')
        
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Feature Details with Expandable Sections
        st.subheader("📚 Feature Documentation & Insights")
        
        feature_details = {
            'amount_to_avg_ratio': {
                'description': "Ratio of current transaction amount to user's 30-day average",
                'risk_threshold': "> 2.5x average amount",
                'business_impact': "High - Primary fraud indicator",
                'data_quality': "98% completeness"
            },
            'merchant_risk_score': {
                'description': "Dynamic risk score based on merchant's fraud history",
                'risk_threshold': "< 0.3 (high risk)",
                'business_impact': "High - Real-time updates",
                'data_quality': "95% completeness"
            },
            'distance_from_home': {
                'description': "Haversine distance from user's registered home location",
                'risk_threshold': "> 200 km",
                'business_impact': "Medium - Geographic anomalies",
                'data_quality': "92% completeness"
            }
        }
        
        for feature, details in feature_details.items():
            with st.expander(f"📊 {feature} (Importance: {feature_importance[feature]:.3f})", expanded=False):
                col_feat1, col_feat2 = st.columns(2)
                
                with col_feat1:
                    st.write(f"**Description**: {details['description']}")
                    st.write(f"**Risk Threshold**: {details['risk_threshold']}")
                
                with col_feat2:
                    st.write(f"**Business Impact**: {details['business_impact']}")
                    st.write(f"**Data Quality**: {details['data_quality']}")
                
                # Progress bar for importance
                importance_pct = feature_importance[feature] / max(feature_importance.values())
                st.progress(importance_pct)
                st.caption(f"Relative Importance: {importance_pct:.1%}")
    
    with tab4:
        st.header("🎯 Live Transaction Predictions")
        
        # Real-time prediction interface
        col_pred1, col_pred2 = st.columns([2, 1])
        
        with col_pred1:
            with st.form("advanced_prediction_form"):
                st.subheader("🔍 Transaction Risk Assessment")
                
                col_form1, col_form2, col_form3 = st.columns(3)
                
                with col_form1:
                    amount = st.number_input("💰 Amount ($)", min_value=0.0, value=150.0, step=10.0)
                    hour = st.slider("🕒 Hour", 0, 23, 14)
                    distance = st.number_input("📍 Distance (km)", min_value=0.0, value=45.0, step=5.0)
                
                with col_form2:
                    time_since_last = st.number_input("⏰ Time Since Last (min)", min_value=0, value=45)
                    txn_24h = st.number_input("📊 TXN Last 24h", min_value=0, value=8)
                    txn_1h = st.number_input("⚡ TXN Last 1h", min_value=0, value=2)
                
                with col_form3:
                    amount_ratio = st.number_input("📈 Amount Ratio", min_value=0.0, value=1.2, step=0.1)
                    merchant_score = st.slider("🏪 Merchant Score", 0.0, 1.0, 0.7)
                    user_score = st.slider("👤 User Score", 0.0, 1.0, 0.9)
                
                submitted = st.form_submit_button("🚀 Assess Fraud Risk", use_container_width=True)
                
                if submitted:
                    # Advanced risk calculation
                    risk_factors = []
                    risk_score = 0.0
                    
                    if amount_ratio > 2.0:
                        risk_factors.append(f"Amount {amount_ratio:.1f}x above average")
                        risk_score += 0.3
                    
                    if distance > 150:
                        risk_factors.append(f"Unusual location ({distance}km)")
                        risk_score += 0.2
                    
                    if txn_1h > 4:
                        risk_factors.append(f"High velocity ({txn_1h} TXN/hour)")
                        risk_score += 0.2
                    
                    if merchant_score < 0.4:
                        risk_factors.append("High-risk merchant")
                        risk_score += 0.15
                    
                    if hour in [0, 1, 2, 3]:
                        risk_factors.append("Unusual time pattern")
                        risk_score += 0.1
                    
                    # Add some randomness for demo
                    risk_score += np.random.uniform(0, 0.15)
                    risk_score = min(risk_score, 0.95)
                    
                    # Display results
                    st.subheader("🎯 Prediction Results")
                    
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    with col_result1:
                        if risk_score > 0.7:
                            st.error(f"🚨 HIGH RISK: {risk_score:.1%}")
                        elif risk_score > 0.4:
                            st.warning(f"⚠️ MEDIUM RISK: {risk_score:.1%}")
                        else:
                            st.success(f"✅ LOW RISK: {risk_score:.1%}")
                    
                    with col_result2:
                        st.metric("Confidence", f"{risk_score:.1%}")
                    
                    with col_result3:
                        st.metric("Recommended Action", 
                                 "BLOCK" if risk_score > 0.7 else "REVIEW" if risk_score > 0.4 else "APPROVE")
                    
                    # Risk factors breakdown
                    if risk_factors:
                        st.subheader("📋 Risk Factors Identified")
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.info("✅ No significant risk factors detected")
        
        with col_pred2:
            st.subheader("📊 Risk Distribution")
            
            # Risk distribution chart
            risk_bins = [0, 0.3, 0.7, 1.0]
            risk_labels = ['Low', 'Medium', 'High']
            risk_data = pd.DataFrame({
                'Risk Level': risk_labels,
                'Percentage': [60, 25, 15]  # Sample distribution
            })
            
            fig_risk_dist = px.pie(risk_data, values='Percentage', names='Risk Level',
                                 title="Transaction Risk Distribution",
                                 color='Risk Level',
                                 color_discrete_map={'Low': '#6bcf7f', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
            st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            # Quick stats
            st.metric("Avg Response Time", "120ms")
            st.metric("Daily Predictions", "45.2K")
            st.metric("System Uptime", "99.98%")
    
    with tab5:
        st.header("🚨 Alerts & Monitoring")
        
        # Real-time alerts
        st.subheader("🔔 Active Alerts")
        
        alerts = [
            {"type": "🚨", "message": "Unusual spike in transactions from Region A", "time": "2 min ago", "severity": "high"},
            {"type": "⚠️", "message": "Model confidence dropping for high-value transactions", "time": "15 min ago", "severity": "medium"},
            {"type": "ℹ️", "message": "Scheduled model retraining completed", "time": "1 hour ago", "severity": "low"},
        ]
        
        for alert in alerts:
            with st.container():
                col_alert1, col_alert2, col_alert3 = st.columns([1, 4, 2])
                with col_alert1:
                    st.write(alert["type"])
                with col_alert2:
                    st.write(alert["message"])
                with col_alert3:
                    st.caption(alert["time"])
                st.divider()
        
        # System monitoring
        st.subheader("🖥️ System Health")
        
        col_health1, col_health2, col_health3, col_health4 = st.columns(4)
        
        with col_health1:
            st.metric("CPU Usage", "42%", "-2%")
        with col_health2:
            st.metric("Memory", "67%", "+5%")
        with col_health3:
            st.metric("API Latency", "89ms", "-12ms")
        with col_health4:
            st.metric("Error Rate", "0.02%", "0.00%")
    
    with tab6:
        st.header("👨‍💻 Developer Hub")
        
        # Generate comprehensive report
        st.subheader("📄 Advanced Analytics Report")
        
        dataset_info = {
            'total_samples': len(X_test),
            'fraud_rate': y_test.mean(),
            'time_period': '30 days',
            'data_quality': 0.97
        }
        
        model_info = {
            'model_type': 'Ensemble (XGBoost + LightGBM + CatBoost)',
            'training_date': '2024-01-15',
            'num_features': len(feature_importance),
            'cv_score': 0.945,
            'hyperparams': 'Optimized via Optuna'
        }
        
        transaction_insights = {
            'peak_fraud_hours': [2, 3, 14],
            'high_risk_amount_range': (500, 2000),
            'geo_risk_zones': ['Region A', 'Region C'],
            'behavioral_patterns': 'Rapid succession, unusual locations',
            'potential_savings': potential_savings
        }
        
        if st.button("🔄 Generate Advanced Report", use_container_width=True):
            with st.spinner("Generating comprehensive analysis..."):
                time.sleep(2)  # Simulate processing
                
                report_text = analyzer.generate_comprehensive_report(
                    metrics, feature_importance, dataset_info, model_info, transaction_insights
                )
                
                # Display report
                with st.expander("📋 View Full Report", expanded=True):
                    st.text_area("Report Content", report_text, height=400, label_visibility="collapsed")
                
                # Download options
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📄 Download Text Report",
                        data=report_text,
                        file_name=f"fraud_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Could add PDF generation here with reportlab
                    st.info("📊 PDF export coming soon")
        
        # Developer Information
        st.subheader("👨‍💻 About the Developer")
        
        col_dev1, col_dev2 = st.columns([1, 2])
        
        with col_dev1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
        
        with col_dev2:
            st.markdown("""
            **🔗 Connect & Collaborate:**
            - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
            - [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/your-username)
            - [![Portfolio](https://img.shields.io/badge/Portfolio-FF7139?style=for-the-badge&logo=Firefox-Browser&logoColor=white)](https://your-portfolio.com)
            
            **🛠️ Technical Stack:**
            - Machine Learning & Deep Learning
            - Real-time Data Processing
            - Cloud Architecture (AWS/GCP/Azure)
            - MLOps & Model Deployment
            - Big Data Analytics
            
            **📈 Specializations:**
            - Fraud Detection Systems
            - Anomaly Detection
            - Risk Analytics
            - Real-time Decision Engines
            """)
        
        # API Documentation
        with st.expander("🔧 API Documentation"):
            st.markdown("""
            ```python
            # Example API Usage
            import requests
            
            payload = {
                "transaction_id": "txn_123456",
                "amount": 150.0,
                "user_id": "user_789",
                "merchant_id": "mcht_456",
                "location": {"lat": 40.7128, "lon": -74.0060},
                "timestamp": "2024-01-15T14:30:00Z"
            }
            
            response = requests.post(
                "https://api.your-fraud-system.com/v2/predict",
                json=payload,
                headers={"Authorization": "Bearer YOUR_API_KEY"}
            )
            
            risk_score = response.json()["risk_score"]
            recommendation = response.json()["action"]
            ```
            
            **Endpoints:**
            - `POST /v2/predict` - Real-time predictions
            - `GET /v2/health` - System status
            - `GET /v2/metrics` - Performance metrics
            - `POST /v2/feedback` - Model feedback
            """)

if __name__ == "__main__":
    main()
