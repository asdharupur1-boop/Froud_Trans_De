from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

class PDFReporter(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
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
            f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'License: Proprietary - Confidential'
        ]
        
        for detail in details:
            self.cell(0, 8, detail, 0, 1)
        self.ln(5)

class PDFReporter:
    def generate_report(self, metrics, report_type, include_charts=True, include_data=True, executive_summary=True):
        """Generate a comprehensive PDF report"""
        
        pdf = PDFReporter()
        pdf.add_page()
        
        # Title page
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 20, 'FRAUDSHIELD AI', 0, 1, 'C')
        pdf.set_font('Arial', 'I', 14)
        pdf.cell(0, 10, 'Advanced Real-time Fraud Detection System', 0, 1, 'C')
        pdf.ln(20)
        
        # Developer details
        pdf.add_developer_details()
        
        # Model Performance
        pdf.add_page()
        pdf.add_section_title('Model Performance Metrics')
        pdf.set_font('Arial', '', 10)
        
        if metrics:
            metric_text = [
                f'ROC-AUC Score: {metrics.get("roc_auc", 0):.4f}',
                f'Precision: {metrics.get("precision", 0):.4f}',
                f'Recall: {metrics.get("recall", 0):.4f}',
                f'F1-Score: {metrics.get("f1", 0):.4f}',
                f'Accuracy: {metrics.get("accuracy", 0):.4f}'
            ]
            
            for text in metric_text:
                pdf.cell(0, 8, text, 0, 1)
        
        # System Architecture
        pdf.add_page()
        pdf.add_section_title('System Architecture')
        pdf.set_font('Arial', '', 10)
        architecture_info = [
            'Component: Hybrid AI Fraud Detection',
            '- Supervised Learning: Random Forest Classifier',
            '- Anomaly Detection: Isolation Forest',
            '- Feature Engineering: 20+ behavioral features',
            '- Real-time Scoring: < 50ms response time',
            '',
            'Key Features:',
            '- Real-time transaction monitoring',
            '- Behavioral pattern analysis',
            '- Adaptive learning system',
            '- Multi-layer validation',
            '- Comprehensive reporting'
        ]
        
        for info in architecture_info:
            pdf.cell(0, 8, info, 0, 1)
        
        # Save to bytes buffer
        pdf_output = pdf.output(dest='S').encode('latin1')
        return pdf_output