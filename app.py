"""
🦷 Oral Health AI - Premium Edition
A comprehensive AI-powered oral disease screening application
Version: 2.0.0
Author: Arihant

Features:
- 8-class oral disease detection using EfficientNetB0
- Real-time image analysis with confidence scores
- GradCAM heatmap visualization
- Comprehensive disease information database
- Risk assessment questionnaire
- Multi-language support (English + Hindi)
- Professional medical-grade UI/UX
- Mobile responsive design
"""

# ============================================
# IMPORTS AND CONFIGURATION
# ============================================
import os
import sys

# Set environment variables before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Import OpenCV for heatmap
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================
st.set_page_config(
    page_title="Oral Health AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ArihantKhaitan/oral-health-ai',
        'Report a bug': 'https://github.com/ArihantKhaitan/oral-health-ai/issues',
        'About': 'Oral Health AI - Early Detection Saves Lives'
    }
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'analysis_complete': False,
        'current_image': None,
        'prediction_result': None,
        'confidence_scores': None,
        'selected_class': None,
        'risk_factors': {
            'tobacco': False,
            'paan': False,
            'smoke': False,
            'alcohol': False
        },
        'language': 'en',
        'show_all_scores': False,
        'image_analyzed': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# CUSTOM CSS - PREMIUM UI DESIGN
# ============================================
def load_custom_css():
    """Load custom CSS for premium UI"""
    st.markdown("""
    <style>
        /* ===== GLOBAL STYLES ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        }
        
        /* Hide Streamlit defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* ===== MAIN HEADER ===== */
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .main-subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #a0aec0;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* ===== CARD COMPONENTS ===== */
        .premium-card {
            background: linear-gradient(145deg, #1e1e2f 0%, #252540 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .premium-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.2);
        }
        
        /* ===== STEP INDICATORS ===== */
        .step-container {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .step-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .step-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #e2e8f0;
            margin: 0;
        }
        
        /* ===== RESULT CARDS ===== */
        .result-card-danger {
            background: linear-gradient(145deg, #2d1f1f 0%, #3d2020 100%);
            border: 2px solid #ef4444;
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
        }
        
        .result-card-warning {
            background: linear-gradient(145deg, #2d2a1f 0%, #3d3520 100%);
            border: 2px solid #f59e0b;
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
        }
        
        .result-card-success {
            background: linear-gradient(145deg, #1f2d1f 0%, #203d20 100%);
            border: 2px solid #22c55e;
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
        }
        
        .result-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 15px;
        }
        
        .result-title-danger { color: #f87171; }
        .result-title-warning { color: #fbbf24; }
        .result-title-success { color: #4ade80; }
        
        /* ===== CONFIDENCE SCORE ===== */
        .confidence-box {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
        }
        
        .confidence-label {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .confidence-value {
            font-size: 3rem;
            font-weight: 800;
        }
        
        .conf-high { color: #f87171; }
        .conf-medium { color: #fbbf24; }
        .conf-low { color: #4ade80; }
        
        /* ===== INFO CARDS ===== */
        .info-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            height: 100%;
        }
        
        .info-card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .info-card-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .info-card-list li {
            color: #94a3b8;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.9rem;
        }
        
        .info-card-list li:last-child {
            border-bottom: none;
        }
        
        /* ===== RISK BADGES ===== */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
            color: white;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
            color: white;
        }
        
        /* ===== DISCLAIMER BOX ===== */
        .disclaimer-box {
            background: linear-gradient(145deg, #422006 0%, #451a03 100%);
            border: 2px solid #f59e0b;
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
        }
        
        .disclaimer-title {
            color: #fbbf24;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .disclaimer-text {
            color: #fcd34d;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        
        /* ===== SIDEBAR STYLES ===== */
        .sidebar-metric {
            background: linear-gradient(145deg, #1e1e2f 0%, #252540 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            text-align: center;
        }
        
        .sidebar-metric-value {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sidebar-metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }
        
        /* ===== CONDITION LIST ===== */
        .condition-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            color: #e2e8f0;
            font-size: 0.9rem;
        }
        
        .condition-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        
        .dot-red { background: #ef4444; }
        .dot-orange { background: #f59e0b; }
        .dot-green { background: #22c55e; }
        
        /* ===== BUTTON STYLES ===== */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        /* ===== IMAGE CONTAINER ===== */
        .image-container {
            background: rgba(0,0,0,0.2);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 10px;
            overflow: hidden;
        }
        
        /* ===== HEATMAP SECTION ===== */
        .heatmap-container {
            background: linear-gradient(145deg, #1e1e2f 0%, #252540 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            margin: 20px 0;
        }
        
        .heatmap-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .heatmap-description {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 20px;
        }
        
        /* ===== FIND DENTIST BUTTON ===== */
        .dentist-button {
            display: block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            text-decoration: none;
            padding: 18px 35px;
            border-radius: 15px;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .dentist-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
            color: white !important;
        }
        
        /* ===== TABS STYLING ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 10px 20px;
            color: #94a3b8;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* ===== CHECKBOX STYLING ===== */
        .risk-checkbox {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 8px 0;
        }
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        
        /* ===== URGENCY BADGE ===== */
        .urgency-badge {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            display: inline-block;
            margin-top: 10px;
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ============================================
# DISEASE INFORMATION DATABASE
# ============================================
DISEASE_DATABASE = {
    'Oral_Cancer': {
        'name': 'Oral Cancer',
        'name_hi': 'मुंह का कैंसर',
        'emoji': '🚨',
        'risk_level': 'high',
        'description': 'Oral cancer is a serious condition where malignant cells form in the tissues of the mouth or throat. Early detection significantly improves survival rates.',
        'description_hi': 'मुंह का कैंसर एक गंभीर स्थिति है जहां मुंह या गले के ऊतकों में घातक कोशिकाएं बनती हैं।',
        'symptoms': [
            'Persistent mouth sores that don\'t heal',
            'White or red patches in mouth',
            'Lump or thickening in cheek',
            'Difficulty swallowing or chewing',
            'Numbness in tongue or mouth',
            'Unexplained bleeding',
            'Chronic sore throat',
            'Jaw pain or stiffness'
        ],
        'causes': [
            'Tobacco use (smoking, chewing)',
            'Heavy alcohol consumption',
            'HPV infection',
            'Excessive sun exposure (lip cancer)',
            'Poor nutrition',
            'Weakened immune system',
            'Family history of cancer'
        ],
        'treatments': [
            'Surgical removal of tumor',
            'Radiation therapy',
            'Chemotherapy',
            'Targeted drug therapy',
            'Immunotherapy',
            'Reconstructive surgery'
        ],
        'urgency': 'CRITICAL - Seek immediate medical attention within 24-48 hours',
        'urgency_hi': 'गंभीर - 24-48 घंटों के भीतर तुरंत चिकित्सा सहायता लें'
    },
    'Ulcers': {
        'name': 'Mouth Ulcers',
        'name_hi': 'मुंह के छाले',
        'emoji': '⚠️',
        'risk_level': 'medium',
        'description': 'Mouth ulcers (canker sores) are painful sores that appear inside the mouth. Most heal within 1-2 weeks without treatment.',
        'description_hi': 'मुंह के छाले दर्दनाक घाव हैं जो मुंह के अंदर दिखाई देते हैं।',
        'symptoms': [
            'Painful round or oval sores',
            'White or yellow center with red border',
            'Burning sensation before appearing',
            'Difficulty eating spicy/acidic foods',
            'Swelling around the sore',
            'Tingling sensation'
        ],
        'causes': [
            'Stress and anxiety',
            'Minor mouth injuries',
            'Acidic or spicy foods',
            'Vitamin deficiencies (B12, iron, folate)',
            'Hormonal changes',
            'Food allergies',
            'Certain medications'
        ],
        'treatments': [
            'Antiseptic mouthwash',
            'Pain-relieving gels (Benzocaine)',
            'Saltwater rinse',
            'Avoid spicy/acidic foods',
            'Vitamin B12 supplements',
            'Corticosteroid ointments'
        ],
        'urgency': 'Monitor - See dentist if ulcer persists beyond 2 weeks',
        'urgency_hi': 'निगरानी करें - यदि 2 सप्ताह से अधिक रहे तो दंत चिकित्सक से मिलें'
    },
    'Gingivitis': {
        'name': 'Gingivitis',
        'name_hi': 'मसूड़ों की सूजन',
        'emoji': '⚠️',
        'risk_level': 'medium',
        'description': 'Gingivitis is inflammation of the gums caused by bacterial infection. If left untreated, it can progress to periodontitis and tooth loss.',
        'description_hi': 'मसूड़े की सूजन बैक्टीरिया के संक्रमण के कारण होने वाली मसूड़ों की सूजन है।',
        'symptoms': [
            'Red, swollen gums',
            'Bleeding while brushing or flossing',
            'Bad breath (halitosis)',
            'Receding gums',
            'Tender or painful gums',
            'Soft, puffy gums',
            'Dark red gum color'
        ],
        'causes': [
            'Poor oral hygiene',
            'Plaque and tartar buildup',
            'Smoking or tobacco use',
            'Diabetes',
            'Hormonal changes',
            'Certain medications',
            'Dry mouth'
        ],
        'treatments': [
            'Professional dental cleaning',
            'Improved brushing technique',
            'Daily flossing',
            'Antibacterial mouthwash',
            'Regular dental checkups',
            'Quit smoking'
        ],
        'urgency': 'Schedule dental visit within 1-2 weeks',
        'urgency_hi': '1-2 सप्ताह के भीतर दंत चिकित्सक से मिलें'
    },
    'Caries': {
        'name': 'Dental Caries (Cavities)',
        'name_hi': 'दांतों की सड़न',
        'emoji': '⚠️',
        'risk_level': 'medium',
        'description': 'Dental caries (cavities) are permanently damaged areas in teeth that develop into tiny holes. They are one of the most common health problems worldwide.',
        'description_hi': 'दंत क्षय (कैविटी) दांतों में स्थायी रूप से क्षतिग्रस्त क्षेत्र हैं।',
        'symptoms': [
            'Toothache or sensitivity',
            'Pain when eating sweet, hot, or cold foods',
            'Visible holes or pits in teeth',
            'Brown, black, or white staining',
            'Bad breath',
            'Pain when biting down'
        ],
        'causes': [
            'Frequent snacking on sugary foods',
            'Sugary drinks consumption',
            'Poor brushing habits',
            'Bacteria in mouth',
            'Dry mouth',
            'Lack of fluoride',
            'Eating disorders'
        ],
        'treatments': [
            'Dental fillings',
            'Dental crowns (for severe decay)',
            'Root canal treatment',
            'Fluoride treatments',
            'Tooth extraction (if necessary)',
            'Dental sealants'
        ],
        'urgency': 'Schedule dental appointment within 1-2 weeks',
        'urgency_hi': '1-2 सप्ताह के भीतर दंत चिकित्सक से मिलें'
    },
    'Calculus': {
        'name': 'Calculus (Tartar)',
        'name_hi': 'टार्टर',
        'emoji': '📋',
        'risk_level': 'low',
        'description': 'Calculus (tartar) is hardite plaque that has mineralized on teeth. It cannot be removed by regular brushing and requires professional cleaning.',
        'description_hi': 'कैलकुलस (टार्टर) कठोर पट्टिका है जो दांतों पर खनिज हो गई है।',
        'symptoms': [
            'Yellow or brown deposits on teeth',
            'Rough feeling on tooth surface',
            'Bad breath',
            'Gum irritation and inflammation',
            'Bleeding gums',
            'Teeth appear darker'
        ],
        'causes': [
            'Poor oral hygiene',
            'Not flossing regularly',
            'Smoking or tobacco use',
            'Dry mouth conditions',
            'Diet high in sugar and starch',
            'Irregular dental visits'
        ],
        'treatments': [
            'Professional scaling and cleaning',
            'Root planing',
            'Improved daily oral hygiene',
            'Electric toothbrush',
            'Regular dental cleanings',
            'Tartar-control toothpaste'
        ],
        'urgency': 'Schedule professional cleaning within 1 month',
        'urgency_hi': '1 महीने के भीतर पेशेवर सफाई करवाएं'
    },
    'Tooth Discoloration': {
        'name': 'Tooth Discoloration',
        'name_hi': 'दांतों का मलिनकिरण',
        'emoji': '📋',
        'risk_level': 'low',
        'description': 'Tooth discoloration refers to staining or color changes in teeth. It can be extrinsic (surface stains) or intrinsic (internal discoloration).',
        'description_hi': 'दांतों का मलिनकिरण दांतों में दाग या रंग परिवर्तन को संदर्भित करता है।',
        'symptoms': [
            'Yellow or brown teeth',
            'White spots on teeth',
            'Gray or dark colored teeth',
            'Uneven tooth coloring',
            'Stains between teeth'
        ],
        'causes': [
            'Coffee, tea, or red wine',
            'Tobacco use',
            'Poor dental hygiene',
            'Certain medications',
            'Aging',
            'Excessive fluoride (fluorosis)',
            'Dental trauma'
        ],
        'treatments': [
            'Professional teeth whitening',
            'Whitening toothpaste',
            'Dental veneers',
            'Dental bonding',
            'Better oral hygiene',
            'Avoiding staining foods/drinks'
        ],
        'urgency': 'Non-urgent - Cosmetic concern, consult dentist at convenience',
        'urgency_hi': 'गैर-जरूरी - सौंदर्य संबंधी चिंता'
    },
    'Hypodontia': {
        'name': 'Hypodontia',
        'name_hi': 'दांतों की कमी',
        'emoji': '📋',
        'risk_level': 'low',
        'description': 'Hypodontia is a condition where one or more teeth fail to develop. It can affect dental function and facial appearance.',
        'description_hi': 'हाइपोडोंटिया एक ऐसी स्थिति है जहां एक या अधिक दांत विकसित नहीं होते।',
        'symptoms': [
            'Visible gaps in teeth',
            'Difficulty chewing properly',
            'Speech difficulties',
            'Jawbone issues',
            'Misalignment of existing teeth',
            'Self-esteem concerns'
        ],
        'causes': [
            'Genetic factors',
            'Developmental abnormalities',
            'Trauma during development',
            'Radiation therapy',
            'Certain syndromes',
            'Environmental factors'
        ],
        'treatments': [
            'Dental implants',
            'Fixed bridges',
            'Removable partial dentures',
            'Orthodontic treatment',
            'Space maintainers',
            'Dental bonding'
        ],
        'urgency': 'Non-urgent - Consult dentist for treatment options',
        'urgency_hi': 'गैर-जरूरी - उपचार विकल्पों के लिए दंत चिकित्सक से परामर्श करें'
    },
    'Normal_Mouth': {
        'name': 'Healthy Mouth',
        'name_hi': 'स्वस्थ मुंह',
        'emoji': '✅',
        'risk_level': 'low',
        'description': 'Your oral health appears to be in good condition! Continue maintaining your current oral hygiene practices.',
        'description_hi': 'आपका मौखिक स्वास्थ्य अच्छी स्थिति में दिखाई देता है!',
        'symptoms': [
            'Pink and firm gums',
            'No bleeding when brushing',
            'Fresh breath',
            'Clean teeth without visible plaque',
            'No pain or sensitivity',
            'Properly aligned teeth'
        ],
        'causes': [],
        'treatments': [
            'Continue brushing twice daily',
            'Floss once daily',
            'Use fluoride toothpaste',
            'Regular dental checkups (every 6 months)',
            'Maintain balanced diet',
            'Limit sugary foods and drinks',
            'Stay hydrated'
        ],
        'urgency': 'Routine dental checkup every 6 months',
        'urgency_hi': 'हर 6 महीने में नियमित दंत जांच'
    }
}

# ============================================
# MODEL LOADING FUNCTIONS
# ============================================
@st.cache_resource
def load_model():
    """Load the trained TensorFlow model"""
    if not TF_AVAILABLE:
        return None
    
    model_path = 'model/oral_disease_model.h5'
    
    if not os.path.exists(model_path):
        return None
    
    try:
        # Try loading with compile=False first
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        # Fallback: recreate architecture and load weights
        try:
            from tensorflow.keras.applications import EfficientNetB0
            from tensorflow.keras import layers, Model
            
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(8, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=outputs)
            model.load_weights(model_path)
            
            return model
        except Exception as e2:
            return None

@st.cache_data
def load_class_names():
    """Load class names from JSON file"""
    json_path = 'model/class_names.json'
    
    default_classes = [
        'Calculus', 'Caries', 'Gingivitis', 'Hypodontia',
        'Normal_Mouth', 'Oral_Cancer', 'Tooth Discoloration', 'Ulcers'
    ]
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data.get('class_names', default_classes)
        except:
            return default_classes
    
    return default_classes

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_gradcam_heatmap(img_array, model, pred_index):
    """Generate GradCAM heatmap for visualization"""
    if not TF_AVAILABLE or model is None:
        return None
    
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_output)
        
        if grads is None:
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    except Exception as e:
        return None

def apply_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """Apply heatmap overlay on original image"""
    if heatmap is None or not CV2_AVAILABLE:
        return None
    
    try:
        # Resize original image
        img = np.array(original_image.resize((224, 224)))
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlay = np.uint8(heatmap_colored * alpha + img * (1 - alpha))
        
        return overlay
    
    except Exception as e:
        return None

# ============================================
# ANALYSIS FUNCTIONS
# ============================================
def analyze_image(image, model, class_names):
    """Analyze image and return predictions"""
    if model is None:
        return None, None
    
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    
    pred_idx = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_idx]
    confidence = float(predictions[0][pred_idx]) * 100
    
    return {
        'class': pred_class,
        'index': pred_idx,
        'confidence': confidence,
        'all_scores': {
            class_names[i]: float(predictions[0][i]) * 100 
            for i in range(len(class_names))
        }
    }, processed

def calculate_risk_score(risk_factors):
    """Calculate overall risk score from questionnaire"""
    score = sum([
        risk_factors.get('tobacco', False),
        risk_factors.get('paan', False),
        risk_factors.get('smoke', False),
        risk_factors.get('alcohol', False)
    ])
    return score

# ============================================
# UI COMPONENTS
# ============================================
def render_sidebar():
    """Render sidebar with info and settings"""
    with st.sidebar:
        # Language selector
        st.markdown("### 🌐 Language / भाषा")
        lang = st.selectbox(
            "Select Language",
            ["English", "हिंदी"],
            key="language_selector",
            label_visibility="collapsed"
        )
        st.session_state.language = 'en' if lang == "English" else 'hi'
        
        st.markdown("---")
        
        # Model metrics
        st.markdown("### 📊 Model Performance")
        
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">86.96%</div>
            <div class="sidebar-metric-label">Overall Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">91%</div>
            <div class="sidebar-metric-label">Cancer Detection</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">10,860</div>
            <div class="sidebar-metric-label">Training Images</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detectable conditions
        st.markdown("### 🎯 Detectable Conditions")
        
        conditions = [
            ("dot-red", "Oral Cancer"),
            ("dot-orange", "Mouth Ulcers"),
            ("dot-orange", "Gingivitis"),
            ("dot-orange", "Dental Caries"),
            ("dot-green", "Calculus"),
            ("dot-green", "Tooth Discoloration"),
            ("dot-green", "Hypodontia"),
            ("dot-green", "Normal/Healthy")
        ]
        
        for dot_class, name in conditions:
            st.markdown(f"""
            <div class="condition-item">
                <div class="condition-dot {dot_class}"></div>
                <span>{name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Links
        st.markdown("### 🔗 Links")
        st.markdown("[📂 GitHub Repository](https://github.com/ArihantKhaitan/oral-health-ai)")
        st.markdown("[🤗 Hugging Face Space](https://huggingface.co/spaces/Arihant2409/oral-health-ai)")

def render_header():
    """Render main header"""
    st.markdown('<h1 class="main-title">🦷 Oral Health AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">AI-Powered Oral Disease Screening • Early Detection Saves Lives</p>', unsafe_allow_html=True)

def render_risk_assessment():
    """Render risk assessment section"""
    st.markdown("""
    <div class="step-container">
        <div class="step-number">1</div>
        <h2 class="step-title">Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Answer these questions to assess your oral health risk factors:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tobacco = st.checkbox("🚬 Do you use tobacco or gutkha?", key="tobacco_check")
        paan = st.checkbox("🌿 Do you consume paan or betel?", key="paan_check")
    
    with col2:
        smoke = st.checkbox("🔥 Do you smoke?", key="smoke_check")
        alcohol = st.checkbox("🍺 Do you consume alcohol regularly?", key="alcohol_check")
    
    # Update session state
    st.session_state.risk_factors = {
        'tobacco': tobacco,
        'paan': paan,
        'smoke': smoke,
        'alcohol': alcohol
    }
    
    risk_score = calculate_risk_score(st.session_state.risk_factors)
    
    # Display risk level
    if risk_score >= 3:
        st.markdown("""
        <div class="risk-badge risk-high">
            🚨 HIGH RISK - {} of 4 risk factors identified
        </div>
        """.format(risk_score), unsafe_allow_html=True)
        st.error("⚠️ You have multiple risk factors for oral cancer. Regular screening is strongly recommended!")
    elif risk_score >= 1:
        st.markdown("""
        <div class="risk-badge risk-medium">
            ⚠️ MODERATE RISK - {} of 4 risk factors identified
        </div>
        """.format(risk_score), unsafe_allow_html=True)
        st.warning("You have some risk factors. Consider regular dental checkups.")
    else:
        st.markdown("""
        <div class="risk-badge risk-low">
            ✅ LOW RISK - No major risk factors identified
        </div>
        """, unsafe_allow_html=True)
        st.success("Great! No major risk factors. Maintain good oral hygiene!")
    
    return risk_score

def render_image_input():
    """Render image input section"""
    st.markdown("---")
    st.markdown("""
    <div class="step-container">
        <div class="step-number">2</div>
        <h2 class="step-title">Upload or Capture Image</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])
    
    image = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a clear image of your mouth or teeth",
            type=['jpg', 'jpeg', 'png'],
            key="file_uploader",
            help="Supported formats: JPG, JPEG, PNG. Max size: 200MB"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    with tab2:
        st.info("📸 Position your camera to capture a clear image of the affected area in your mouth.")
        camera_image = st.camera_input("Take a photo", key="camera_input")
        if camera_image is not None:
            image = Image.open(camera_image)
    
    return image

def render_results(prediction, original_image, processed_image, model, risk_score):
    """Render analysis results"""
    st.markdown("---")
    st.markdown("""
    <div class="step-container">
        <div class="step-number">3</div>
        <h2 class="step-title">Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    pred_class = prediction['class']
    confidence = prediction['confidence']
    disease_info = DISEASE_DATABASE.get(pred_class, DISEASE_DATABASE['Normal_Mouth'])
    
    # Determine risk level
    risk_level = disease_info['risk_level']
    if risk_score >= 2 and risk_level == 'medium':
        risk_level = 'high'
    
    # Layout: Image and Result Card
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📷 Your Image")
        st.image(original_image, use_column_width=True)
    
    with col2:
        # Result card
        if risk_level == 'high':
            card_class = 'result-card-danger'
            title_class = 'result-title-danger'
        elif risk_level == 'medium':
            card_class = 'result-card-warning'
            title_class = 'result-title-warning'
        else:
            card_class = 'result-card-success'
            title_class = 'result-title-success'
        
        st.markdown(f"""
        <div class="{card_class}">
            <div class="result-title {title_class}">
                {disease_info['emoji']} {disease_info['name']}
            </div>
            <div class="confidence-box">
                <div class="confidence-label">AI Confidence Score</div>
                <div class="confidence-value {'conf-high' if confidence > 85 else 'conf-medium' if confidence > 60 else 'conf-low'}">
                    {confidence:.1f}%
                </div>
            </div>
            <p style="color: #e2e8f0; margin-top: 15px;">
                {disease_info['description']}
            </p>
            <div class="urgency-badge">
                ⏰ {disease_info['urgency']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Information
    st.markdown("#### 📋 Detailed Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🔍 Symptoms</div>
            <ul class="info-card-list">
        """, unsafe_allow_html=True)
        for symptom in disease_info['symptoms'][:5]:
            st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">⚡ Common Causes</div>
            <ul class="info-card-list">
        """, unsafe_allow_html=True)
        for cause in disease_info['causes'][:5]:
            st.markdown(f"<li>{cause}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">💊 Treatment Options</div>
            <ul class="info-card-list">
        """, unsafe_allow_html=True)
        for treatment in disease_info['treatments'][:5]:
            st.markdown(f"<li>{treatment}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # All predictions expander
    with st.expander("📊 View All Prediction Scores"):
        sorted_scores = sorted(
            prediction['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, score in sorted_scores:
            st.progress(score / 100, text=f"{class_name}: {score:.1f}%")
    
    # GradCAM Heatmap
    st.markdown("""
    <div class="heatmap-container">
        <div class="heatmap-title">🔥 AI Attention Heatmap</div>
        <div class="heatmap-description">
            This visualization shows where the AI focused when making its prediction. 
            Red/yellow areas indicate high attention, blue areas indicate low attention.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    heatmap = create_gradcam_heatmap(processed_image, model, prediction['index'])
    
    if heatmap is not None:
        overlay = apply_heatmap_overlay(original_image, heatmap)
        
        if overlay is not None:
            hm_col1, hm_col2 = st.columns(2)
            
            with hm_col1:
                st.image(
                    original_image.resize((224, 224)),
                    caption="Original Image",
                    use_column_width=True
                )
            
            with hm_col2:
                st.image(
                    overlay,
                    caption="AI Focus Areas (Red = High Attention)",
                    use_column_width=True
                )
        else:
            st.info("Heatmap visualization could not be generated for this image.")
    else:
        st.info("Heatmap visualization is not available.")

def render_footer():
    """Render footer with dentist button and disclaimer"""
    st.markdown("---")
    
    # Find Dentist
    st.markdown("#### 🏥 Need Professional Help?")
    st.markdown("""
    <a href="https://www.google.com/maps/search/dentist+near+me" target="_blank" class="dentist-button">
        📍 Find Dentists Near You
    </a>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <div class="disclaimer-title">
            ⚠️ IMPORTANT MEDICAL DISCLAIMER
        </div>
        <div class="disclaimer-text">
            This AI tool is intended for <strong>SCREENING PURPOSES ONLY</strong> and should not be used as a substitute 
            for professional medical diagnosis, advice, or treatment. The AI model has an accuracy of approximately 87% 
            and may produce incorrect results.
            <br><br>
            <strong>Always consult a qualified healthcare professional</strong> (dentist, oral surgeon, or doctor) 
            for proper diagnosis and treatment of any oral health conditions. Do not delay seeking professional 
            medical attention based on results from this tool.
            <br><br>
            If you experience severe pain, bleeding, difficulty swallowing, or notice any persistent changes 
            in your mouth, <strong>seek immediate medical attention</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    
    # Check if model loaded
    if model is None:
        st.error("⚠️ Model could not be loaded. Please ensure the model file exists at 'model/oral_disease_model.h5'")
        st.info("Required files:\n- model/oral_disease_model.h5\n- model/class_names.json")
        return
    
    # Step 1: Risk Assessment
    risk_score = render_risk_assessment()
    
    # Step 2: Image Input
    image = render_image_input()
    
    # Step 3: Analysis (if image provided)
    if image is not None:
        # Perform analysis
        with st.spinner("🔍 Analyzing your image..."):
            prediction, processed_image = analyze_image(image, model, class_names)
        
        if prediction is not None:
            render_results(prediction, image, processed_image, model, risk_score)
        else:
            st.error("❌ Could not analyze the image. Please try again with a different image.")
    
    # Footer
    render_footer()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()