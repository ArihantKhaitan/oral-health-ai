"""
🦷 Oral Health AI - Professional Edition v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A comprehensive AI-powered oral disease screening application

Author: Arihant Khaitan
Version: 3.0.0
License: MIT

Features:
━━━━━━━━━
✅ 8-class oral disease detection using EfficientNetB0
✅ Proper GradCAM heatmap visualization (red-yellow for attention)
✅ Full Hindi language support with translations
✅ Manual analyze button (not automatic)
✅ Camera on/off toggle
✅ Dashboard layout with navigation tabs
✅ Professional medical-grade UI
✅ Comprehensive disease information
✅ Risk assessment questionnaire
✅ Mobile responsive design

Classes Detected:
━━━━━━━━━━━━━━━━
1. Oral Cancer (High Risk)
2. Mouth Ulcers (Medium Risk)
3. Gingivitis (Medium Risk)
4. Dental Caries (Medium Risk)
5. Calculus/Tartar (Low Risk)
6. Tooth Discoloration (Low Risk)
7. Hypodontia (Low Risk)
8. Normal/Healthy (Low Risk)
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS AND ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import io
import base64
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Core imports
import streamlit as st
import numpy as np
from PIL import Image
import json

# TensorFlow import with error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not available. Please install tensorflow.")

# OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Matplotlib for heatmap
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

# SciPy for gaussian filter
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Oral Health AI - मौखिक स्वास्थ्य AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",  # Changed from "collapsed"
    menu_items={
        'Get Help': 'https://github.com/ArihantKhaitan/oral-health-ai',
        'Report a bug': 'https://github.com/ArihantKhaitan/oral-health-ai/issues',
        'About': """
        # Oral Health AI v3.0
        AI-powered oral disease screening tool.
        
        **Accuracy:** 86.96%
        **Classes:** 8 oral conditions
        **Training Data:** 10,860 images
        
        © 2024 Arihant Khaitan
        """
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables with default values"""
    
    # Language settings
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    # Navigation
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'home'
    
    # Image handling
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'camera_image' not in st.session_state:
        st.session_state.camera_image = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'image_source' not in st.session_state:
        st.session_state.image_source = None
    
    # Camera state
    if 'camera_enabled' not in st.session_state:
        st.session_state.camera_enabled = False
    
    # Analysis state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'heatmap_image' not in st.session_state:
        st.session_state.heatmap_image = None
    if 'processed_array' not in st.session_state:
        st.session_state.processed_array = None
    
    # Risk assessment
    if 'risk_tobacco' not in st.session_state:
        st.session_state.risk_tobacco = False
    if 'risk_paan' not in st.session_state:
        st.session_state.risk_paan = False
    if 'risk_smoke' not in st.session_state:
        st.session_state.risk_smoke = False
    if 'risk_alcohol' not in st.session_state:
        st.session_state.risk_alcohol = False
    
    # Analysis counter for unique keys
    if 'analysis_counter' not in st.session_state:
        st.session_state.analysis_counter = 0

# Initialize session state
initialize_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPREHENSIVE TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS = {
    'en': {
        # App title and navigation
        'app_title': '🦷 Oral Health AI',
        'app_subtitle': 'AI-Powered Oral Disease Screening • Early Detection Saves Lives',
        'nav_home': '🏠 Home',
        'nav_scan': '🔍 Scan',
        'nav_history': '📊 Results',
        'nav_info': 'ℹ️ About',
        
        # Risk Assessment
        'risk_title': 'Risk Assessment',
        'risk_subtitle': 'Answer these questions to assess your oral health risk factors',
        'risk_tobacco': 'Do you use tobacco or gutkha?',
        'risk_paan': 'Do you consume paan or betel?',
        'risk_smoke': 'Do you smoke?',
        'risk_alcohol': 'Do you consume alcohol regularly?',
        'risk_high': 'HIGH RISK',
        'risk_medium': 'MODERATE RISK',
        'risk_low': 'LOW RISK',
        'risk_high_msg': 'You have multiple risk factors for oral cancer. Regular screening is strongly recommended!',
        'risk_medium_msg': 'You have some risk factors. Consider regular dental checkups.',
        'risk_low_msg': 'Great! No major risk factors. Maintain good oral hygiene!',
        
        # Image Upload
        'upload_title': 'Upload or Capture Image',
        'upload_tab': '📁 Upload Image',
        'camera_tab': '📷 Camera',
        'upload_prompt': 'Upload a clear image of your mouth or teeth',
        'camera_enable': '📷 Enable Camera',
        'camera_disable': '❌ Disable Camera',
        'camera_prompt': 'Position your camera to capture a clear image of the affected area',
        'take_photo': 'Take a photo',
        'analyze_btn': '🔍 Analyze Image',
        'analyzing': 'Analyzing your image...',
        'clear_btn': '🗑️ Clear & Start Over',
        
        # Results
        'results_title': 'Analysis Results',
        'confidence': 'AI Confidence Score',
        'detected': 'Condition Detected',
        'urgency': 'Recommended Action',
        'symptoms_title': 'Symptoms',
        'causes_title': 'Common Causes',
        'treatment_title': 'Treatment Options',
        'all_scores': 'View All Prediction Scores',
        'heatmap_title': 'AI Attention Heatmap',
        'heatmap_desc': 'This visualization shows where the AI focused when making its prediction. Red/yellow areas indicate high attention, blue areas indicate low attention.',
        'original_image': 'Original Image',
        'heatmap_image': 'AI Focus Areas',
        
        # Footer
        'find_dentist': 'Find Dentists Near You',
        'disclaimer_title': 'IMPORTANT MEDICAL DISCLAIMER',
        'disclaimer_text': 'This AI tool is intended for SCREENING PURPOSES ONLY and should not be used as a substitute for professional medical diagnosis. The AI model has an accuracy of approximately 87% and may produce incorrect results. Always consult a qualified healthcare professional for proper diagnosis and treatment.',
        
        # Sidebar
        'language': 'Language',
        'model_performance': 'Model Performance',
        'accuracy': 'Overall Accuracy',
        'cancer_detection': 'Cancer Detection',
        'training_images': 'Training Images',
        'conditions': 'Detectable Conditions',
        
        # Disease names
        'disease_Oral_Cancer': 'Oral Cancer',
        'disease_Ulcers': 'Mouth Ulcers',
        'disease_Gingivitis': 'Gingivitis',
        'disease_Caries': 'Dental Caries (Cavities)',
        'disease_Calculus': 'Calculus (Tartar)',
        'disease_Tooth Discoloration': 'Tooth Discoloration',
        'disease_Hypodontia': 'Hypodontia',
        'disease_Normal_Mouth': 'Healthy Mouth',
        
        # Misc
        'loading': 'Loading...',
        'error': 'Error',
        'success': 'Success',
        'warning': 'Warning',
        'no_image': 'No image selected. Please upload an image or take a photo.',
    },
    
    'hi': {
        # App title and navigation
        'app_title': '🦷 मौखिक स्वास्थ्य AI',
        'app_subtitle': 'AI-संचालित मौखिक रोग जांच • जल्दी पता लगाने से जीवन बचता है',
        'nav_home': '🏠 होम',
        'nav_scan': '🔍 स्कैन',
        'nav_history': '📊 परिणाम',
        'nav_info': 'ℹ️ जानकारी',
        
        # Risk Assessment
        'risk_title': 'जोखिम मूल्यांकन',
        'risk_subtitle': 'अपने मौखिक स्वास्थ्य जोखिम कारकों का आकलन करने के लिए इन प्रश्नों का उत्तर दें',
        'risk_tobacco': 'क्या आप तंबाकू या गुटखा का उपयोग करते हैं?',
        'risk_paan': 'क्या आप पान या सुपारी खाते हैं?',
        'risk_smoke': 'क्या आप धूम्रपान करते हैं?',
        'risk_alcohol': 'क्या आप नियमित रूप से शराब पीते हैं?',
        'risk_high': 'उच्च जोखिम',
        'risk_medium': 'मध्यम जोखिम',
        'risk_low': 'कम जोखिम',
        'risk_high_msg': 'आपके पास मुंह के कैंसर के कई जोखिम कारक हैं। नियमित जांच की दृढ़ता से अनुशंसा की जाती है!',
        'risk_medium_msg': 'आपके पास कुछ जोखिम कारक हैं। नियमित दंत जांच पर विचार करें।',
        'risk_low_msg': 'बहुत बढ़िया! कोई प्रमुख जोखिम कारक नहीं। अच्छी मौखिक स्वच्छता बनाए रखें!',
        
        # Image Upload
        'upload_title': 'छवि अपलोड या कैप्चर करें',
        'upload_tab': '📁 छवि अपलोड',
        'camera_tab': '📷 कैमरा',
        'upload_prompt': 'अपने मुंह या दांतों की एक स्पष्ट छवि अपलोड करें',
        'camera_enable': '📷 कैमरा चालू करें',
        'camera_disable': '❌ कैमरा बंद करें',
        'camera_prompt': 'प्रभावित क्षेत्र की स्पष्ट छवि लेने के लिए अपना कैमरा स्थित करें',
        'take_photo': 'फोटो लें',
        'analyze_btn': '🔍 छवि का विश्लेषण करें',
        'analyzing': 'आपकी छवि का विश्लेषण किया जा रहा है...',
        'clear_btn': '🗑️ साफ़ करें और फिर से शुरू करें',
        
        # Results
        'results_title': 'विश्लेषण परिणाम',
        'confidence': 'AI विश्वास स्कोर',
        'detected': 'पता लगाई गई स्थिति',
        'urgency': 'अनुशंसित कार्रवाई',
        'symptoms_title': 'लक्षण',
        'causes_title': 'सामान्य कारण',
        'treatment_title': 'उपचार विकल्प',
        'all_scores': 'सभी भविष्यवाणी स्कोर देखें',
        'heatmap_title': 'AI ध्यान हीटमैप',
        'heatmap_desc': 'यह विज़ुअलाइज़ेशन दिखाता है कि AI ने अपनी भविष्यवाणी करते समय कहाँ ध्यान केंद्रित किया। लाल/पीले क्षेत्र उच्च ध्यान इंगित करते हैं।',
        'original_image': 'मूल छवि',
        'heatmap_image': 'AI फोकस क्षेत्र',
        
        # Footer
        'find_dentist': 'अपने पास दंत चिकित्सक खोजें',
        'disclaimer_title': 'महत्वपूर्ण चिकित्सा अस्वीकरण',
        'disclaimer_text': 'यह AI उपकरण केवल स्क्रीनिंग उद्देश्यों के लिए है और पेशेवर चिकित्सा निदान का विकल्प नहीं है। AI मॉडल की सटीकता लगभग 87% है। उचित निदान और उपचार के लिए हमेशा योग्य स्वास्थ्य पेशेवर से परामर्श करें।',
        
        # Sidebar
        'language': 'भाषा',
        'model_performance': 'मॉडल प्रदर्शन',
        'accuracy': 'समग्र सटीकता',
        'cancer_detection': 'कैंसर पहचान',
        'training_images': 'प्रशिक्षण छवियां',
        'conditions': 'पता लगाने योग्य स्थितियां',
        
        # Disease names
        'disease_Oral_Cancer': 'मुंह का कैंसर',
        'disease_Ulcers': 'मुंह के छाले',
        'disease_Gingivitis': 'मसूड़ों की सूजन',
        'disease_Caries': 'दांतों की सड़न (कैविटी)',
        'disease_Calculus': 'कैलकुलस (टार्टर)',
        'disease_Tooth Discoloration': 'दांतों का मलिनकिरण',
        'disease_Hypodontia': 'हाइपोडोंटिया',
        'disease_Normal_Mouth': 'स्वस्थ मुंह',
        
        # Misc
        'loading': 'लोड हो रहा है...',
        'error': 'त्रुटि',
        'success': 'सफलता',
        'warning': 'चेतावनी',
        'no_image': 'कोई छवि चयनित नहीं है। कृपया एक छवि अपलोड करें या फोटो लें।',
    }
}

def get_text(key):
    """Get translated text based on current language"""
    lang = st.session_state.language
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]
    # Fallback to English
    if key in TRANSLATIONS['en']:
        return TRANSLATIONS['en'][key]
    return key

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DISEASE DATABASE WITH FULL TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_DATABASE = {
    'Oral_Cancer': {
        'en': {
            'name': 'Oral Cancer',
            'emoji': '🚨',
            'risk_level': 'high',
            'description': 'Oral cancer is a serious condition where malignant cells form in the tissues of the mouth or throat. Early detection significantly improves survival rates.',
            'symptoms': [
                'Persistent mouth sores that don\'t heal (>2 weeks)',
                'White or red patches inside mouth',
                'Lump or thickening in cheek or neck',
                'Difficulty swallowing or chewing',
                'Numbness in tongue, lip, or mouth',
                'Unexplained bleeding in mouth',
                'Chronic sore throat or hoarseness',
                'Jaw pain or stiffness'
            ],
            'causes': [
                'Tobacco use (smoking, chewing, gutka)',
                'Heavy alcohol consumption',
                'Human papillomavirus (HPV) infection',
                'Excessive sun exposure (lip cancer)',
                'Poor nutrition and diet',
                'Weakened immune system',
                'Family history of cancer',
                'Chronic irritation from rough teeth'
            ],
            'treatments': [
                'Surgical removal of tumor',
                'Radiation therapy',
                'Chemotherapy',
                'Targeted drug therapy',
                'Immunotherapy',
                'Reconstructive surgery',
                'Speech and swallowing therapy',
                'Regular follow-up monitoring'
            ],
            'urgency': 'CRITICAL - Seek immediate medical attention within 24-48 hours. Do not delay!'
        },
        'hi': {
            'name': 'मुंह का कैंसर',
            'emoji': '🚨',
            'risk_level': 'high',
            'description': 'मुंह का कैंसर एक गंभीर स्थिति है जहां मुंह या गले के ऊतकों में घातक कोशिकाएं बनती हैं। जल्दी पता लगाने से जीवित रहने की दर में काफी सुधार होता है।',
            'symptoms': [
                'मुंह में न भरने वाले घाव (>2 सप्ताह)',
                'मुंह के अंदर सफेद या लाल धब्बे',
                'गाल या गर्दन में गांठ',
                'निगलने या चबाने में कठिनाई',
                'जीभ या होंठ में सुन्नता',
                'मुंह में अस्पष्ट रक्तस्राव',
                'लंबे समय तक गले में खराश',
                'जबड़े में दर्द या अकड़न'
            ],
            'causes': [
                'तंबाकू का उपयोग (धूम्रपान, चबाना, गुटखा)',
                'अत्यधिक शराब का सेवन',
                'HPV संक्रमण',
                'अत्यधिक धूप',
                'खराब पोषण',
                'कमजोर प्रतिरक्षा प्रणाली',
                'कैंसर का पारिवारिक इतिहास',
                'खुरदरे दांतों से पुरानी जलन'
            ],
            'treatments': [
                'ट्यूमर को शल्य चिकित्सा से हटाना',
                'विकिरण चिकित्सा',
                'कीमोथेरेपी',
                'लक्षित दवा चिकित्सा',
                'इम्यूनोथेरेपी',
                'पुनर्निर्माण सर्जरी',
                'भाषण और निगलने की थेरेपी',
                'नियमित अनुवर्ती निगरानी'
            ],
            'urgency': 'गंभीर - 24-48 घंटों के भीतर तुरंत चिकित्सा सहायता लें। देरी न करें!'
        }
    },
    
    'Ulcers': {
        'en': {
            'name': 'Mouth Ulcers (Canker Sores)',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'Mouth ulcers are painful sores that appear inside the mouth. Most heal within 1-2 weeks without treatment, but persistent ulcers need evaluation.',
            'symptoms': [
                'Painful round or oval sores',
                'White or yellow center with red border',
                'Burning sensation before appearing',
                'Difficulty eating spicy or acidic foods',
                'Swelling around the sore',
                'Tingling sensation in mouth',
                'Multiple sores at once',
                'Pain when talking or eating'
            ],
            'causes': [
                'Stress and anxiety',
                'Minor mouth injuries (biting cheek)',
                'Acidic or spicy foods',
                'Vitamin deficiencies (B12, iron, folate)',
                'Hormonal changes',
                'Food allergies or sensitivities',
                'Certain medications',
                'Weakened immune system'
            ],
            'treatments': [
                'Antiseptic mouthwash',
                'Pain-relieving gels (Benzocaine)',
                'Saltwater rinse (warm)',
                'Avoid spicy and acidic foods',
                'Vitamin B12 supplements',
                'Corticosteroid ointments',
                'Soft diet during healing',
                'Maintain good oral hygiene'
            ],
            'urgency': 'Monitor closely - See a dentist if ulcer persists beyond 2 weeks or recurs frequently.'
        },
        'hi': {
            'name': 'मुंह के छाले',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'मुंह के छाले दर्दनाक घाव हैं जो मुंह के अंदर दिखाई देते हैं। अधिकांश 1-2 सप्ताह में बिना उपचार के ठीक हो जाते हैं।',
            'symptoms': [
                'दर्दनाक गोल या अंडाकार घाव',
                'लाल बॉर्डर के साथ सफेद या पीला केंद्र',
                'प्रकट होने से पहले जलन',
                'मसालेदार खाना खाने में कठिनाई',
                'घाव के आसपास सूजन',
                'मुंह में झुनझुनी',
                'एक साथ कई घाव',
                'बात करने या खाने में दर्द'
            ],
            'causes': [
                'तनाव और चिंता',
                'मामूली मुंह की चोट',
                'अम्लीय या मसालेदार भोजन',
                'विटामिन की कमी (B12, आयरन)',
                'हार्मोनल परिवर्तन',
                'खाद्य एलर्जी',
                'कुछ दवाइयां',
                'कमजोर प्रतिरक्षा प्रणाली'
            ],
            'treatments': [
                'एंटीसेप्टिक माउथवॉश',
                'दर्द निवारक जेल',
                'गर्म नमक के पानी से गरारे',
                'मसालेदार भोजन से बचें',
                'विटामिन B12 सप्लीमेंट',
                'कॉर्टिकोस्टेरॉइड मलहम',
                'नरम आहार',
                'अच्छी मौखिक स्वच्छता'
            ],
            'urgency': 'निगरानी करें - यदि 2 सप्ताह से अधिक रहे या बार-बार हो तो दंत चिकित्सक से मिलें।'
        }
    },
    
    'Gingivitis': {
        'en': {
            'name': 'Gingivitis (Gum Disease)',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'Gingivitis is inflammation of the gums caused by bacterial infection. If left untreated, it can progress to periodontitis and eventual tooth loss.',
            'symptoms': [
                'Red, swollen gums',
                'Bleeding while brushing or flossing',
                'Bad breath (halitosis)',
                'Receding gums',
                'Tender or painful gums',
                'Soft, puffy gum tissue',
                'Dark red or purple gum color',
                'Spaces between teeth and gums'
            ],
            'causes': [
                'Poor oral hygiene',
                'Plaque and tartar buildup',
                'Smoking or tobacco use',
                'Diabetes',
                'Hormonal changes (pregnancy)',
                'Certain medications',
                'Dry mouth conditions',
                'Poor nutrition'
            ],
            'treatments': [
                'Professional dental cleaning (scaling)',
                'Improved brushing technique',
                'Daily flossing',
                'Antibacterial mouthwash',
                'Regular dental checkups',
                'Quit smoking',
                'Treat underlying conditions',
                'Soft-bristled toothbrush'
            ],
            'urgency': 'Schedule dental visit within 1-2 weeks for professional evaluation and cleaning.'
        },
        'hi': {
            'name': 'मसूड़ों की सूजन (जिंजिवाइटिस)',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'मसूड़े की सूजन बैक्टीरिया के संक्रमण के कारण होती है। अनुपचारित छोड़ने पर यह पेरियोडोंटाइटिस में बदल सकती है।',
            'symptoms': [
                'लाल, सूजे हुए मसूड़े',
                'ब्रश करते समय खून आना',
                'सांसों की दुर्गंध',
                'मसूड़ों का पीछे हटना',
                'मसूड़ों में दर्द',
                'नरम, फूले हुए मसूड़े',
                'गहरे लाल मसूड़े',
                'दांतों और मसूड़ों के बीच गैप'
            ],
            'causes': [
                'खराब मौखिक स्वच्छता',
                'प्लाक और टार्टर जमाव',
                'धूम्रपान या तंबाकू',
                'मधुमेह',
                'हार्मोनल परिवर्तन (गर्भावस्था)',
                'कुछ दवाइयां',
                'सूखा मुंह',
                'खराब पोषण'
            ],
            'treatments': [
                'पेशेवर दंत सफाई (स्केलिंग)',
                'बेहतर ब्रशिंग तकनीक',
                'दैनिक फ्लॉसिंग',
                'एंटीबैक्टीरियल माउथवॉश',
                'नियमित दंत जांच',
                'धूम्रपान छोड़ें',
                'अंतर्निहित स्थितियों का इलाज',
                'नरम ब्रिसल वाला टूथब्रश'
            ],
            'urgency': 'पेशेवर मूल्यांकन और सफाई के लिए 1-2 सप्ताह के भीतर दंत चिकित्सक से मिलें।'
        }
    },
    
    'Caries': {
        'en': {
            'name': 'Dental Caries (Cavities)',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'Dental caries (cavities) are permanently damaged areas in teeth that develop into tiny holes. They are among the world\'s most common health problems.',
            'symptoms': [
                'Toothache or spontaneous pain',
                'Sensitivity to sweet, hot, or cold',
                'Visible holes or pits in teeth',
                'Brown, black, or white staining',
                'Bad breath',
                'Pain when biting down',
                'Visible dark spots on teeth',
                'Food getting stuck in teeth'
            ],
            'causes': [
                'Frequent snacking on sugary foods',
                'Sugary drinks consumption',
                'Poor brushing habits',
                'Bacteria in mouth',
                'Dry mouth',
                'Lack of fluoride',
                'Eating disorders',
                'Acid reflux (GERD)'
            ],
            'treatments': [
                'Dental fillings (amalgam or composite)',
                'Dental crowns (severe decay)',
                'Root canal treatment',
                'Fluoride treatments',
                'Tooth extraction (if necessary)',
                'Dental sealants',
                'Improved oral hygiene',
                'Dietary changes'
            ],
            'urgency': 'Schedule dental appointment within 1-2 weeks to prevent further decay and complications.'
        },
        'hi': {
            'name': 'दांतों की सड़न (कैविटी)',
            'emoji': '⚠️',
            'risk_level': 'medium',
            'description': 'दंत क्षय (कैविटी) दांतों में स्थायी रूप से क्षतिग्रस्त क्षेत्र हैं जो छोटे छेद बन जाते हैं। ये दुनिया की सबसे आम स्वास्थ्य समस्याओं में से एक है।',
            'symptoms': [
                'दांत दर्द',
                'मीठे, गर्म या ठंडे के प्रति संवेदनशीलता',
                'दांतों में दिखाई देने वाले छेद',
                'भूरे, काले या सफेद दाग',
                'सांसों की दुर्गंध',
                'काटते समय दर्द',
                'दांतों पर काले धब्बे',
                'दांतों में खाना फंसना'
            ],
            'causes': [
                'मीठे खाद्य पदार्थों का बार-बार सेवन',
                'शर्करा युक्त पेय',
                'खराब ब्रशिंग आदतें',
                'मुंह में बैक्टीरिया',
                'सूखा मुंह',
                'फ्लोराइड की कमी',
                'खाने के विकार',
                'एसिड रिफ्लक्स'
            ],
            'treatments': [
                'डेंटल फिलिंग',
                'डेंटल क्राउन (गंभीर सड़न)',
                'रूट कैनाल उपचार',
                'फ्लोराइड उपचार',
                'दांत निकालना (यदि आवश्यक)',
                'डेंटल सीलेंट',
                'बेहतर मौखिक स्वच्छता',
                'आहार में बदलाव'
            ],
            'urgency': 'आगे की सड़न को रोकने के लिए 1-2 सप्ताह के भीतर दंत चिकित्सक से मिलें।'
        }
    },
    
    'Calculus': {
        'en': {
            'name': 'Calculus (Tartar)',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'Calculus (tartar) is hardened dental plaque that has mineralized on teeth. It cannot be removed by regular brushing and requires professional cleaning.',
            'symptoms': [
                'Yellow or brown deposits on teeth',
                'Rough feeling on tooth surface',
                'Bad breath',
                'Gum irritation and inflammation',
                'Bleeding gums',
                'Teeth appear darker',
                'Buildup along gum line',
                'Receding gums'
            ],
            'causes': [
                'Poor oral hygiene',
                'Not flossing regularly',
                'Smoking or tobacco use',
                'Dry mouth conditions',
                'Diet high in sugar and starch',
                'Irregular dental visits',
                'Certain medications',
                'Age-related changes'
            ],
            'treatments': [
                'Professional scaling and cleaning',
                'Root planing',
                'Improved daily oral hygiene',
                'Electric toothbrush',
                'Regular dental cleanings (every 6 months)',
                'Tartar-control toothpaste',
                'Antiseptic mouthwash',
                'Dietary modifications'
            ],
            'urgency': 'Schedule professional dental cleaning within 1 month to prevent gum disease.'
        },
        'hi': {
            'name': 'कैलकुलस (टार्टर)',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'कैलकुलस (टार्टर) कठोर दंत पट्टिका है जो दांतों पर खनिज हो गई है। इसे नियमित ब्रश से नहीं हटाया जा सकता।',
            'symptoms': [
                'दांतों पर पीले या भूरे जमाव',
                'दांतों की सतह पर खुरदरापन',
                'सांसों की दुर्गंध',
                'मसूड़ों में जलन और सूजन',
                'मसूड़ों से खून',
                'दांत गहरे दिखना',
                'मसूड़ों की रेखा पर जमाव',
                'मसूड़ों का पीछे हटना'
            ],
            'causes': [
                'खराब मौखिक स्वच्छता',
                'नियमित फ्लॉसिंग न करना',
                'धूम्रपान या तंबाकू',
                'सूखा मुंह',
                'चीनी और स्टार्च युक्त आहार',
                'अनियमित दंत जांच',
                'कुछ दवाइयां',
                'उम्र से संबंधित परिवर्तन'
            ],
            'treatments': [
                'पेशेवर स्केलिंग और सफाई',
                'रूट प्लानिंग',
                'बेहतर दैनिक मौखिक स्वच्छता',
                'इलेक्ट्रिक टूथब्रश',
                'नियमित दंत सफाई (हर 6 महीने)',
                'टार्टर-कंट्रोल टूथपेस्ट',
                'एंटीसेप्टिक माउथवॉश',
                'आहार में संशोधन'
            ],
            'urgency': 'मसूड़ों की बीमारी को रोकने के लिए 1 महीने के भीतर पेशेवर दंत सफाई करवाएं।'
        }
    },
    
    'Tooth Discoloration': {
        'en': {
            'name': 'Tooth Discoloration',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'Tooth discoloration refers to staining or color changes in teeth. It can be extrinsic (surface stains) or intrinsic (internal discoloration).',
            'symptoms': [
                'Yellow or brown teeth',
                'White spots on teeth',
                'Gray or dark colored teeth',
                'Uneven tooth coloring',
                'Stains between teeth',
                'Dull appearance of teeth',
                'Brownish spots near gum line',
                'Discoloration after injury'
            ],
            'causes': [
                'Coffee, tea, or red wine consumption',
                'Tobacco use',
                'Poor dental hygiene',
                'Certain medications (tetracycline)',
                'Aging',
                'Excessive fluoride (fluorosis)',
                'Dental trauma',
                'Genetic factors'
            ],
            'treatments': [
                'Professional teeth whitening',
                'Whitening toothpaste',
                'Dental veneers',
                'Dental bonding',
                'Better oral hygiene routine',
                'Avoiding staining foods/drinks',
                'At-home whitening kits',
                'Dental crowns (severe cases)'
            ],
            'urgency': 'Non-urgent - Cosmetic concern. Consult dentist at your convenience for whitening options.'
        },
        'hi': {
            'name': 'दांतों का मलिनकिरण',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'दांतों का मलिनकिरण दांतों में दाग या रंग परिवर्तन को संदर्भित करता है। यह बाहरी (सतह के दाग) या आंतरिक हो सकता है।',
            'symptoms': [
                'पीले या भूरे दांत',
                'दांतों पर सफेद धब्बे',
                'धूसर या गहरे रंग के दांत',
                'असमान दांतों का रंग',
                'दांतों के बीच दाग',
                'दांतों की सुस्त उपस्थिति',
                'मसूड़ों की रेखा के पास भूरे धब्बे',
                'चोट के बाद मलिनकिरण'
            ],
            'causes': [
                'कॉफी, चाय या रेड वाइन',
                'तंबाकू का उपयोग',
                'खराब दंत स्वच्छता',
                'कुछ दवाइयां (टेट्रासाइक्लिन)',
                'उम्र बढ़ना',
                'अत्यधिक फ्लोराइड',
                'दंत आघात',
                'आनुवंशिक कारक'
            ],
            'treatments': [
                'पेशेवर दांत सफेद करना',
                'व्हाइटनिंग टूथपेस्ट',
                'डेंटल वेनीर्स',
                'डेंटल बॉन्डिंग',
                'बेहतर मौखिक स्वच्छता',
                'दाग लगाने वाले खाद्य पदार्थों से बचें',
                'होम व्हाइटनिंग किट',
                'डेंटल क्राउन (गंभीर मामले)'
            ],
            'urgency': 'गैर-जरूरी - सौंदर्य संबंधी चिंता। व्हाइटनिंग विकल्पों के लिए अपनी सुविधा अनुसार दंत चिकित्सक से मिलें।'
        }
    },
    
    'Hypodontia': {
        'en': {
            'name': 'Hypodontia (Missing Teeth)',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'Hypodontia is a developmental condition where one or more teeth fail to develop. It can affect dental function, appearance, and jaw development.',
            'symptoms': [
                'Visible gaps between teeth',
                'Difficulty chewing properly',
                'Speech difficulties',
                'Jawbone development issues',
                'Misalignment of existing teeth',
                'Aesthetic concerns',
                'Baby teeth that don\'t fall out',
                'Smaller than normal teeth'
            ],
            'causes': [
                'Genetic factors (inherited)',
                'Developmental abnormalities',
                'Trauma during tooth development',
                'Radiation therapy',
                'Certain genetic syndromes',
                'Environmental factors',
                'Infections during pregnancy',
                'Unknown causes'
            ],
            'treatments': [
                'Dental implants',
                'Fixed dental bridges',
                'Removable partial dentures',
                'Orthodontic treatment (braces)',
                'Space maintainers (for children)',
                'Dental bonding',
                'Resin-retained bridges',
                'Regular monitoring'
            ],
            'urgency': 'Non-urgent - Consult a dentist or orthodontist for evaluation of treatment options.'
        },
        'hi': {
            'name': 'हाइपोडोंटिया (गायब दांत)',
            'emoji': '📋',
            'risk_level': 'low',
            'description': 'हाइपोडोंटिया एक विकासात्मक स्थिति है जहां एक या अधिक दांत विकसित नहीं होते। यह दंत कार्य और जबड़े के विकास को प्रभावित कर सकता है।',
            'symptoms': [
                'दांतों के बीच दिखाई देने वाले गैप',
                'ठीक से चबाने में कठिनाई',
                'बोलने में कठिनाई',
                'जबड़े के विकास की समस्या',
                'मौजूदा दांतों का गलत संरेखण',
                'सौंदर्य संबंधी चिंताएं',
                'दूध के दांत जो नहीं गिरते',
                'सामान्य से छोटे दांत'
            ],
            'causes': [
                'आनुवंशिक कारक (विरासत में मिला)',
                'विकासात्मक असामान्यताएं',
                'दांत विकास के दौरान आघात',
                'विकिरण चिकित्सा',
                'कुछ आनुवंशिक सिंड्रोम',
                'पर्यावरणीय कारक',
                'गर्भावस्था के दौरान संक्रमण',
                'अज्ञात कारण'
            ],
            'treatments': [
                'डेंटल इम्प्लांट',
                'फिक्स्ड डेंटल ब्रिज',
                'रिमूवेबल पार्शियल डेंचर',
                'ऑर्थोडॉन्टिक उपचार (ब्रेसेस)',
                'स्पेस मेंटेनर (बच्चों के लिए)',
                'डेंटल बॉन्डिंग',
                'रेजिन-रिटेन्ड ब्रिज',
                'नियमित निगरानी'
            ],
            'urgency': 'गैर-जरूरी - उपचार विकल्पों के मूल्यांकन के लिए दंत चिकित्सक या ऑर्थोडॉन्टिस्ट से मिलें।'
        }
    },
    
    'Normal_Mouth': {
        'en': {
            'name': 'Healthy Mouth',
            'emoji': '✅',
            'risk_level': 'low',
            'description': 'Great news! Your oral health appears to be in good condition. Continue maintaining your current oral hygiene practices to keep your teeth and gums healthy.',
            'symptoms': [
                'Pink and firm gums',
                'No bleeding when brushing',
                'Fresh breath',
                'Clean teeth without visible plaque',
                'No pain or sensitivity',
                'Properly aligned teeth',
                'No visible cavities or decay',
                'Healthy tongue color'
            ],
            'causes': [],
            'treatments': [
                'Continue brushing twice daily (2 minutes)',
                'Floss once daily',
                'Use fluoride toothpaste',
                'Regular dental checkups (every 6 months)',
                'Maintain balanced diet',
                'Limit sugary foods and drinks',
                'Stay hydrated',
                'Replace toothbrush every 3-4 months'
            ],
            'urgency': 'Routine dental checkup every 6 months to maintain optimal oral health.'
        },
        'hi': {
            'name': 'स्वस्थ मुंह',
            'emoji': '✅',
            'risk_level': 'low',
            'description': 'बहुत अच्छी खबर! आपका मौखिक स्वास्थ्य अच्छी स्थिति में दिखाई देता है। अपने दांतों और मसूड़ों को स्वस्थ रखने के लिए अपनी वर्तमान मौखिक स्वच्छता प्रथाओं को जारी रखें।',
            'symptoms': [
                'गुलाबी और मजबूत मसूड़े',
                'ब्रश करते समय खून नहीं आता',
                'ताजी सांस',
                'बिना प्लाक के साफ दांत',
                'कोई दर्द या संवेदनशीलता नहीं',
                'ठीक से संरेखित दांत',
                'कोई दिखाई देने वाली कैविटी नहीं',
                'स्वस्थ जीभ का रंग'
            ],
            'causes': [],
            'treatments': [
                'दिन में दो बार ब्रश करना जारी रखें',
                'रोजाना फ्लॉस करें',
                'फ्लोराइड टूथपेस्ट का उपयोग करें',
                'नियमित दंत जांच (हर 6 महीने)',
                'संतुलित आहार बनाए रखें',
                'मीठे खाद्य पदार्थ सीमित करें',
                'हाइड्रेटेड रहें',
                'हर 3-4 महीने में टूथब्रश बदलें'
            ],
            'urgency': 'इष्टतम मौखिक स्वास्थ्य बनाए रखने के लिए हर 6 महीने में नियमित दंत जांच।'
        }
    }
}

def get_disease_info(disease_key, lang='en'):
    """Get disease information in specified language"""
    if disease_key in DISEASE_DATABASE:
        if lang in DISEASE_DATABASE[disease_key]:
            return DISEASE_DATABASE[disease_key][lang]
        return DISEASE_DATABASE[disease_key]['en']
    return None
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CUSTOM CSS STYLES
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* ═══════════════════════════════════════════════════════════════════════
       GLOBAL STYLES AND FONTS
       ═══════════════════════════════════════════════════════════════════════ */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    div[data-testid="stDecoration"] {visibility: hidden !important;}
    
    /* ═══════════════════════════════════════════════════════════════════════
       HEADER AND LOGO SECTION
       ═══════════════════════════════════════════════════════════════════════ */
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        padding: 20px 0;
        margin-bottom: 10px;
    }
    
    .logo-icon {
        font-size: 4rem;
        filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5));
    }
    
    .logo-text {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    
    .app-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       NAVIGATION TABS
       ═══════════════════════════════════════════════════════════════════════ */
    
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    
    .nav-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #94a3b8;
        padding: 12px 25px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .nav-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
        color: #e2e8f0;
        transform: translateY(-2px);
    }
    
    .nav-btn-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: transparent;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       CARD COMPONENTS
       ═══════════════════════════════════════════════════════════════════════ */
    
    .card {
        background: linear-gradient(145deg, rgba(30, 30, 47, 0.9) 0%, rgba(37, 37, 64, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .card-icon {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
    }
    
    .card-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       RESULT CARDS - COLOR CODED BY RISK
       ═══════════════════════════════════════════════════════════════════════ */
    
    .result-card {
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card-high {
        background: linear-gradient(145deg, rgba(127, 29, 29, 0.8) 0%, rgba(153, 27, 27, 0.6) 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
    }
    
    .result-card-medium {
        background: linear-gradient(145deg, rgba(120, 53, 15, 0.8) 0%, rgba(146, 64, 14, 0.6) 100%);
        border: 2px solid #f59e0b;
        box-shadow: 0 10px 40px rgba(245, 158, 11, 0.3);
    }
    
    .result-card-low {
        background: linear-gradient(145deg, rgba(20, 83, 45, 0.8) 0%, rgba(22, 101, 52, 0.6) 100%);
        border: 2px solid #22c55e;
        box-shadow: 0 10px 40px rgba(34, 197, 94, 0.3);
    }
    
    .result-disease-name {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 15px;
    }
    
    .result-disease-name-high { color: #fca5a5; }
    .result-disease-name-medium { color: #fcd34d; }
    .result-disease-name-low { color: #86efac; }
    
    /* ═══════════════════════════════════════════════════════════════════════
       CONFIDENCE SCORE DISPLAY
       ═══════════════════════════════════════════════════════════════════════ */
    
    .confidence-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
    }
    
    .confidence-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    
    .confidence-value {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
    }
    
    .confidence-high { color: #f87171; }
    .confidence-medium { color: #fbbf24; }
    .confidence-low { color: #4ade80; }
    
    /* ═══════════════════════════════════════════════════════════════════════
       INFO CARDS (SYMPTOMS, CAUSES, TREATMENTS)
       ═══════════════════════════════════════════════════════════════════════ */
    
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-3px);
    }
    
    .info-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .info-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }
    
    .info-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .info-list li {
        color: #cbd5e1;
        padding: 8px 0;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }
    
    .info-list li:last-child {
        border-bottom: none;
    }
    
    .info-list li::before {
        content: "•";
        color: #667eea;
        font-weight: bold;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       PREDICTION SCORES BAR
       ═══════════════════════════════════════════════════════════════════════ */
    
    .prediction-bar-container {
        margin: 10px 0;
    }
    
    .prediction-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }
    
    .prediction-bar-name {
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .prediction-bar-value {
        color: #94a3b8;
        font-weight: 600;
    }
    
    .prediction-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
    }
    
    .prediction-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
    
    .prediction-bar-fill-high {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    .prediction-bar-fill-top {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       HEATMAP SECTION
       ═══════════════════════════════════════════════════════════════════════ */
    
    .heatmap-container {
        background: linear-gradient(145deg, rgba(30, 30, 47, 0.9) 0%, rgba(37, 37, 64, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin: 25px 0;
    }
    
    .heatmap-title {
        font-size: 1.3rem;
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
        line-height: 1.6;
    }
    
    .heatmap-legend {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    
    .heatmap-legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
    }
    
    .legend-red { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .legend-yellow { background: linear-gradient(135deg, #fbbf24, #f59e0b); }
    .legend-blue { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    
    /* ═══════════════════════════════════════════════════════════════════════
       RISK ASSESSMENT BADGES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-badge-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    
    .risk-badge-medium {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(217, 119, 6, 0.4);
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.4);
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       URGENCY BADGE
       ═══════════════════════════════════════════════════════════════════════ */
    
    .urgency-badge {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .urgency-badge-medium {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }
    
    .urgency-badge-low {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.5);
        color: #86efac;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       DISCLAIMER BOX
       ═══════════════════════════════════════════════════════════════════════ */
    
    .disclaimer-box {
        background: linear-gradient(145deg, rgba(120, 53, 15, 0.6) 0%, rgba(146, 64, 14, 0.4) 100%);
        border: 2px solid #f59e0b;
        border-radius: 15px;
        padding: 25px;
        margin: 30px 0;
    }
    
    .disclaimer-title {
        color: #fbbf24;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .disclaimer-text {
        color: #fef3c7;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       FIND DENTIST BUTTON
       ═══════════════════════════════════════════════════════════════════════ */
    
    .dentist-btn {
        display: block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        text-decoration: none;
        padding: 18px 40px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 20px auto;
        max-width: 400px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .dentist-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        color: white !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       SIDEBAR STYLES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .sidebar-section {
        margin-bottom: 25px;
    }
    
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
    
    .sidebar-metric {
        background: linear-gradient(145deg, rgba(30, 30, 47, 0.9) 0%, rgba(37, 37, 64, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
    }
    
    .sidebar-metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        word-break: break-word;
    }
    
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    .condition-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
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
    
    /* ═══════════════════════════════════════════════════════════════════════
       BUTTON STYLES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Camera toggle button */
    .camera-toggle-btn {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .camera-toggle-btn:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Clear button */
    .clear-btn > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       IMAGE DISPLAY
       ═══════════════════════════════════════════════════════════════════════ */
    
    .image-frame {
        background: rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
        overflow: hidden;
    }
    
    .image-caption {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 10px;
        font-weight: 500;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       CHECKBOX AND INPUT STYLES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .risk-checkbox-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 15px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    
    .risk-checkbox-container:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       TAB STYLES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 25px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: transparent !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       EXPANDER STYLES
       ═══════════════════════════════════════════════════════════════════════ */
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.2) !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════════════════════════════════ */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
       RESPONSIVE DESIGN
       ═══════════════════════════════════════════════════════════════════════ */
    
    @media (max-width: 768px) {
        .logo-text {
            font-size: 2rem;
        }
        
        .logo-icon {
            font-size: 3rem;
        }
        
        .result-disease-name {
            font-size: 1.5rem;
        }
        
        .confidence-value {
            font-size: 2.5rem;
        }
        
        .card {
            padding: 15px;
        }
        
        .nav-container {
            gap: 5px;
        }
        
        .nav-btn {
            padding: 8px 15px;
            font-size: 0.85rem;
        }
    }
    /* Remove red indicator bar from tabs */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Better tab panel spacing */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 15px;
        border-top: none !important;
    }
</style>
"""

def load_css():
    """Load custom CSS styles"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL LOADING AND PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained TensorFlow model with caching"""
    if not TF_AVAILABLE:
        return None
    
    model_path = 'model/oral_disease_model.h5'
    
    if not os.path.exists(model_path):
        return None
    
    try:
        # Method 1: Load with compile=False
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e1:
        try:
            # Method 2: Recreate architecture and load weights
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

@st.cache_data(show_spinner=False)
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

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for EfficientNetB0 model prediction.
    CRITICAL: Must match training preprocessing exactly.
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize with high quality
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # ═══════════════════════════════════════════════════════════════════════
    # OPTION 1: Simple 0-1 scaling (most common with ImageDataGenerator)
    # If this doesn't work, try OPTION 2 below
    # ═══════════════════════════════════════════════════════════════════════
    img_array = img_array / 255.0
    
    # OPTION 2: EfficientNet preprocessing (scales to [-1, 1])
    # Uncomment below and comment OPTION 1 if needed
    # img_array = (img_array / 127.5) - 1.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, img_array, class_names):
    """Run prediction with detailed debugging output"""
    if model is None:
        return None
    
    try:
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        pred_values = predictions[0]
        
        # Debug output
        print("=" * 60)
        print("PREDICTION DEBUG:")
        print(f"Raw values: {pred_values}")
        print(f"Sum: {np.sum(pred_values):.4f}, Std: {np.std(pred_values):.6f}")
        
        # Warning check
        if np.std(pred_values) < 0.01:
            print("WARNING: Low variance in predictions!")
            print("This means preprocessing doesn't match training.")
            print("Try switching OPTION 1 <-> OPTION 2 in preprocess_image()")
        
        # Get top prediction
        pred_idx = int(np.argmax(pred_values))
        pred_class = class_names[pred_idx]
        confidence = float(pred_values[pred_idx]) * 100
        
        # Get all scores
        all_scores = {}
        for i, class_name in enumerate(class_names):
            all_scores[class_name] = float(pred_values[i]) * 100
        
        # Print sorted predictions
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        print("Predictions (sorted):")
        for name, score in sorted_scores:
            marker = " <--" if name == pred_class else ""
            print(f"  {name}: {score:.2f}%{marker}")
        print("=" * 60)
        
        return {
            'class': pred_class,
            'index': pred_idx,
            'confidence': confidence,
            'all_scores': all_scores
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: ROBUST GRADCAM HEATMAP IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def find_target_layer(model):
    """Find the last convolutional layer for GradCAM"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    # Fallback: find any layer with 4D output
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            continue
    return None

def compute_gradcam_heatmap(model, img_array, pred_index):
    """
    Compute GradCAM heatmap using TensorFlow GradientTape.
    Returns a normalized 2D heatmap array.
    """
    if not TF_AVAILABLE:
        return None
    
    try:
        # Find target layer - look for conv layers in EfficientNet
        target_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and 'bn' not in layer.name.lower():
                target_layer_name = layer.name
                break
        
        if target_layer_name is None:
            # Try to find top_conv in EfficientNet
            for layer in model.layers:
                if 'top_conv' in layer.name.lower():
                    target_layer_name = layer.name
                    break
        
        if target_layer_name is None:
            print("No conv layer found")
            return None
        
        print(f"Using layer: {target_layer_name}")
        
        # Create gradient model
        target_layer = model.get_layer(target_layer_name)
        gradient_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        
        # Convert to tensor and enable gradient tracking
        img_tensor = tf.cast(img_array, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_output, predictions = gradient_model(img_tensor, training=False)
            class_output = predictions[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(class_output, conv_output)
        
        if grads is None:
            print("Gradients are None")
            return None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get conv output for this image
        conv_output = conv_output[0]
        
        # Weight each channel by gradient importance
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        
        # ReLU to keep only positive influence
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    
    except Exception as e:
        print(f"GradCAM error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_heatmap_overlay(original_image, heatmap, intensity=0.5):
    """
    Create a colored heatmap overlay using JET colormap.
    Blue (low attention) -> Green -> Yellow -> Red (high attention)
    """
    if heatmap is None:
        return None
    
    try:
        img_size = (224, 224)
        
        # Prepare original image
        img = original_image.copy()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(img_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), img_size)
        
        # Smooth the heatmap
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
        
        # Normalize to 0-1
        heatmap_min = heatmap_resized.min()
        heatmap_max = heatmap_resized.max()
        
        if heatmap_max - heatmap_min > 1e-8:
            heatmap_normalized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            return np.uint8(img_array)
        
        # Convert to uint8 for colormap
        heatmap_uint8 = np.uint8(255 * heatmap_normalized)
        
        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # CRITICAL: Convert BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = np.uint8(img_array * (1 - intensity) + heatmap_colored * intensity)
        
        return overlay
    
    except Exception as e:
        print(f"Heatmap error: {e}")
        return None
    
    except Exception as e:
        print(f"Overlay error: {e}")
        return None

def generate_heatmap_visualization(original_image, model, pred_idx):
    """
    Main function to generate GradCAM visualization.
    """
    if model is None or original_image is None:
        return None
    
    try:
        # Preprocess image for model
        img_array = preprocess_image(original_image)
        
        # Compute GradCAM heatmap
        heatmap = compute_gradcam_heatmap(model, img_array, pred_idx)
        
        if heatmap is None:
            # Fallback: create a simple activation-based heatmap
            return create_fallback_heatmap(original_image, model, img_array)
        
        # Create colored overlay
        overlay = create_heatmap_overlay(original_image, heatmap, intensity=0.5)
        
        return overlay
    
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        return None

def create_fallback_heatmap(original_image, model, img_array):
    """
    Fallback heatmap using last conv layer activations when GradCAM fails.
    """
    try:
        # Find a conv layer
        target_layer_name = find_target_layer(model)
        if target_layer_name is None:
            return None
        
        # Create model to get conv outputs
        activation_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(target_layer_name).output
        )
        
        # Get activations
        activations = activation_model(img_array)
        
        # Average across all feature maps
        heatmap = tf.reduce_mean(activations, axis=-1)[0]
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        # Create overlay
        return create_heatmap_overlay(original_image, heatmap.numpy(), intensity=0.5)
    
    except Exception as e:
        print(f"Fallback heatmap error: {e}")
        return None
    
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the app header with logo and title"""
    lang = st.session_state.language
    
    st.markdown(f"""
    <div class="logo-container">
        <span class="logo-icon">🦷</span>
        <span class="logo-text">{get_text('app_title').replace('🦷 ', '')}</span>
    </div>
    <p class="app-subtitle">{get_text('app_subtitle')}</p>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with settings and info"""
    with st.sidebar:
        # Language selector
        st.markdown(f"### 🌐 {get_text('language')}")
        lang_options = {"English": "en", "हिंदी": "hi"}
        current_lang_name = "English" if st.session_state.language == "en" else "हिंदी"
        
        selected_lang = st.selectbox(
            "Language",
            options=list(lang_options.keys()),
            index=0 if st.session_state.language == "en" else 1,
            label_visibility="collapsed",
            key="lang_selector"
        )
        
        # Update language if changed
        new_lang = lang_options[selected_lang]
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()
        
        st.markdown("---")
        
        # Model Performance
        st.markdown(f"### 📊 {get_text('model_performance')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="sidebar-metric">
                <div class="sidebar-metric-value">86.96%</div>
                <div class="sidebar-metric-label">ACCURACY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="sidebar-metric">
                <div class="sidebar-metric-value">91%</div>
                <div class="sidebar-metric-label">CANCER</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">10,860</div>
            <div class="sidebar-metric-label">TRAINING IMAGES</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detectable Conditions
        st.markdown(f"### 🎯 {get_text('conditions')}")
        
        # Fixed: Simple text without duplicate emojis
        st.markdown("🔴 Oral Cancer")
        st.markdown("🟠 Mouth Ulcers")
        st.markdown("🟠 Gingivitis")
        st.markdown("🟠 Dental Caries")
        st.markdown("🟢 Calculus")
        st.markdown("🟢 Tooth Discoloration")
        st.markdown("🟢 Hypodontia")
        st.markdown("🟢 Healthy Mouth")
        
        st.markdown("---")
        
        # Links
        st.markdown("### 🔗 Links")
        st.markdown("[📂 GitHub](https://github.com/ArihantKhaitan/oral-health-ai)")
        st.markdown("[🤗 Hugging Face](https://huggingface.co/spaces/Arihant2409/oral-health-ai)")

def render_risk_assessment():
    """Render risk assessment section"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">📋</div>
            <div>
                <h3 class="card-title">""" + get_text('risk_title') + """</h3>
                <p class="card-subtitle">""" + get_text('risk_subtitle') + """</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.risk_tobacco = st.checkbox(
            f"🚬 {get_text('risk_tobacco')}",
            value=st.session_state.risk_tobacco,
            key="cb_tobacco"
        )
        st.session_state.risk_paan = st.checkbox(
            f"🌿 {get_text('risk_paan')}",
            value=st.session_state.risk_paan,
            key="cb_paan"
        )
    
    with col2:
        st.session_state.risk_smoke = st.checkbox(
            f"🔥 {get_text('risk_smoke')}",
            value=st.session_state.risk_smoke,
            key="cb_smoke"
        )
        st.session_state.risk_alcohol = st.checkbox(
            f"🍺 {get_text('risk_alcohol')}",
            value=st.session_state.risk_alcohol,
            key="cb_alcohol"
        )
    
    # Calculate risk score
    risk_score = sum([
        st.session_state.risk_tobacco,
        st.session_state.risk_paan,
        st.session_state.risk_smoke,
        st.session_state.risk_alcohol
    ])
    
    # Display risk level - integrated look
    if risk_score >= 3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(220,38,38,0.2) 0%, rgba(185,28,28,0.2) 100%); 
                    border: 1px solid #ef4444; border-radius: 12px; padding: 15px; margin-top: 15px;">
            <span style="color: #fca5a5; font-weight: 700; font-size: 1rem;">
                🚨 {get_text('risk_high')} ({risk_score}/4) - {get_text('risk_high_msg')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif risk_score >= 1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(217,119,6,0.2) 0%, rgba(180,83,9,0.2) 100%); 
                    border: 1px solid #f59e0b; border-radius: 12px; padding: 15px; margin-top: 15px;">
            <span style="color: #fcd34d; font-weight: 700; font-size: 1rem;">
                ⚠️ {get_text('risk_medium')} ({risk_score}/4) - {get_text('risk_medium_msg')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(22,163,74,0.2) 0%, rgba(21,128,61,0.2) 100%); 
                    border: 1px solid #22c55e; border-radius: 12px; padding: 15px; margin-top: 15px;">
            <span style="color: #86efac; font-weight: 700; font-size: 1rem;">
                ✅ {get_text('risk_low')} - {get_text('risk_low_msg')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return risk_score

def render_image_input():
    """Render image input section with upload and camera options"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">📸</div>
            <div>
                <h3 class="card-title">""" + get_text('upload_title') + """</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for upload and camera
    tab1, tab2 = st.tabs([get_text('upload_tab'), get_text('camera_tab')])
    
    with tab1:
        uploaded_file = st.file_uploader(
            get_text('upload_prompt'),
            type=['jpg', 'jpeg', 'png'],
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Create new image each time
            new_image = Image.open(uploaded_file)
            st.session_state.current_image = new_image
            st.session_state.image_source = 'upload'
            # Reset analysis when new image uploaded
            st.session_state.analysis_done = False
            st.session_state.analysis_result = None
            st.session_state.heatmap_image = None
    
    with tab2:
        # Camera toggle button
        if st.button(
            get_text('camera_disable') if st.session_state.camera_enabled else get_text('camera_enable'),
            key="camera_toggle",
            use_container_width=True
        ):
            st.session_state.camera_enabled = not st.session_state.camera_enabled
            st.rerun()
        
        # Show camera only if enabled
        if st.session_state.camera_enabled:
            st.info(f"📸 {get_text('camera_prompt')}")
            
            camera_image = st.camera_input(
                get_text('take_photo'),
                key="camera_input",
                label_visibility="collapsed"
            )
            
            if camera_image is not None:
                new_image = Image.open(camera_image)
                st.session_state.current_image = new_image
                st.session_state.image_source = 'camera'
                # Reset analysis when new image captured
                st.session_state.analysis_done = False
                st.session_state.analysis_result = None
                st.session_state.heatmap_image = None
        else:
            st.info(f"👆 Click above to enable camera")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show selected image preview
    if st.session_state.current_image is not None:
        st.markdown("### 📷 Selected Image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                st.session_state.current_image,
                use_column_width=True
            )
        
        # Analyze and Clear buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            analyze_clicked = st.button(
                get_text('analyze_btn'),
                key="analyze_button",
                use_container_width=True,
                type="primary"
            )
        
        with col3:
            clear_clicked = st.button(
                get_text('clear_btn'),
                key="clear_button",
                use_container_width=True
            )
            
            if clear_clicked:
                st.session_state.current_image = None
                st.session_state.analysis_done = False
                st.session_state.analysis_result = None
                st.session_state.heatmap_image = None
                st.rerun()
        
        return analyze_clicked
    
    return False

def render_results(result, original_image, heatmap_overlay, risk_score):
    """Render analysis results"""
    lang = st.session_state.language
    pred_class = result['class']
    confidence = result['confidence']
    
    # Get disease info
    disease_info = get_disease_info(pred_class, lang)
    if disease_info is None:
        disease_info = get_disease_info(pred_class, 'en')
    
    # Determine risk level
    risk_level = disease_info.get('risk_level', 'low')
    if risk_score >= 2 and risk_level == 'medium':
        risk_level = 'high'
    
    # Results header
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">📊</div>
            <div>
                <h3 class="card-title">""" + get_text('results_title') + """</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Main result layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"#### 📷 {get_text('original_image')}")
        st.image(original_image, use_column_width=True)
    
    with col2:
        # Result card based on risk level
        card_class = f"result-card-{risk_level}"
        name_class = f"result-disease-name-{risk_level}"
        conf_class = 'confidence-high' if confidence > 85 else ('confidence-medium' if confidence > 60 else 'confidence-low')
        
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-disease-name {name_class}">
                {disease_info['emoji']} {disease_info['name']}
            </div>
            <div class="confidence-container">
                <div class="confidence-label">{get_text('confidence')}</div>
                <div class="confidence-value {conf_class}">{confidence:.1f}%</div>
            </div>
            <p style="color: #e2e8f0; line-height: 1.6; margin-top: 15px;">
                {disease_info['description']}
            </p>
            <div class="urgency-badge urgency-badge-{risk_level}">
                ⏰ {disease_info['urgency']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed Information Cards
    st.markdown(f"### 📋 {get_text('symptoms_title')}, {get_text('causes_title')} & {get_text('treatment_title')}")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-header">
                <span>🔍</span>
                <span class="info-card-title">{get_text('symptoms_title')}</span>
            </div>
            <ul class="info-list">
        """, unsafe_allow_html=True)
        
        for symptom in disease_info.get('symptoms', [])[:6]:
            st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-header">
                <span>⚡</span>
                <span class="info-card-title">{get_text('causes_title')}</span>
            </div>
            <ul class="info-list">
        """, unsafe_allow_html=True)
        
        for cause in disease_info.get('causes', [])[:6]:
            st.markdown(f"<li>{cause}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with info_col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-header">
                <span>💊</span>
                <span class="info-card-title">{get_text('treatment_title')}</span>
            </div>
            <ul class="info-list">
        """, unsafe_allow_html=True)
        
        for treatment in disease_info.get('treatments', [])[:6]:
            st.markdown(f"<li>{treatment}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # All Predictions
    with st.expander(f"📊 {get_text('all_scores')}"):
        sorted_scores = sorted(
            result['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (class_name, score) in enumerate(sorted_scores):
            disease_name = get_text(f'disease_{class_name}')
            fill_class = 'prediction-bar-fill-top' if i == 0 else ''
            
            st.markdown(f"""
            <div class="prediction-bar-container">
                <div class="prediction-bar-label">
                    <span class="prediction-bar-name">{disease_name}</span>
                    <span class="prediction-bar-value">{score:.1f}%</span>
                </div>
                <div class="prediction-bar-bg">
                    <div class="prediction-bar-fill {fill_class}" style="width: {score}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Heatmap Section
    st.markdown(f"""
    <div class="heatmap-container">
        <div class="heatmap-title">🔥 {get_text('heatmap_title')}</div>
        <div class="heatmap-description">{get_text('heatmap_desc')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    hm_col1, hm_col2 = st.columns(2)
    
    with hm_col1:
        st.image(
            original_image.resize((224, 224)),
            caption=get_text('original_image'),
            use_column_width=True
        )
    
    with hm_col2:
        if heatmap_overlay is not None:
            st.image(
                heatmap_overlay,
                caption=get_text('heatmap_image'),
                use_column_width=True
            )
        else:
            st.info("Heatmap could not be generated for this image.")
    
    # Legend
    st.markdown("""
    <div class="heatmap-legend">
        <div class="heatmap-legend-item">
            <div class="legend-color legend-red"></div>
            <span>High Attention</span>
        </div>
        <div class="heatmap-legend-item">
            <div class="legend-color legend-yellow"></div>
            <span>Medium Attention</span>
        </div>
        <div class="heatmap-legend-item">
            <div class="legend-color legend-blue"></div>
            <span>Low Attention</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render footer with dentist finder and disclaimer"""
    # Find Dentist Button
    st.markdown(f"""
    <a href="https://www.google.com/maps/search/dentist+near+me" target="_blank" class="dentist-btn">
        📍 {get_text('find_dentist')}
    </a>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-box">
        <div class="disclaimer-title">
            ⚠️ {get_text('disclaimer_title')}
        </div>
        <div class="disclaimer-text">
            {get_text('disclaimer_text')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Load CSS
    load_css()
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Load model
    model = load_model()
    class_names = load_class_names()
    
    print(f"Loaded class names: {class_names}")
    
    # Check model
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure 'model/oral_disease_model.h5' exists.")
        return
    
    # Step 1: Risk Assessment
    st.markdown("---")
    risk_score = render_risk_assessment()
    
    # Step 2: Image Input
    st.markdown("---")
    should_analyze = render_image_input()
    
    # Step 3: Analysis - runs when button is clicked
    if should_analyze and st.session_state.current_image is not None:
        with st.spinner(get_text('analyzing')):
            # Preprocess and predict
            img_array = preprocess_image(st.session_state.current_image)
            result = predict_image(model, img_array, class_names)
            
            if result is not None:
                # Store results in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_done = True
                
                # Generate heatmap
                try:
                    heatmap_overlay = generate_heatmap_visualization(
                        st.session_state.current_image,
                        model,
                        result['index']
                    )
                    st.session_state.heatmap_image = heatmap_overlay
                except:
                    st.session_state.heatmap_image = None
    
    # Step 4: Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.analysis_result is not None:
        st.markdown("---")
        render_results(
            st.session_state.analysis_result,
            st.session_state.current_image,
            st.session_state.heatmap_image,
            risk_score
        )
    
    # Footer
    st.markdown("---")
    render_footer()

# ══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()