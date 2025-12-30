"""
🦷 Oral Health AI - Early Detection Saves Lives
A comprehensive oral disease screening tool powered by AI

Features:
- 8-class oral disease detection
- GradCAM visualization
- Risk assessment questionnaire  
- Multi-language support (English + Hindi)
- Disease information and treatment suggestions
- Professional medical-grade UI
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Oral Health AI - मौखिक स्वास्थ्य AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - PROFESSIONAL MEDICAL UI
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Step indicator */
    .step-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-danger {
        background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
    }
    .result-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 6px solid #f59e0b;
    }
    .result-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #22c55e;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .result-danger .result-title { color: #dc2626; }
    .result-warning .result-title { color: #d97706; }
    .result-success .result-title { color: #16a34a; }
    
    /* Confidence meter */
    .confidence-container {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    .confidence-label {
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
    }
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .conf-high { color: #dc2626; }
    .conf-medium { color: #d97706; }
    .conf-low { color: #16a34a; }
    
    /* Disease info card */
    .disease-info {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    .disease-info h4 {
        color: #334155;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    .disease-info ul {
        margin: 0;
        padding-left: 20px;
        color: #64748b;
    }
    .disease-info li {
        margin: 5px 0;
    }
    
    /* Risk assessment card */
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .risk-card label {
        color: white !important;
    }
    
    /* Disclaimer - VERY VISIBLE */
    .disclaimer-box {
        background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
        border: 2px solid #eab308;
        border-radius: 12px;
        padding: 20px;
        margin: 25px 0;
        box-shadow: 0 4px 15px rgba(234, 179, 8, 0.3);
    }
    .disclaimer-box h4 {
        color: #a16207;
        margin: 0 0 10px 0;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .disclaimer-box p {
        color: #854d0e;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Image container */
    .image-container {
        border: 2px dashed #cbd5e1;
        border-radius: 15px;
        padding: 10px;
        background: #f8fafc;
    }
    
    /* Heatmap section */
    .heatmap-section {
        background: #fefce8;
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        border: 1px solid #fde047;
    }
    
    /* Find dentist button */
    .dentist-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        text-align: center;
        width: 100%;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .dentist-btn:hover {
        transform: translateY(-2px);
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar-metric {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
    }
    .sidebar-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0369a1;
    }
    .sidebar-metric-label {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* Risk level badges */
    .risk-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px 0;
    }
    .risk-high { background: #fee2e2; color: #dc2626; }
    .risk-medium { background: #fef3c7; color: #d97706; }
    .risk-low { background: #dcfce7; color: #16a34a; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Checkbox styling */
    .stCheckbox {
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DISEASE INFORMATION DATABASE
# ============================================
DISEASE_INFO = {
    'en': {
        'Oral_Cancer': {
            'name': '⚠️ Oral Cancer Signs Detected',
            'description': 'Oral cancer is a serious condition where malignant cells form in the tissues of the mouth. Early detection is CRITICAL for survival.',
            'symptoms': [
                'White or red patches in mouth',
                'Non-healing sores or ulcers (>2 weeks)',
                'Lumps or thickening in cheek',
                'Difficulty swallowing or chewing',
                'Numbness in tongue or mouth',
                'Unexplained bleeding'
            ],
            'causes': ['Tobacco use (smoking, chewing)', 'Excessive alcohol', 'HPV infection', 'Sun exposure (lip cancer)', 'Poor nutrition'],
            'treatment': ['Surgical removal', 'Radiation therapy', 'Chemotherapy', 'Targeted drug therapy'],
            'urgency': 'IMMEDIATE - See an oncologist within 24-48 hours'
        },
        'Ulcers': {
            'name': 'Mouth Ulcers (Canker Sores)',
            'description': 'Mouth ulcers are painful sores that appear inside the mouth. Most heal within 1-2 weeks.',
            'symptoms': [
                'Painful round/oval sores',
                'White/yellow center with red border',
                'Burning sensation before appearance',
                'Difficulty eating spicy/acidic foods'
            ],
            'causes': ['Stress', 'Minor injuries', 'Acidic foods', 'Vitamin deficiencies (B12, iron)', 'Hormonal changes'],
            'treatment': ['Antiseptic mouthwash', 'Pain-relieving gels', 'Avoid spicy foods', 'Vitamin supplements', 'Salt water rinse'],
            'urgency': 'Monitor - See dentist if persists >2 weeks'
        },
        'Gingivitis': {
            'name': 'Gingivitis (Gum Disease)',
            'description': 'Gingivitis is inflammation of the gums, usually caused by bacterial infection. If untreated, it can lead to periodontitis.',
            'symptoms': [
                'Red, swollen gums',
                'Bleeding while brushing/flossing',
                'Bad breath (halitosis)',
                'Receding gums',
                'Tender gums'
            ],
            'causes': ['Poor oral hygiene', 'Plaque buildup', 'Smoking', 'Diabetes', 'Certain medications'],
            'treatment': ['Professional cleaning', 'Improved brushing technique', 'Antibacterial mouthwash', 'Regular flossing', 'Dental checkups'],
            'urgency': 'Schedule dental visit within 2 weeks'
        },
        'Caries': {
            'name': 'Dental Caries (Cavities)',
            'description': 'Cavities are permanently damaged areas in teeth that develop into tiny holes. They are among the most common health problems.',
            'symptoms': [
                'Toothache or sensitivity',
                'Pain when eating sweet/hot/cold',
                'Visible holes in teeth',
                'Brown/black staining',
                'Bad breath'
            ],
            'causes': ['Frequent snacking', 'Sugary drinks', 'Poor brushing', 'Dry mouth', 'Bacteria in mouth'],
            'treatment': ['Dental fillings', 'Crowns (severe cases)', 'Root canal (deep decay)', 'Fluoride treatments', 'Tooth extraction (extreme)'],
            'urgency': 'Schedule dental visit within 1-2 weeks'
        },
        'Calculus': {
            'name': 'Calculus (Tartar)',
            'description': 'Calculus is hardite tartar buildup on teeth. It cannot be removed by regular brushing and requires professional cleaning.',
            'symptoms': [
                'Yellow/brown deposits on teeth',
                'Rough feeling on teeth',
                'Bad breath',
                'Gum irritation',
                'Bleeding gums'
            ],
            'causes': ['Poor oral hygiene', 'Not flossing', 'Smoking', 'Dry mouth', 'Diet high in sugar/starch'],
            'treatment': ['Professional scaling', 'Root planing', 'Improved oral hygiene', 'Regular dental cleanings', 'Electric toothbrush'],
            'urgency': 'Schedule dental cleaning within 1 month'
        },
        'Tooth Discoloration': {
            'name': 'Tooth Discoloration',
            'description': 'Tooth discoloration refers to staining or changes in tooth color. It can be extrinsic (surface) or intrinsic (internal).',
            'symptoms': [
                'Yellow or brown teeth',
                'White spots on teeth',
                'Gray or dark teeth',
                'Uneven coloring'
            ],
            'causes': ['Coffee, tea, wine', 'Tobacco use', 'Poor hygiene', 'Medications', 'Aging', 'Fluorosis'],
            'treatment': ['Professional whitening', 'Whitening toothpaste', 'Dental veneers', 'Bonding', 'Better oral hygiene'],
            'urgency': 'Non-urgent - Cosmetic concern'
        },
        'Hypodontia': {
            'name': 'Hypodontia (Missing Teeth)',
            'description': 'Hypodontia is a condition where one or more teeth fail to develop. It can affect appearance and dental function.',
            'symptoms': [
                'Gaps in teeth',
                'Difficulty chewing',
                'Speech problems',
                'Jawbone issues',
                'Self-esteem concerns'
            ],
            'causes': ['Genetic factors', 'Developmental issues', 'Trauma', 'Infection during development'],
            'treatment': ['Dental implants', 'Bridges', 'Partial dentures', 'Orthodontic treatment', 'Space maintainers'],
            'urgency': 'Non-urgent - Consult dentist for options'
        },
        'Normal_Mouth': {
            'name': '✅ Healthy Mouth',
            'description': 'Your oral health appears normal! Continue maintaining good oral hygiene practices.',
            'symptoms': [
                'Pink, firm gums',
                'No bleeding when brushing',
                'Fresh breath',
                'Clean teeth',
                'No pain or sensitivity'
            ],
            'causes': [],
            'treatment': ['Continue brushing twice daily', 'Floss daily', 'Regular dental checkups', 'Balanced diet', 'Limit sugary foods'],
            'urgency': 'Routine checkup every 6 months'
        }
    },
    'hi': {
        'Oral_Cancer': {
            'name': '⚠️ मुंह के कैंसर के संकेत',
            'description': 'मुंह का कैंसर एक गंभीर स्थिति है। जल्दी पता लगाना जीवन बचा सकता है।',
            'symptoms': ['मुंह में सफेद या लाल धब्बे', 'न भरने वाले घाव', 'गाल में गांठ', 'निगलने में कठिनाई'],
            'causes': ['तंबाकू का उपयोग', 'शराब', 'HPV संक्रमण'],
            'treatment': ['सर्जरी', 'रेडिएशन थेरेपी', 'कीमोथेरेपी'],
            'urgency': 'तत्काल - 24-48 घंटों के भीतर ऑन्कोलॉजिस्ट से मिलें'
        },
        'Ulcers': {
            'name': 'मुंह के छाले',
            'description': 'मुंह के छाले दर्दनाक घाव हैं जो आमतौर पर 1-2 सप्ताह में ठीक हो जाते हैं।',
            'symptoms': ['दर्दनाक गोल घाव', 'जलन', 'खाने में कठिनाई'],
            'causes': ['तनाव', 'विटामिन की कमी', 'मसालेदार भोजन'],
            'treatment': ['एंटीसेप्टिक माउथवॉश', 'दर्द निवारक जेल', 'नमक के पानी से गरारे'],
            'urgency': 'निगरानी करें - 2 सप्ताह से अधिक रहे तो डॉक्टर से मिलें'
        },
        'Normal_Mouth': {
            'name': '✅ स्वस्थ मुंह',
            'description': 'आपका मौखिक स्वास्थ्य सामान्य दिखता है!',
            'symptoms': ['गुलाबी मसूड़े', 'कोई रक्तस्राव नहीं', 'ताजी सांस'],
            'causes': [],
            'treatment': ['दिन में दो बार ब्रश करें', 'नियमित जांच करवाएं'],
            'urgency': 'हर 6 महीने में नियमित जांच'
        }
    }
}

# Risk levels
RISK_LEVELS = {
    'Oral_Cancer': 'high',
    'Ulcers': 'medium',
    'Gingivitis': 'medium',
    'Caries': 'medium',
    'Calculus': 'low',
    'Tooth Discoloration': 'low',
    'Hypodontia': 'low',
    'Normal_Mouth': 'low'
}

# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'model/oral_disease_model.h5'
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            try:
                from tensorflow.keras.applications import EfficientNetB0
                from tensorflow.keras import layers, Model
                
                base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
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
                st.error(f"Error loading model: {e2}")
                return None
    return None

@st.cache_data
def load_class_names():
    """Load class names"""
    json_path = 'model/class_names.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)['class_names']
    return ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Normal_Mouth', 'Oral_Cancer', 'Tooth Discoloration', 'Ulcers']

# ============================================
# IMAGE PROCESSING
# ============================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate GradCAM heatmap"""
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer.name
                break
        if not last_conv_layer:
            return None
            
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
        
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    if heatmap is None:
        return img
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(heatmap * alpha + img * (1 - alpha))

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Session state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌐 Language / भाषा")
        lang = st.selectbox("Select Language", ["English", "हिंदी"], label_visibility="collapsed")
        lang_code = 'en' if lang == "English" else 'hi'
        
        st.markdown("---")
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
            <div class="sidebar-metric-label">Cancer Detection Precision</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">10,860</div>
            <div class="sidebar-metric-label">Training Images</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Detectable Conditions")
        conditions = [
            ("🔴", "Oral Cancer"),
            ("🟠", "Mouth Ulcers"),
            ("🟠", "Gingivitis"),
            ("🟠", "Dental Caries"),
            ("🟢", "Calculus"),
            ("🟢", "Tooth Discoloration"),
            ("🟢", "Hypodontia"),
            ("🟢", "Normal/Healthy")
        ]
        for icon, name in conditions:
            st.markdown(f"{icon} {name}")
    
    # Main content
    st.markdown('<h1 class="main-header">🦷 Oral Health AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Oral Disease Screening • Early Detection Saves Lives</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    class_names = load_class_names()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check model files.")
        return
    
    # ===== STEP 1: RISK ASSESSMENT =====
    st.markdown('<p class="section-header"><span class="step-badge">Step 1</span> Risk Assessment</p>', unsafe_allow_html=True)
    
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        tobacco = st.checkbox("🚬 Do you use tobacco/gutkha?")
        paan = st.checkbox("🌿 Do you consume paan/betel?")
    with risk_col2:
        smoke = st.checkbox("🔥 Do you smoke?")
        alcohol = st.checkbox("🍺 Do you consume alcohol regularly?")
    
    risk_factors = sum([tobacco, paan, smoke, alcohol])
    
    if risk_factors >= 3:
        st.markdown('<span class="risk-badge risk-high">🚨 HIGH RISK - {} of 4 risk factors</span>'.format(risk_factors), unsafe_allow_html=True)
        st.error("You have multiple risk factors for oral cancer. Regular screening is STRONGLY recommended!")
    elif risk_factors >= 1:
        st.markdown('<span class="risk-badge risk-medium">⚠️ MODERATE RISK - {} of 4 risk factors</span>'.format(risk_factors), unsafe_allow_html=True)
        st.warning("You have some risk factors. Consider regular dental checkups.")
    else:
        st.markdown('<span class="risk-badge risk-low">✅ LOW RISK - No major risk factors</span>', unsafe_allow_html=True)
        st.success("Great! No major risk factors identified. Maintain good oral hygiene!")
    
    st.markdown("---")
    
    # ===== STEP 2: IMAGE INPUT =====
    st.markdown('<p class="section-header"><span class="step-badge">Step 2</span> Upload or Capture Image</p>', unsafe_allow_html=True)
    
    input_tab1, input_tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])
    
    image_source = None
    
    with input_tab1:
        uploaded_file = st.file_uploader("Upload a clear image of your mouth/teeth", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image_source = uploaded_file
    
    with input_tab2:
        st.info("📸 Position your camera to capture a clear image of the affected area")
        camera_input = st.camera_input("Take a photo")
        if camera_input:
            image_source = camera_input
    
    # ===== ANALYSIS =====
    if image_source:
        image = Image.open(image_source)
        
        st.markdown("---")
        st.markdown('<p class="section-header"><span class="step-badge">Step 3</span> Analysis Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📷 Uploaded Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.spinner("🔍 Analyzing image with AI..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img, verbose=0)
                
                pred_idx = np.argmax(predictions[0])
                pred_class = class_names[pred_idx]
                confidence = predictions[0][pred_idx] * 100
                
                risk_level = RISK_LEVELS.get(pred_class, 'low')
                if risk_factors >= 2 and risk_level == 'medium':
                    risk_level = 'high'
                
                # Get disease info
                disease_data = DISEASE_INFO.get(lang_code, DISEASE_INFO['en']).get(pred_class, DISEASE_INFO['en'].get(pred_class, {}))
                
                # Result card
                if risk_level == 'high':
                    card_class = 'result-danger'
                elif risk_level == 'medium':
                    card_class = 'result-warning'
                else:
                    card_class = 'result-success'
                
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-title">{disease_data.get('name', pred_class)}</div>
                    <div class="confidence-container">
                        <div class="confidence-label">AI Confidence Score</div>
                        <div class="confidence-value {'conf-high' if confidence > 80 else 'conf-medium' if confidence > 50 else 'conf-low'}">{confidence:.1f}%</div>
                    </div>
                    <p>{disease_data.get('description', '')}</p>
                    <p><strong>⏰ Urgency:</strong> {disease_data.get('urgency', 'Consult a dentist')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Disease Information
        st.markdown("#### 📋 Detailed Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.markdown('<div class="disease-info">', unsafe_allow_html=True)
            st.markdown("##### 🔍 Symptoms")
            symptoms = disease_data.get('symptoms', [])
            for s in symptoms[:5]:
                st.markdown(f"• {s}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with info_col2:
            st.markdown('<div class="disease-info">', unsafe_allow_html=True)
            st.markdown("##### ⚡ Common Causes")
            causes = disease_data.get('causes', [])
            for c in causes[:5]:
                st.markdown(f"• {c}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with info_col3:
            st.markdown('<div class="disease-info">', unsafe_allow_html=True)
            st.markdown("##### 💊 Treatment Options")
            treatments = disease_data.get('treatment', [])
            for t in treatments[:5]:
                st.markdown(f"• {t}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # All predictions
        with st.expander("📊 View All Prediction Scores"):
            for cls, prob in sorted(zip(class_names, predictions[0]), key=lambda x: x[1], reverse=True):
                st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
        
        # GradCAM Heatmap
        st.markdown("#### 🔥 AI Attention Heatmap")
        st.caption("This shows where the AI focused to make its prediction (red = high attention)")
        
        try:
            heatmap = make_gradcam_heatmap(processed_img, model, pred_idx)
            if heatmap is not None:
                img_array = np.array(image.resize((224, 224)))
                gradcam_img = overlay_gradcam(img_array, heatmap)
                
                hm_col1, hm_col2 = st.columns(2)
                with hm_col1:
                    st.image(image.resize((224, 224)), caption="Original", use_column_width=True)
                with hm_col2:
                    st.image(gradcam_img, caption="AI Focus Areas", use_column_width=True)
        except Exception as e:
            st.info("Heatmap visualization not available for this image.")
    
    st.markdown("---")
    
    # Find Dentist
    st.markdown("#### 🏥 Need Professional Help?")
    st.markdown(
        '<a href="https://www.google.com/maps/search/dentist+near+me" target="_blank" class="dentist-btn">📍 Find Dentists Near You</a>',
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <h4>⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
        <p>
            This AI tool is for <strong>SCREENING PURPOSES ONLY</strong> and is NOT a substitute for professional medical diagnosis. 
            The AI model may make errors and has an accuracy of approximately 87%. 
            <strong>Always consult a qualified healthcare professional</strong> (dentist, oral surgeon, or doctor) for proper diagnosis and treatment.
            Do not delay seeking medical attention based on results from this tool. If you notice any concerning symptoms, 
            please visit a healthcare provider immediately.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()