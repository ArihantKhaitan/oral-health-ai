"""
🦷 Oral Health AI - Early Detection Saves Lives
A comprehensive oral disease screening tool powered by AI
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
# CUSTOM CSS FOR BETTER UI
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    
    /* Result boxes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-danger {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .result-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        color: #e65100;
    }
    .result-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    
    /* Risk assessment card */
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    .risk-card h3 {
        color: white;
        margin-bottom: 15px;
    }
    
    /* Disclaimer - more visible */
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        font-size: 0.9rem;
        color: #856404;
        margin-top: 20px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #1E88E5;
    }
    
    /* Camera toggle button */
    .camera-btn {
        background-color: #1E88E5;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Confidence meter */
    .confidence-high { color: #d32f2f; font-weight: bold; font-size: 1.2rem; }
    .confidence-medium { color: #f57c00; font-weight: bold; font-size: 1.2rem; }
    .confidence-low { color: #388e3c; font-weight: bold; font-size: 1.2rem; }
    
    /* Analysis button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1E88E5, #00ACC1);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565C0, #00838F);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LANGUAGE SUPPORT
# ============================================
TRANSLATIONS = {
    'en': {
        'title': '🦷 Oral Health AI',
        'subtitle': 'Early Detection Saves Lives',
        'risk_title': '📋 Step 1: Risk Assessment',
        'risk_subtitle': 'Answer these questions to assess your oral cancer risk',
        'upload_title': '📸 Step 2: Upload or Capture Image',
        'analyze_btn': '🔍 Analyze Image',
        'results': '📊 Analysis Results',
        'confidence': 'Confidence',
        'recommendation': 'Recommendation',
        'tobacco_q': 'Do you use tobacco/gutkha?',
        'paan_q': 'Do you consume paan/betel?',
        'smoke_q': 'Do you smoke?',
        'alcohol_q': 'Do you consume alcohol regularly?',
        'find_dentist': '📍 Find Nearby Dentists',
        'disclaimer': '⚠️ IMPORTANT: This is a screening tool only, NOT a medical diagnosis. The AI may make errors. Please consult a qualified healthcare professional (dentist/doctor) for proper diagnosis and treatment. Do not delay seeking medical advice based on these results.',
        'high_risk': '⚠️ HIGH RISK - Please see a dentist within 48 hours',
        'medium_risk': '⚡ MEDIUM RISK - Schedule a dental checkup soon',
        'low_risk': '✅ LOW RISK - Maintain regular dental hygiene',
        'camera_on': '📷 Open Camera',
        'camera_off': '❌ Close Camera',
        'upload_option': '📁 Upload Image',
        'camera_option': '📷 Use Camera',
        'or_text': 'OR',
        'classes': {
            'Calculus': 'Calculus (Tartar)',
            'Caries': 'Dental Caries (Cavities)',
            'Gingivitis': 'Gingivitis (Gum Disease)',
            'Hypodontia': 'Hypodontia (Missing Teeth)',
            'Normal_Mouth': 'Normal/Healthy',
            'Oral_Cancer': '⚠️ Oral Cancer Signs',
            'Tooth Discoloration': 'Tooth Discoloration',
            'Ulcers': 'Mouth Ulcers'
        }
    },
    'hi': {
        'title': '🦷 मौखिक स्वास्थ्य AI',
        'subtitle': 'जल्दी पता लगाने से जीवन बचता है',
        'risk_title': '📋 चरण 1: जोखिम मूल्यांकन',
        'risk_subtitle': 'अपने मुंह के कैंसर के जोखिम का आकलन करने के लिए इन सवालों के जवाब दें',
        'upload_title': '📸 चरण 2: छवि अपलोड करें या कैप्चर करें',
        'analyze_btn': '🔍 छवि का विश्लेषण करें',
        'results': '📊 विश्लेषण परिणाम',
        'confidence': 'विश्वास स्तर',
        'recommendation': 'सिफारिश',
        'tobacco_q': 'क्या आप तंबाकू/गुटखा का उपयोग करते हैं?',
        'paan_q': 'क्या आप पान/सुपारी खाते हैं?',
        'smoke_q': 'क्या आप धूम्रपान करते हैं?',
        'alcohol_q': 'क्या आप नियमित रूप से शराब पीते हैं?',
        'find_dentist': '📍 नजदीकी दंत चिकित्सक खोजें',
        'disclaimer': '⚠️ महत्वपूर्ण: यह केवल एक स्क्रीनिंग टूल है, चिकित्सा निदान नहीं। AI गलतियाँ कर सकता है। उचित निदान और उपचार के लिए कृपया योग्य स्वास्थ्य पेशेवर (दंत चिकित्सक/डॉक्टर) से परामर्श करें।',
        'high_risk': '⚠️ उच्च जोखिम - कृपया 48 घंटे के भीतर दंत चिकित्सक से मिलें',
        'medium_risk': '⚡ मध्यम जोखिम - जल्द ही दंत जांच करवाएं',
        'low_risk': '✅ कम जोखिम - नियमित दंत स्वच्छता बनाए रखें',
        'camera_on': '📷 कैमरा खोलें',
        'camera_off': '❌ कैमरा बंद करें',
        'upload_option': '📁 छवि अपलोड करें',
        'camera_option': '📷 कैमरा उपयोग करें',
        'or_text': 'या',
        'classes': {
            'Calculus': 'कैलकुलस (टार्टर)',
            'Caries': 'दंत क्षय (कैविटी)',
            'Gingivitis': 'मसूड़े की सूजन',
            'Hypodontia': 'हाइपोडोंटिया (दांत गायब)',
            'Normal_Mouth': 'सामान्य/स्वस्थ',
            'Oral_Cancer': '⚠️ मुंह के कैंसर के संकेत',
            'Tooth Discoloration': 'दांतों का मलिनकिरण',
            'Ulcers': 'मुंह के छाले'
        }
    }
}

# Risk levels for each class
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

RECOMMENDATIONS = {
    'en': {
        'Oral_Cancer': '🚨 URGENT: Potential signs of oral cancer detected. Please consult an oncologist or oral surgeon IMMEDIATELY. Early detection saves lives!',
        'Ulcers': 'Mouth ulcers detected. If persistent for more than 2 weeks, consult a dentist. Avoid spicy foods.',
        'Gingivitis': 'Signs of gum disease detected. Improve brushing technique, use antibacterial mouthwash, and consider professional cleaning.',
        'Caries': 'Dental cavities detected. Visit a dentist for filling treatment. Reduce sugar intake.',
        'Calculus': 'Tartar buildup detected. Schedule a professional dental cleaning (scaling).',
        'Tooth Discoloration': 'Tooth staining observed. Consider professional whitening or check for underlying issues.',
        'Hypodontia': 'Missing teeth condition noted. Consult a dentist for replacement options (implants/bridges).',
        'Normal_Mouth': '✅ Your oral health appears normal! Continue regular brushing twice daily and flossing.'
    },
    'hi': {
        'Oral_Cancer': '🚨 तत्काल: मुंह के कैंसर के संभावित लक्षण पाए गए। कृपया तुरंत ऑन्कोलॉजिस्ट या ओरल सर्जन से परामर्श करें।',
        'Ulcers': 'मुंह के छाले पाए गए। यदि 2 सप्ताह से अधिक समय तक रहें तो दंत चिकित्सक से मिलें।',
        'Gingivitis': 'मसूड़ों की बीमारी के लक्षण। ब्रश करने की तकनीक में सुधार करें।',
        'Caries': 'दांतों में कैविटी पाई गई। फिलिंग के लिए दंत चिकित्सक से मिलें।',
        'Calculus': 'टार्टर जमाव पाया गया। पेशेवर दंत सफाई करवाएं।',
        'Tooth Discoloration': 'दांतों पर दाग देखे गए। पेशेवर व्हाइटनिंग पर विचार करें।',
        'Hypodontia': 'दांत गायब हैं। प्रतिस्थापन विकल्पों के लिए दंत चिकित्सक से परामर्श करें।',
        'Normal_Mouth': '✅ आपका मौखिक स्वास्थ्य सामान्य दिखता है! दिन में दो बार ब्रश करना जारी रखें।'
    }
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
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=None
            )
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
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
    else:
        return None

@st.cache_data
def load_class_names():
    """Load class names from JSON"""
    json_path = 'model/class_names.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['class_names']
    else:
        return ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 
                'Normal_Mouth', 'Oral_Cancer', 'Tooth Discoloration', 'Ulcers']

# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================
# GRADCAM VISUALIZATION
# ============================================
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
            
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except:
        return None

def overlay_gradcam(img, heatmap, alpha=0.4):
    """Overlay GradCAM heatmap on image"""
    if heatmap is None:
        return img
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap * alpha + img * (1 - alpha)
    superimposed = np.uint8(superimposed)
    
    return superimposed

# ============================================
# MAIN APP
# ============================================
def main():
    # Initialize session state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = 0
    
    # Sidebar - Language Selection & Info
    with st.sidebar:
        st.markdown("### 🌐 Language / भाषा")
        lang = st.selectbox("", ["English", "हिंदी"], label_visibility="collapsed")
        lang_code = 'en' if lang == "English" else 'hi'
        t = TRANSLATIONS[lang_code]
        
        st.markdown("---")
        st.markdown("### 📖 About This Tool")
        st.markdown("""
        This AI detects **8 oral conditions**:
        - 🔴 Oral Cancer Signs
        - 🟠 Mouth Ulcers
        - 🟠 Gingivitis
        - 🟠 Dental Caries
        - 🟢 Calculus (Tartar)
        - 🟢 Tooth Discoloration
        - 🟢 Hypodontia
        - 🟢 Normal/Healthy
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        st.metric("Accuracy", "86.96%")
        st.metric("Cancer Detection", "91% precision")
        st.metric("Training Data", "10,860 images")
    
    # Main Header
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t["subtitle"]}</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    class_names = load_class_names()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure model files are in the 'model/' directory.")
        st.info("Required files: `model/oral_disease_model.h5` and `model/class_names.json`")
        return
    
    # ========== STEP 1: RISK ASSESSMENT ==========
    st.markdown(f'<p class="section-header">{t["risk_title"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<small>{t["risk_subtitle"]}</small>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        tobacco = st.checkbox(f"🚬 {t['tobacco_q']}")
        paan = st.checkbox(f"🌿 {t['paan_q']}")
    with col2:
        smoke = st.checkbox(f"🔥 {t['smoke_q']}")
        alcohol = st.checkbox(f"🍺 {t['alcohol_q']}")
    
    risk_factors = sum([tobacco, paan, smoke, alcohol])
    st.session_state.risk_score = risk_factors
    
    if risk_factors > 0:
        if risk_factors >= 3:
            st.error(f"🚨 **HIGH RISK PROFILE**: You have {risk_factors}/4 risk factors. Regular oral cancer screening is STRONGLY recommended!")
        elif risk_factors >= 1:
            st.warning(f"⚠️ **MODERATE RISK**: You have {risk_factors}/4 risk factor(s). Regular dental checkups advised.")
    else:
        st.success("✅ **LOW RISK PROFILE**: No major risk factors identified. Keep up good oral hygiene!")
    
    st.markdown("---")
    
    # ========== STEP 2: IMAGE INPUT ==========
    st.markdown(f'<p class="section-header">{t["upload_title"]}</p>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        [t['upload_option'], t['camera_option']],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    image_source = None
    
    if input_method == t['upload_option']:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image of your mouth/teeth",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image_source = uploaded_file
            
    else:
        # Camera input with toggle
        if st.button(t['camera_off'] if st.session_state.camera_on else t['camera_on']):
            st.session_state.camera_on = not st.session_state.camera_on
            st.rerun()
        
        if st.session_state.camera_on:
            camera_input = st.camera_input("Take a photo of your mouth")
            if camera_input:
                image_source = camera_input
    
    # ========== ANALYSIS ==========
    if image_source:
        image = Image.open(image_source)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Your Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown(f"### {t['results']}")
            
            with st.spinner("🔍 Analyzing..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img, verbose=0)
                
                # Get top prediction
                pred_idx = np.argmax(predictions[0])
                pred_class = class_names[pred_idx]
                confidence = predictions[0][pred_idx] * 100
                
                # Determine risk level
                risk_level = RISK_LEVELS.get(pred_class, 'low')
                
                # Combine with lifestyle risk
                if st.session_state.risk_score >= 2 and risk_level == 'medium':
                    risk_level = 'high'
                
                # Display result
                if risk_level == 'high':
                    box_class = 'result-danger'
                    risk_text = t['high_risk']
                elif risk_level == 'medium':
                    box_class = 'result-warning'
                    risk_text = t['medium_risk']
                else:
                    box_class = 'result-success'
                    risk_text = t['low_risk']
                
                translated_class = t['classes'].get(pred_class, pred_class)
                
                st.markdown(f"""
                <div class="result-box {box_class}">
                    <h3>{translated_class}</h3>
                    <p><strong>{t['confidence']}:</strong> {confidence:.1f}%</p>
                    <p><strong>{t['recommendation']}:</strong><br>{RECOMMENDATIONS[lang_code].get(pred_class, '')}</p>
                    <hr>
                    <p><strong>{risk_text}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all predictions in expander
            with st.expander("📊 See all predictions"):
                for cls, prob in sorted(zip(class_names, predictions[0]), key=lambda x: x[1], reverse=True):
                    translated = t['classes'].get(cls, cls)
                    st.progress(float(prob), text=f"{translated}: {prob*100:.1f}%")
        
        # GradCAM visualization
        st.markdown("### 🔥 AI Focus Area")
        st.caption("This heatmap shows where the AI looked to make its prediction (red = high attention)")
        
        try:
            heatmap = make_gradcam_heatmap(processed_img, model, pred_idx)
            if heatmap is not None:
                img_array = np.array(image.resize((224, 224)))
                gradcam_img = overlay_gradcam(img_array, heatmap)
                st.image(gradcam_img, use_container_width=True)
            else:
                st.info("GradCAM visualization not available for this prediction.")
        except:
            st.info("GradCAM visualization not available.")
    
    st.markdown("---")
    
    # Find Dentist Button
    st.link_button(
        t['find_dentist'],
        "https://www.google.com/maps/search/dentist+near+me",
        use_container_width=True
    )
    
    # Disclaimer - More visible
    st.markdown(f'<div class="disclaimer">{t["disclaimer"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()