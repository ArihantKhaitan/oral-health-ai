"""
🦷 Oral Health AI - Early Detection Saves Lives
A comprehensive oral disease screening tool powered by AI

Features:
- 8-class oral disease detection
- GradCAM visualization (shows WHERE issues detected)
- Risk assessment questionnaire
- Multi-language support (English + Hindi)
- Mobile-responsive design

Author: Arihant
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import os

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-danger {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .result-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .result-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .confidence-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #388e3c;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        font-size: 0.85rem;
        color: #666;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
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
        'upload_prompt': 'Upload an image of your mouth/teeth',
        'take_photo': 'Or take a photo',
        'analyze_btn': '🔍 Analyze Image',
        'results': 'Analysis Results',
        'confidence': 'Confidence',
        'recommendation': 'Recommendation',
        'risk_assessment': 'Risk Assessment',
        'tobacco_q': 'Do you use tobacco/gutkha?',
        'paan_q': 'Do you consume paan/betel?',
        'smoke_q': 'Do you smoke?',
        'alcohol_q': 'Do you consume alcohol regularly?',
        'find_dentist': '📍 Find Nearby Dentists',
        'disclaimer': '⚠️ Disclaimer: This is a screening tool only, not a medical diagnosis. Please consult a qualified healthcare professional for proper diagnosis and treatment.',
        'high_risk': '⚠️ HIGH RISK - Please see a dentist within 48 hours',
        'medium_risk': '⚡ MEDIUM RISK - Schedule a dental checkup soon',
        'low_risk': '✅ LOW RISK - Maintain regular dental hygiene',
        'classes': {
            'Calculus': 'Calculus (Tartar)',
            'Caries': 'Dental Caries (Cavities)',
            'Gingivitis': 'Gingivitis (Gum Disease)',
            'Hypodontia': 'Hypodontia (Missing Teeth)',
            'Normal_Mouth': 'Normal/Healthy',
            'Oral_Cancer': 'Oral Cancer Signs',
            'Tooth Discoloration': 'Tooth Discoloration',
            'Ulcers': 'Mouth Ulcers'
        }
    },
    'hi': {
        'title': '🦷 मौखिक स्वास्थ्य AI',
        'subtitle': 'जल्दी पता लगाने से जीवन बचता है',
        'upload_prompt': 'अपने मुंह/दांतों की तस्वीर अपलोड करें',
        'take_photo': 'या फोटो लें',
        'analyze_btn': '🔍 छवि का विश्लेषण करें',
        'results': 'विश्लेषण परिणाम',
        'confidence': 'विश्वास स्तर',
        'recommendation': 'सिफारिश',
        'risk_assessment': 'जोखिम मूल्यांकन',
        'tobacco_q': 'क्या आप तंबाकू/गुटखा का उपयोग करते हैं?',
        'paan_q': 'क्या आप पान/सुपारी खाते हैं?',
        'smoke_q': 'क्या आप धूम्रपान करते हैं?',
        'alcohol_q': 'क्या आप नियमित रूप से शराब पीते हैं?',
        'find_dentist': '📍 नजदीकी दंत चिकित्सक खोजें',
        'disclaimer': '⚠️ अस्वीकरण: यह केवल एक स्क्रीनिंग टूल है, चिकित्सा निदान नहीं। उचित निदान और उपचार के लिए कृपया योग्य स्वास्थ्य पेशेवर से परामर्श करें।',
        'high_risk': '⚠️ उच्च जोखिम - कृपया 48 घंटे के भीतर दंत चिकित्सक से मिलें',
        'medium_risk': '⚡ मध्यम जोखिम - जल्द ही दंत जांच करवाएं',
        'low_risk': '✅ कम जोखिम - नियमित दंत स्वच्छता बनाए रखें',
        'classes': {
            'Calculus': 'कैलकुलस (टार्टर)',
            'Caries': 'दंत क्षय (कैविटी)',
            'Gingivitis': 'मसूड़े की सूजन',
            'Hypodontia': 'हाइपोडोंटिया (दांत गायब)',
            'Normal_Mouth': 'सामान्य/स्वस्थ',
            'Oral_Cancer': 'मुंह के कैंसर के संकेत',
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
        'Oral_Cancer': 'Potential signs of oral cancer detected. Please consult an oncologist or oral surgeon IMMEDIATELY.',
        'Ulcers': 'Mouth ulcers detected. If persistent for more than 2 weeks, consult a dentist.',
        'Gingivitis': 'Signs of gum disease. Improve brushing technique and consider professional cleaning.',
        'Caries': 'Dental cavities detected. Visit a dentist for filling treatment.',
        'Calculus': 'Tartar buildup detected. Schedule a professional dental cleaning.',
        'Tooth Discoloration': 'Tooth staining observed. Consider professional whitening or check for underlying issues.',
        'Hypodontia': 'Missing teeth condition. Consult a dentist for replacement options.',
        'Normal_Mouth': 'Your oral health appears normal. Continue regular dental hygiene practices.'
    },
    'hi': {
        'Oral_Cancer': 'मुंह के कैंसर के संभावित लक्षण पाए गए। कृपया तुरंत ऑन्कोलॉजिस्ट या ओरल सर्जन से परामर्श करें।',
        'Ulcers': 'मुंह के छाले पाए गए। यदि 2 सप्ताह से अधिक समय तक रहें तो दंत चिकित्सक से मिलें।',
        'Gingivitis': 'मसूड़ों की बीमारी के लक्षण। ब्रश करने की तकनीक में सुधार करें।',
        'Caries': 'दांतों में कैविटी पाई गई। फिलिंग के लिए दंत चिकित्सक से मिलें।',
        'Calculus': 'टार्टर जमाव पाया गया। पेशेवर दंत सफाई करवाएं।',
        'Tooth Discoloration': 'दांतों पर दाग देखे गए। पेशेवर व्हाइटनिंग पर विचार करें।',
        'Hypodontia': 'दांत गायब हैं। प्रतिस्थापन विकल्पों के लिए दंत चिकित्सक से परामर्श करें।',
        'Normal_Mouth': 'आपका मौखिक स्वास्थ्य सामान्य दिखता है। नियमित दंत स्वच्छता जारी रखें।'
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
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"Model not found at {model_path}")
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
        # Default class names
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
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate GradCAM heatmap"""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
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
    except Exception as e:
        st.warning(f"Could not generate GradCAM: {e}")
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
    # Sidebar - Language Selection
    with st.sidebar:
        st.markdown("### 🌐 Language / भाषा")
        lang = st.selectbox("", ["English", "हिंदी"], label_visibility="collapsed")
        lang_code = 'en' if lang == "English" else 'hi'
        t = TRANSLATIONS[lang_code]
        
        st.markdown("---")
        st.markdown("### 📊 About This Tool")
        st.markdown("""
        This AI tool can detect **8 types** of oral conditions:
        - Oral Cancer Signs
        - Mouth Ulcers
        - Gingivitis
        - Dental Caries
        - Calculus (Tartar)
        - Tooth Discoloration
        - Hypodontia
        - Normal/Healthy
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        st.metric("Test Accuracy", "86.96%")
        st.metric("Oral Cancer Precision", "91%")
        st.metric("Training Images", "10,860")
    
    # Main content
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t["subtitle"]}</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    class_names = load_class_names()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure model files are in the 'model/' directory.")
        st.info("Required files: `model/oral_disease_model.h5` and `model/class_names.json`")
        return
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 📸 {t['upload_prompt']}")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        # Camera input
        st.markdown(f"### 📷 {t['take_photo']}")
        camera_input = st.camera_input("", label_visibility="collapsed")
        
        # Use camera input if no file uploaded
        image_source = uploaded_file if uploaded_file else camera_input
        
        if image_source:
            image = Image.open(image_source)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if image_source:
            st.markdown(f"### 🔍 {t['results']}")
            
            with st.spinner("Analyzing..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img, verbose=0)
                
                # Get top prediction
                pred_idx = np.argmax(predictions[0])
                pred_class = class_names[pred_idx]
                confidence = predictions[0][pred_idx] * 100
                
                # Determine risk level
                risk_level = RISK_LEVELS.get(pred_class, 'low')
                
                # Display result with appropriate styling
                if risk_level == 'high':
                    box_class = 'result-danger'
                    conf_class = 'confidence-high'
                    risk_text = t['high_risk']
                elif risk_level == 'medium':
                    box_class = 'result-warning'
                    conf_class = 'confidence-medium'
                    risk_text = t['medium_risk']
                else:
                    box_class = 'result-success'
                    conf_class = 'confidence-low'
                    risk_text = t['low_risk']
                
                # Result box
                translated_class = t['classes'].get(pred_class, pred_class)
                st.markdown(f"""
                <div class="result-box {box_class}">
                    <h3>{translated_class}</h3>
                    <p><strong>{t['confidence']}:</strong> <span class="{conf_class}">{confidence:.1f}%</span></p>
                    <p><strong>{t['recommendation']}:</strong> {RECOMMENDATIONS[lang_code].get(pred_class, '')}</p>
                    <p>{risk_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all predictions
                with st.expander("📊 All Predictions"):
                    for i, (cls, prob) in enumerate(sorted(zip(class_names, predictions[0]), key=lambda x: x[1], reverse=True)):
                        translated = t['classes'].get(cls, cls)
                        st.progress(float(prob), text=f"{translated}: {prob*100:.1f}%")
                
                # GradCAM visualization
                st.markdown("### 🔥 AI Focus Area (GradCAM)")
                try:
                    # Find last conv layer
                    last_conv_layer = None
                    for layer in reversed(model.layers):
                        if 'conv' in layer.name.lower():
                            last_conv_layer = layer.name
                            break
                    
                    if last_conv_layer:
                        heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer, pred_idx)
                        if heatmap is not None:
                            img_array = np.array(image.resize((224, 224)))
                            gradcam_img = overlay_gradcam(img_array, heatmap)
                            st.image(gradcam_img, caption="Areas of AI Focus (Red = High Attention)", use_container_width=True)
                except Exception as e:
                    st.info("GradCAM visualization not available for this model architecture.")
    
    # Risk Assessment Section
    st.markdown("---")
    st.markdown(f"### 📋 {t['risk_assessment']}")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        tobacco = st.checkbox(t['tobacco_q'])
        paan = st.checkbox(t['paan_q'])
    
    with risk_col2:
        smoke = st.checkbox(t['smoke_q'])
        alcohol = st.checkbox(t['alcohol_q'])
    
    risk_factors = sum([tobacco, paan, smoke, alcohol])
    
    if risk_factors > 0:
        if risk_factors >= 3:
            st.error(f"⚠️ **HIGH RISK**: You have {risk_factors} risk factors. Regular oral cancer screening is strongly recommended!")
        elif risk_factors >= 1:
            st.warning(f"⚡ **MODERATE RISK**: You have {risk_factors} risk factor(s). Consider regular dental checkups.")
    
    # Find Dentist Button
    st.markdown("---")
    if st.button(t['find_dentist'], use_container_width=True):
        st.markdown("[🔗 Click here to find dentists near you (Google Maps)](https://www.google.com/maps/search/dentist+near+me)")
    
    # Disclaimer
    st.markdown(f'<div class="disclaimer">{t["disclaimer"]}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        Made with ❤️ for India | Powered by EfficientNetB0 | 
        <a href="https://github.com/Arihant240" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
