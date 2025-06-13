import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# --- Page Configuration & Style ---
st.set_page_config(
    page_title="Klasifikasi Lesi Kulit",
    page_icon="🔬",
    layout="wide"
)

# --- Contextual Data Dictionary ---
lesion_info = {
    'Actinic keratoses': {'description': "Lesi pra-kanker yang disebabkan oleh paparan sinar UV.", 'risk': 'Tinggi', 'alert_level': 'warning'},
    'Basal cell carcinoma': {'description': "Jenis kanker kulit paling umum dan jarang menyebar.", 'risk': 'Tinggi', 'alert_level': 'warning'},
    'Benign keratosis-like lesions': {'description': "Pertumbuhan kulit non-kanker yang umum seiring usia.", 'risk': 'Rendah', 'alert_level': 'success'},
    'Dermatofibroma': {'description': "Benjolan kulit jinak yang umum di kaki bagian bawah.", 'risk': 'Rendah', 'alert_level': 'success'},
    'Melanocytic nevi': {'description': "Nama lain tahi lalat, sebagian besar tidak berbahaya.", 'risk': 'Rendah', 'alert_level': 'success'},
    'Melanoma': {'description': "Jenis kanker kulit paling serius karena kemampuannya menyebar.", 'risk': 'Sangat Tinggi', 'alert_level': 'error'},
    'Vascular lesions': {'description': "Kelainan kulit dari pembuluh darah, umumnya tidak berbahaya.", 'risk': 'Rendah', 'alert_level': 'success'}
}
class_names = list(lesion_info.keys())

# --- Model Loading ---
# This function loads the models from local .h5 files.
# It uses Streamlit's caching to load them only once.
@st.cache_resource
def load_models():
    """Loads both models from files with error handling."""
    try:
        # Load the fine-tuned DenseNet model
        densenet_model = tf.keras.models.load_model('skin_cancer_densenet121_finetuned.h5')
        # Load the baseline model
        baseline_model = tf.keras.models.load_model('baseline_model.h5')
        return {'DenseNet121 (Advanced)': densenet_model, 'Baseline CNN': baseline_model}
    except Exception as e:
        st.error(f"Critical error while loading models: {e}")
        st.error("Ensure 'baseline_model.h5' and 'skin_cancer_densenet121_finetuned.h5' are present in the same folder as this script.")
        return None

models = load_models()

# --- Sidebar for Project Info ---
with st.sidebar:
    st.title("Project Details")
    st.info(
        '''
        **Course:** Biomedical Signal & Image Processing
        **Dataset:** HAM10000
        **Models:** Custom CNN vs. Fine-Tuned DenseNet121
        '''
    )
    st.warning("**Disclaimer:** This is a prediction tool, not a substitute for professional medical diagnosis.", icon="⚠️")

# --- Main Application Area ---
st.title("🔬 Klasifikasi Lesi Kulit (Skin Lesion Classification)")
st.markdown("Unggah gambar untuk membandingkan performa model CNN baseline dengan model DenseNet121 yang telah di-fine-tuning.")

upload_col, info_col = st.columns([1, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Pilih atau jatuhkan gambar di sini (Choose or drop an image here)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file is None:
    info_col.info("Silakan unggah gambar untuk dianalisis (Please upload an image to analyze)...")
elif models is None:
    info_col.error("Models could not be loaded. Please check the file paths and try again.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    upload_col.image(image, caption="Gambar yang akan dianalisis.", use_column_width=True)

    if info_col.button("🔬 Hasil Analisis (Analyze)", type="primary", use_container_width=True):
        with st.spinner("Model sedang menganalisis..."):
            # --- Baseline CNN Model Prediction ---
            img_base = image.resize((100, 75))
            arr_base = tf.keras.preprocessing.image.img_to_array(img_base) / 255.0
            pred_base = models['Baseline CNN'].predict(np.expand_dims(arr_base, axis=0))
            label_base = class_names[np.argmax(pred_base)]

            # --- DenseNet121 Model Prediction ---
            img_dense = image.resize((224, 224))
            arr_dense = tf.keras.preprocessing.image.img_to_array(img_dense)
            arr_dense_preprocessed = tf.keras.applications.densenet.preprocess_input(np.expand_dims(arr_dense, axis=0))
            pred_dense = models['DenseNet121 (Advanced)'].predict(arr_dense_preprocessed)
            top_pred_index_dense = np.argmax(pred_dense)
            label_dense = class_names[top_pred_index_dense]
            info_dense = lesion_info[label_dense]

        st.markdown("---")
        st.subheader("Hasil Prediksi Model (Model Prediction Results)")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Baseline CNN Prediction", value=label_base)
        with res_col2:
            st.metric(label="DenseNet121 Advanced Prediction", value=label_dense)

        st.markdown("---")
        st.subheader("Analisis Detail (Model DenseNet121)")

        alert_level = info_dense['alert_level']
        alert_text = f"**Risiko Umum (General Risk):** {info_dense['risk']}. {info_dense['description']}"
        if alert_level == 'error': st.error(alert_text, icon="🚨")
        elif alert_level == 'warning': st.warning(alert_text, icon="⚠️")
        else: st.success(alert_text, icon="✅")

        st.bar_chart(pd.DataFrame({
            'Probabilitas (Probability)': pred_dense.flatten(),
            'Tipe Lesi (Lesion Type)': class_names
        }).set_index('Tipe Lesi (Lesion Type)'))
