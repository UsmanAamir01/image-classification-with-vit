import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="FoodVision AI - Food Image Classifier",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e40af;
        --secondary-color: #1e293b;
        --accent-color: #3b82f6;
        --background-color: #F8F9FA;
        --text-color: #000000;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: #ffffff;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .upload-card {
        background: white;
        padding: 0;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #1e40af;
        overflow: hidden;
    }
    
    .upload-card h2 {
        color: #ffffff !important;
        background: linear-gradient(135deg, #1e40af 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        margin: 0;
        border-radius: 15px 15px 0 0;
    }
    
    .upload-card > :not(h2) {
        padding: 0 2rem 2rem 2rem;
    }
    
    .result-card {
        background: white;
        padding: 0;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #3b82f6;
        overflow: hidden;
    }
    
    .result-card h2 {
        color: #ffffff !important;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        padding: 1.5rem 2rem;
        margin: 0;
        border-radius: 15px 15px 0 0;
    }
    
    .result-card > :not(h2) {
        padding: 0 2rem 2rem 2rem;
    }
    
    /* Top prediction styling */
    .top-prediction {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(30, 64, 175, 0.4);
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .top-prediction h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-transform: capitalize;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .top-prediction h3 {
        color: white;
        font-size: 1.5rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Prediction item styling */
    .prediction-item {
        background: linear-gradient(90deg, #f0f9ff 0%, #ffffff 100%);
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .prediction-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        border-left-color: #1e40af;
    }
    
    .prediction-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        text-transform: capitalize;
    }
    
    .prediction-confidence {
        font-size: 1rem;
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #10b981;
        color: white;
    }
    
    .badge-info {
        background: #3b82f6;
        color: white;
    }
    
    .badge-warning {
        background: #f59e0b;
        color: white;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e293b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        min-width: 150px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 5px 15px rgba(30, 64, 175, 0.3);
    }
    
    .stat-box h3 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
        color: white;
    }
    
    .stat-box p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        color: white;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fade {
        animation: fadeIn 0.6s ease-in;
    }
    
    .animate-slide {
        animation: slideIn 0.4s ease-out;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.6);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1e40af;
        margin: 1rem 0;
        color: #000000;
    }
    
    .info-box strong {
        color: #1e40af;
    }
    
    .info-box h3, .info-box p {
        color: #000000;
    }
    
    /* Dark background text - white font */
    .info-box.dark-bg {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left-color: #3b82f6;
        color: #ffffff;
    }
    
    .info-box.dark-bg h3, .info-box.dark-bg p {
        color: #ffffff;
    }
    
    /* Upload section styling */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #000000 !important;
    }
    
    /* Text color fix for all main content */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #000000;
    }
    
    /* Ensure headings are black by default */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* White text on dark backgrounds */
    .main-header h1, .main-header p {
        color: #ffffff !important;
    }
    
    .top-prediction h2, .top-prediction h3 {
        color: #ffffff !important;
    }
    
    .stat-box h3, .stat-box p {
        color: #ffffff !important;
    }
    
    /* Expander content with white text */
    .streamlit-expanderHeader {
        color: #000000 !important;
    }
    
    [data-testid="stExpander"] {
        background: transparent;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"] .streamlit-expanderContent p,
    [data-testid="stExpander"] .streamlit-expanderContent strong,
    [data-testid="stExpander"] .streamlit-expanderContent li {
        color: #ffffff !important;
    }
    
    /* Selectbox and other input labels */
    .stSelectbox label, .stFileUploader label {
        color: #000000 !important;
    }

</style>
""", unsafe_allow_html=True)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the trained Vision Transformer model"""
    model_path = "results/vit-food-classifier-final"
    
    # Load the processor and model
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    return processor, model

def predict_image(image, processor, model, top_k=5):
    """
    Make prediction on the uploaded image
    
    Args:
        image: PIL Image
        processor: ViTImageProcessor
        model: ViTForImageClassification
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples (label, probability)
    """
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to label-probability pairs
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        predictions.append((label, prob.item()))
    
    return predictions

def main():
    # Hero Header
    st.markdown("""
    <div class="main-header animate-fade">
        <h1>🍽️ FoodVision AI</h1>
        <p>Advanced Food Recognition powered by Vision Transformer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        with st.spinner("🔄 Loading AI model... Please wait."):
            processor, model = load_model()
            time.sleep(0.5)  # Brief pause for smooth UX
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📊 Model Statistics")
        
        st.markdown("""
        <div class="stat-box">
            <h3>101</h3>
            <p>Food Categories</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #2563eb 0%, #1e293b 100%);">
            <h3>224×224</h3>
            <p>Image Resolution</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #0ea5e9 0%, #1e40af 100%);">
            <h3>ViT</h3>
            <p>Transformer Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Upload** a food image
        2. **Wait** for AI analysis
        3. **View** top predictions
        4. **Explore** confidence scores
        """)
        
        st.markdown("---")
        
        st.markdown("### 🍕 Sample Categories")
        categories_list = [
            "🍕 Pizza", "🍣 Sushi", "🍔 Hamburger", 
            "🍰 Cheesecake", "🍝 Pasta", "🥗 Salad",
            "🍦 Ice Cream", "🌮 Tacos", "🍜 Ramen"
        ]
        for cat in categories_list:
            st.markdown(f"• {cat}")
        
        st.markdown("---")
        st.markdown("##### Built with ❤️ using Streamlit & Hugging Face")
    
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="upload-card animate-slide">', unsafe_allow_html=True)
        st.markdown("## 📤 Upload Your Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample images section
        st.markdown('<div class="upload-card animate-slide" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("## 🖼️ Or Try Sample Images")
        
        sample_options = {
            "Select a sample...": None,
            "🍕 Pizza": "archive/images/pizza",
            "🍣 Sushi": "archive/images/sushi",
            "🍔 Hamburger": "archive/images/hamburger",
            "🍦 Ice Cream": "archive/images/ice_cream",
            "🥞 Pancakes": "archive/images/pancakes",
            "🍝 Spaghetti Carbonara": "archive/images/spaghetti_carbonara",
            "🌮 Tacos": "archive/images/tacos",
            "🍰 Cheesecake": "archive/images/cheesecake"
        }
        
        selected_sample = st.selectbox(
            "Choose from popular foods:",
            list(sample_options.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Tips section
        st.markdown("""
        <div class="info-box animate-fade">
            <strong>💡 Tips for Best Results:</strong><br>
            • Use clear, well-lit images<br>
            • Center the food in the frame<br>
            • Avoid cluttered backgrounds<br>
            • Single food item works best
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-card animate-slide">', unsafe_allow_html=True)
        st.markdown("## 🎯 Analysis Results")
        
        # Process uploaded image or sample
        image_to_predict = None
        
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file).convert("RGB")
            st.image(
                image_to_predict, 
                caption="📸 Uploaded Image", 
                use_container_width=True,
                output_format="PNG"
            )
        
        elif selected_sample != "Select a sample..." and sample_options[selected_sample] is not None:
            import os
            sample_path = sample_options[selected_sample]
            if os.path.exists(sample_path):
                # Get first image from the folder
                images = [f for f in os.listdir(sample_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    image_path = os.path.join(sample_path, images[0])
                    image_to_predict = Image.open(image_path).convert("RGB")
                    st.image(
                        image_to_predict, 
                        caption=f"📸 Sample: {selected_sample}", 
                        use_container_width=True,
                        output_format="PNG"
                    )
        
        # Make prediction
        if image_to_predict is not None:
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing image...")
            progress_bar.progress(30)
            time.sleep(0.3)
            
            try:
                status_text.text("🧠 Running AI model...")
                progress_bar.progress(60)
                
                predictions = predict_image(image_to_predict, processor, model, top_k=5)
                
                status_text.text("✨ Generating results...")
                progress_bar.progress(100)
                time.sleep(0.2)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display top prediction prominently
                top_label, top_prob = predictions[0]
                
                # Determine confidence badge
                if top_prob >= 0.8:
                    confidence_badge = '<span class="badge badge-success">High Confidence</span>'
                elif top_prob >= 0.5:
                    confidence_badge = '<span class="badge badge-info">Medium Confidence</span>'
                else:
                    confidence_badge = '<span class="badge badge-warning">Low Confidence</span>'
                
                st.markdown(f"""
                <div class="top-prediction">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏆</div>
                    <h2>{top_label.replace('_', ' ')}</h2>
                    <h3>{top_prob*100:.1f}% Confidence</h3>
                    {confidence_badge}
                </div>
                """, unsafe_allow_html=True)
                
                # Display all top 5 predictions with enhanced styling
                st.markdown("### 📊 Detailed Predictions")
                
                for i, (label, prob) in enumerate(predictions, 1):
                    # Medal emoji for top 3
                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
                    
                    st.markdown(f"""
                    <div class="prediction-item animate-fade">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span class="prediction-label">{medal} {label.replace('_', ' ')}</span>
                            <span class="prediction-confidence">{prob*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(prob)
                
                # Additional insights
                st.markdown("---")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        label="Top Match Confidence", 
                        value=f"{predictions[0][1]*100:.1f}%",
                        delta=f"{(predictions[0][1] - predictions[1][1])*100:.1f}% gap" if len(predictions) > 1 else None
                    )
                
                with col_b:
                    # Calculate prediction certainty
                    certainty = "Very High" if predictions[0][1] >= 0.9 else "High" if predictions[0][1] >= 0.7 else "Moderate" if predictions[0][1] >= 0.5 else "Low"
                    st.metric(
                        label="Prediction Certainty", 
                        value=certainty
                    )
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🍽️</div>
                <h3 style="color: #000000;">Ready to Identify Your Food!</h3>
                <p style="color: #1e293b;">Upload an image or select a sample to get started</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with model info
    st.markdown("---")
    st.markdown("### ⚡ Performance Highlights")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);">
            <h3>Fast</h3>
            <p>Quick Inference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);">
            <h3>Accurate</h3>
            <p>High Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #0ea5e9 0%, #1e40af 100%);">
            <h3>Robust</h3>
            <p>Diverse Foods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown("""
        <div class="stat-box" style="background: linear-gradient(135deg, #2563eb 0%, #1e293b 100%);">
            <h3>Modern</h3>
            <p>ViT Architecture</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
