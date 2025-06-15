import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="أذكار المسلم الذكي - Smart Islamic Adhkar",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from collections import Counter
import re

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Try to import scikit-learn, fallback gracefully if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Modern React-style CSS with updated colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Modern React-style Variables - Updated Color Palette */
    :root {
        --primary-50: #f0f9ff;
        --primary-100: #e0f2fe;
        --primary-500: #0ea5e9;
        --primary-600: #0284c7;
        --primary-700: #0369a1;
        --primary-800: #075985;
        --secondary-50: #fdf4ff;
        --secondary-500: #a855f7;
        --secondary-600: #9333ea;
        --accent-50: #ecfeff;
        --accent-500: #06b6d4;
        --accent-600: #0891b2;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --success-100: #dcfce7;
        --success-500: #22c55e;
        --success-800: #166534;
        --warning-100: #fef3c7;
        --warning-800: #92400e;
        --error-100: #fee2e2;
        --error-500: #ef4444;
        --white: #ffffff;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--primary-50) 0%, var(--white) 50%, var(--accent-50) 100%);
        min-height: 100vh;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-600) 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.2);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Amiri', serif;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-family: 'Inter', sans-serif;
        font-weight: 300;
    }
    
    /* Modern Tabs */
    .modern-tabs {
        display: flex;
        background: var(--white);
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-100);
    }
    
    .tab-button {
        flex: 1;
        padding: 12px 20px;
        background: transparent;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        color: var(--gray-600);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .tab-button.active {
        background: var(--primary-600);
        color: white;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.3);
    }
    
    .tab-button:hover:not(.active) {
        background: var(--primary-50);
        color: var(--primary-700);
    }
    
    /* Enhanced Search Container */
    .search-container {
        position: relative;
        margin-bottom: 2rem;
    }
    
    .search-wrapper {
        position: relative;
        background: var(--white);
        border-radius: 16px;
        border: 2px solid var(--gray-200);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .search-wrapper:focus-within {
        border-color: var(--primary-500);
        box-shadow: 0 4px 25px rgba(14, 165, 233, 0.15);
        transform: translateY(-2px);
    }
    
    .search-icon {
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--gray-400);
        font-size: 1.3rem;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .search-wrapper:focus-within .search-icon {
        color: var(--primary-500);
    }
    
    /* Modern Cards */
    .modern-card {
        background: var(--white);
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-100);
        margin-bottom: 1.5rem;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-200);
    }
    
    .featured-card {
        border: 2px solid var(--primary-500);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.2);
    }
    
    .card-header {
        padding: 1.5rem 1.5rem 1rem 1.5rem;
    }
    
    .card-content {
        padding: 0 1.5rem 1.5rem 1.5rem;
    }
    
    .card-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--gray-100);
    }
    
    /* Category Badges */
    .category-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: capitalize;
    }
    
    .badge-morning { background: var(--warning-100); color: var(--warning-800); }
    .badge-evening { background: var(--secondary-50); color: #7c3aed; }
    .badge-general { background: var(--primary-100); color: var(--primary-800); }
    .badge-istighfar { background: var(--success-100); color: var(--success-800); }
    .badge-protection { background: var(--error-100); color: #dc2626; }
    
    /* Arabic Text */
    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 1.8rem;
        line-height: 1.8;
        color: var(--primary-800);
        margin-bottom: 1rem;
        text-align: right;
        direction: rtl;
    }
    
    .translation {
        color: var(--gray-700);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Modern Buttons */
    .modern-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 8px;
        border: none;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .btn-primary {
        background: var(--primary-600);
        color: white;
    }
    
    .btn-primary:hover {
        background: var(--primary-700);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(2, 132, 199, 0.3);
    }
    
    .btn-ghost {
        background: transparent;
        color: var(--gray-500);
        border: 1px solid var(--gray-200);
    }
    
    .btn-ghost:hover {
        background: var(--gray-50);
        color: var(--gray-700);
    }
    
    .btn-ghost.active {
        color: var(--error-500);
        border-color: var(--error-200);
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, var(--primary-500) 0%, var(--accent-500) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stat-card.blue {
        background: linear-gradient(135deg, var(--primary-500) 0%, #6366f1 100%);
    }
    
    .stat-card.purple {
        background: linear-gradient(135deg, var(--secondary-500) 0%, #ec4899 100%);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Reward Box */
    .reward-box {
        background: var(--primary-50);
        border: 1px solid var(--primary-100);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .reward-text {
        color: var(--primary-800);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: var(--gray-500);
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .arabic-text {
            font-size: 1.5rem;
        }
        
        .modern-tabs {
            flex-direction: column;
            gap: 4px;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0;
    }
    
    .stButton > button {
        background: var(--primary-600) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-700) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(2, 132, 199, 0.3) !important;
    }
    
    /* Active tab styling */
    .stButton > button:focus {
        background: var(--primary-700) !important;
        box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2) !important;
    }
    
    # Enhanced Search Input Styling - Modern Dark Theme
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
        padding: 14px 50px 14px 20px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        background: #1f2937 !important;
        color: #e5e7eb !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
        background: #374151 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
        font-style: normal !important;
    }
    
    /* Search Input Container Enhancement */
    .stTextInput > div {
        position: relative !important;
    }
    
    .search-button {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: var(--primary-600) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        z-index: 10 !important;
    }
    
    .search-button:hover {
        background: var(--primary-700) !important;
    }
</style>
""", unsafe_allow_html=True)

# Clean Arabic text by removing diacritics
def remove_tashkeel(text):
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

# Manual cosine similarity function
def manual_cosine_similarity(a, b):
    a_dense = a.toarray().flatten()
    b_dense = b.toarray()
    dot_products = np.dot(b_dense, a_dense)
    a_norm = np.linalg.norm(a_dense)
    b_norms = np.linalg.norm(b_dense, axis=1)
    return dot_products / (a_norm * b_norms + 1e-10)

@st.cache_data
def load_data():
    """Load and cache the adhkar data"""
    try:
        df = pd.read_csv('adhkar_df.csv')
        return df.dropna()
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return pd.DataFrame()

def load_model_and_vectorizer():
    """Load model and data using joblib if available"""
    try:
        if JOBLIB_AVAILABLE:
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
        else:
            # Fallback to pickle if joblib not available
            with open("tfidf_vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None, pd.DataFrame()

# Function to find the most similar dua
def find_similar_dua(user_dua, vectorizer, adhkar_df):
    clean_dua = remove_tashkeel(user_dua.strip())
    if not clean_dua:
        return "❗ الرجاء إدخال دعاء صحيح", ""
    
    user_vector = vectorizer.transform([clean_dua])
    tfidf_matrix = vectorizer.transform(adhkar_df['clean_text'])
    similarities = manual_cosine_similarity(user_vector, tfidf_matrix)
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score < 0.1:
        return "❌ لم يتم العثور على دعاء مشابه", ""
    
    return adhkar_df.iloc[best_idx]['category'], adhkar_df.iloc[best_idx]['text']

def semantic_search(query, vectorizer, df, top_k=5):
    """Perform semantic search using TF-IDF similarity"""
    try:
        if vectorizer is None:
            return pd.DataFrame(), []
        
        # Transform query and data
        query_vector = vectorizer.transform([query])
        tfidf_matrix = vectorizer.transform(df['clean_text'])
        
        # Calculate cosine similarity using manual method
        similarities = manual_cosine_similarity(query_vector, tfidf_matrix)
        
        # Get top k most similar adhkar
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter out very low similarities
        meaningful_indices = [idx for idx, sim in zip(top_indices, top_similarities) if sim > 0.1]
        meaningful_similarities = [sim for sim in top_similarities if sim > 0.1]
        
        if not meaningful_indices:
            return pd.DataFrame(), []
            
        result_df = df.iloc[meaningful_indices].copy()
        return result_df, meaningful_similarities
        
    except Exception as e:
        st.error(f"خطأ في البحث الدلالي: {e}")
        return pd.DataFrame(), []

def find_similar_adhkar(adhkar_text, vectorizer, df, top_k=3):
    """Find similar adhkar to a given adhkar"""
    try:
        if vectorizer is None:
            return pd.DataFrame(), []
            
        # Find the index of current adhkar
        current_idx = df[df['clean_text'] == adhkar_text].index
        if len(current_idx) == 0:
            return pd.DataFrame(), []
        
        current_idx = current_idx[0]
        
        # Get similarity with all other adhkar
        current_vector = vectorizer.transform([adhkar_text])
        tfidf_matrix = vectorizer.transform(df['clean_text'])
        similarities = manual_cosine_similarity(current_vector, tfidf_matrix)
        
        # Remove self-similarity and get top k
        similarities[current_idx] = -1
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter meaningful similarities
        meaningful_indices = [idx for idx, sim in zip(top_indices, top_similarities) if sim > 0.1]
        meaningful_similarities = [sim for sim in top_similarities if sim > 0.1]
        
        if not meaningful_indices:
            return pd.DataFrame(), []
            
        result_df = df.iloc[meaningful_indices].copy()
        return result_df, meaningful_similarities
        
    except Exception as e:
        st.error(f"خطأ في العثور على أذكار مشابهة: {e}")
        return pd.DataFrame(), []

def get_time_based_greeting():
    """Get appropriate greeting based on current time"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "🌅 صباح الخير - أذكار الصباح", "morning"
    elif 12 <= current_hour < 18:
        return "☀️ مساء الخير - أذكار المساء", "afternoon"
    elif 18 <= current_hour < 22:
        return "🌆 مساء الخير - أذكار المساء", "evening"
    else:
        return "🌙 تصبح على خير - أذكار النوم", "night"

def initialize_session_state():
    """Initialize session state variables"""
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'daily_adhkar_count' not in st.session_state:
        st.session_state.daily_adhkar_count = 0
    if 'favorite_adhkar' not in st.session_state:
        st.session_state.favorite_adhkar = []
    if 'last_date' not in st.session_state:
        st.session_state.last_date = datetime.now().date()
    
    # Reset daily counter if it's a new day
    if st.session_state.last_date != datetime.now().date():
        st.session_state.daily_adhkar_count = 0
        st.session_state.last_date = datetime.now().date()

def get_category_class(category):
    category_classes = {
        'morning': 'badge-morning',
        'evening': 'badge-evening',
        'general': 'badge-general',
        'istighfar': 'badge-istighfar',
        'protection': 'badge-protection',
    }
    return category_classes.get(category, 'badge-general')

def display_adhkar_card(adhkar_row, similarity_score=None, is_similar=False):
    """Display a modern adhkar card using only Streamlit components"""
    category_class = get_category_class(adhkar_row['category'])
    is_favorite = adhkar_row['text'] in st.session_state.favorite_adhkar
    
    # Create card container using Streamlit's container with custom styling
    card_style = """
    <style>
    div[data-testid="stVerticalBlock"] > div:has(div.adhkar-card) {
        background: var(--white);
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-100);
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    </style>
    """
    
    with st.container():
        # Apply card styling
        st.markdown(card_style, unsafe_allow_html=True)
        
        # Mark this as an adhkar card for CSS targeting
        st.markdown('<div class="adhkar-card"></div>', unsafe_allow_html=True)
        
        # Category badge and favorite indicator
        col_badge, col_fav = st.columns([3, 1])
        with col_badge:
            if adhkar_row['category'] == 'morning':
                st.markdown("🌅 **أذكار الصباح**")
            elif adhkar_row['category'] == 'evening':
                st.markdown("🌅 **أذكار المساء**")
            elif adhkar_row['category'] == 'general':
                st.markdown("📿 **أذكار عامة**")
            elif adhkar_row['category'] == 'istighfar':
                st.markdown("🤲 **استغفار**")
            elif adhkar_row['category'] == 'protection':
                st.markdown("🛡️ **حماية**")
            else:
                st.markdown(f"📖 **{adhkar_row['category']}**")
        
        with col_fav:
            if is_favorite:
                st.markdown("❤️")
            else:
                st.markdown("🤍")
        
        # Display similarity score if available
        if similarity_score is not None:
            similarity_percentage = int(similarity_score * 100)
            st.info(f"🎯 تشابه: {similarity_percentage}%")
        
        # Arabic text in a special container
        st.markdown(f"""
        <div style="
            font-family: 'Amiri', serif;
            font-size: 1.8rem;
            line-height: 1.8;
            color: var(--primary-800);
            margin: 1.5rem 0;
            text-align: right;
            direction: rtl;
            background: var(--primary-50);
            padding: 1.5rem;
            border-radius: 12px;
            border-right: 4px solid var(--primary-500);
        ">
            {adhkar_row['text']}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("📖 قرأت هذا الذكر", key=f"read_{adhkar_row.name}", use_container_width=True):
                st.session_state.counter += 1
                st.session_state.daily_adhkar_count += 1
                st.success("✅ تم احتساب القراءة!")
        
        with col2:
            fav_text = "💔 إزالة من المفضلة" if is_favorite else "❤️ إضافة للمفضلة"
            if st.button(fav_text, key=f"fav_{adhkar_row.name}", use_container_width=True):
                if is_favorite:
                    st.session_state.favorite_adhkar.remove(adhkar_row['text'])
                    st.success("تم إزالة الذكر من المفضلة")
                    st.rerun()
                else:
                    st.session_state.favorite_adhkar.append(adhkar_row['text'])
                    st.success("✅ تم إضافة الذكر للمفضلة!")
                    st.rerun()
        
        with col3:
            if SKLEARN_AVAILABLE and st.button("🔍 مشابه", key=f"similar_{adhkar_row.name}", use_container_width=True):
                st.session_state.current_adhkar_for_similarity = adhkar_row['text']
                st.rerun()
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{adhkar_row.name}", use_container_width=True):
                st.code(adhkar_row['text'], language="text")

def show_installation_guide():
    """Show installation guide for missing dependencies"""
    st.markdown("""
    <div class="reward-box">
        <h3>🛠️ دليل التثبيت للميزات الذكية</h3>
        <p>لتفعيل الميزات الذكية، يرجى تثبيت المكتبات المطلوبة:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
# تثبيت المكتبات المطلوبة
pip install scikit-learn joblib

# أو تثبيت جميع المتطلبات معاً
pip install streamlit pandas numpy scikit-learn joblib

# للتثبيت على Streamlit Cloud، أضف إلى requirements.txt:
# streamlit>=1.28.0
# pandas>=1.5.0  
# numpy>=1.24.0
# scikit-learn>=1.3.0
# joblib>=1.3.0
    """, language="bash")
    
    st.markdown("""
    **الميزات المتاحة بعد التثبيت:**
    - 🔍 العثور على أذكار مشابهة
    - 📊 تحليلات ذكية للفئات
    - 🎯 توصيات مخصصة
    """)

def main():
    # Show warnings after page config if needed
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn not installed. AI features will be disabled. Install with: pip install scikit-learn")
    
    if not JOBLIB_AVAILABLE:
        st.info("ℹ️ joblib not installed. Using pickle as fallback for model loading. Install with: pip install joblib")
    
    # Initialize session state
    initialize_session_state()
    
    # Load data and model
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()  # Fallback to regular data loading
        if df.empty:
            st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
            return
    
    # Modern Header with Logo
    ai_status = "🤖 مفعل" if (SKLEARN_AVAILABLE and vectorizer is not None) else "❌ غير متاح"
    st.markdown(f"""
    <div class="modern-header">
        <div class="header-content">
            <div style="margin-bottom: 1rem;">
                <img src="https://raw.githubusercontent.com/Bayaan/bayaanlogo1.png" 
                     alt="Bayaan Logo" 
                     style="height: 80px; width: auto; margin-bottom: 1rem; filter: brightness(0) invert(1);">
            </div>
            <h1 class="header-title">أذكار المسلم الذكي</h1>
            <p class="header-subtitle">Islamic Adhkar AI - الذكاء الاصطناعي: {ai_status}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Time-based greeting
    greeting, time_period = get_time_based_greeting()
    st.markdown(f"""
    <div class="reward-box" style="text-align: center; margin-top: 1rem;">
        <h3>{greeting}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab selection buttons
    col2, col3, col4, col5 = st.columns(4)
    
    with col2:
        if st.button("🤖 الذكي", key="tab_ai", use_container_width=True):
            st.session_state.active_tab = 'ai'
    
    with col3:
        if st.button("❤️ المفضلة", key="tab_favorites", use_container_width=True):
            st.session_state.active_tab = 'favorites'
    
    with col4:
        if st.button("📊 الإحصائيات", key="tab_stats", use_container_width=True):
            st.session_state.active_tab = 'stats'
    
    with col5:
        if st.button("ℹ️ حول", key="tab_about", use_container_width=True):
            st.session_state.active_tab = 'about'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display content based on active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'search'
    
    # AI Tab
    elif st.session_state.active_tab == 'ai' and SKLEARN_AVAILABLE and vectorizer is not None:
        st.markdown("### 🤖 البحث الذكي بالذكاء الاصطناعي")
        
        # Modern React-style Search Container
        st.markdown("""
        <div class="search-container">
            <div class="search-wrapper">
                <div class="search-inner">
                    <div class="search-icon">🔍</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create search input with React styling
        col_search, col_button = st.columns([4, 1])
        
        with col_search:
            search_query = st.text_input(
                "",
                placeholder="Search for components, styles, creators...",
                help="ابحث بأي طريقة: بالكلمات المفتاحية، المعنى، أو وصف حالتك",
                key="unified_search",
                label_visibility="collapsed"
            )
        
        with col_button:
            search_pressed = st.button("Search", key="search_btn", use_container_width=True)
        
        # Add custom styling to make it look like the React component
        st.markdown("""
        <script>
        // Apply React-style classes to search elements
        setTimeout(() => {
            const searchInput = document.querySelector('[data-testid="stTextInput"] input');
            const searchButton = document.querySelector('[data-testid="stButton"] button');
            
            if (searchInput && searchButton) {
                // Wrap search input
                const wrapper = document.createElement('div');
                wrapper.className = 'search-wrapper';
                const inner = document.createElement('div');
                inner.className = 'search-inner';
                
                // Add search icon
                const icon = document.createElement('div');
                icon.className = 'search-icon';
                icon.innerHTML = '🔍';
                
                // Style button
                searchButton.className += ' search-button';
                
                // Apply structure
                searchInput.parentElement.style.position = 'relative';
                inner.appendChild(icon);
            }
        }, 100);
        </script>
        """, unsafe_allow_html=True)
        
        # Search options in a React-style panel
        with st.container():
            st.markdown('<div class="search-options">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                search_mode = st.selectbox(
                    "نوع البحث",
                    ["ذكي شامل", "البحث عن دعاء مناسب", "بحث دلالي"],
                    key="search_mode"
                )
            with col2:
                search_depth = st.selectbox("عدد النتائج", [3, 5, 8, 10], index=1, key="unified_depth")
            with col3:
                min_similarity = st.slider("دقة التشابه", 0.1, 0.8, 0.2, 0.1, key="unified_similarity")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform search when query is entered or button is pressed
        if search_query and (search_pressed or search_query):
            with st.spinner("🤖 جاري البحث الذكي..."):
                
                if search_mode == "البحث عن دعاء مناسب":
                    # Smart Dua Finder
                    category, similar_text = find_similar_dua(search_query, vectorizer, df)
                    if similar_text:
                        st.success(f"✨ تم العثور على دعاء مناسب في فئة: **{category}**")
                        result_row = pd.Series({
                            'text': similar_text,
                            'category': category
                        }, name='dua_result')
                        display_adhkar_card(result_row, is_similar=True)
                    else:
                        st.info(category)
                
                elif search_mode == "بحث دلالي":
                    # Semantic search
                    semantic_results, similarities = semantic_search(
                        search_query, vectorizer, df, top_k=search_depth
                    )
                    
                    if not semantic_results.empty:
                        valid_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
                        if valid_indices:
                            filtered_results = semantic_results.iloc[valid_indices]
                            filtered_similarities = [similarities[i] for i in valid_indices]
                            
                            st.success(f"🎯 تم العثور على {len(filtered_results)} نتيجة")
                            
                            for idx, (_, row) in enumerate(filtered_results.iterrows()):
                                display_adhkar_card(row, 
                                                  similarity_score=filtered_similarities[idx], 
                                                  is_similar=True)
                        else:
                            st.warning("لم يتم العثور على نتائج تتجاوز حد التشابه المحدد")
                    else:
                        st.info("لم يتم العثور على نتائج. جرب كلمات مختلفة.")
                
                else:  # "ذكي شامل"
                    # Try both methods and combine results
                    all_results = []
                    
                    # First try semantic search
                    semantic_results, semantic_similarities = semantic_search(
                        search_query, vectorizer, df, top_k=search_depth//2
                    )
                    
                    if not semantic_results.empty:
                        for idx, (_, row) in enumerate(semantic_results.iterrows()):
                            if semantic_similarities[idx] >= min_similarity:
                                all_results.append((row, semantic_similarities[idx], "دلالي"))
                    
                    # Then try dua finder
                    category, similar_text = find_similar_dua(search_query, vectorizer, df)
                    if similar_text:
                        # Check if this result is already in semantic results
                        is_duplicate = any(result[0]['text'] == similar_text for result in all_results)
                        if not is_duplicate:
                            dua_row = pd.Series({
                                'text': similar_text,
                                'category': category
                            }, name='dua_smart_result')
                            all_results.append((dua_row, 0.95, "دعاء مناسب"))
                    
                    if all_results:
                        st.success(f"🎯 تم العثور على {len(all_results)} نتيجة ذكية")
                        
                        # Sort by similarity score
                        all_results.sort(key=lambda x: x[1], reverse=True)
                        
                        for idx, (row, similarity, search_type) in enumerate(all_results):
                            # Add search type indicator
                            st.markdown(f"**نوع البحث:** {search_type}")
                            display_adhkar_card(row, 
                                              similarity_score=similarity if similarity < 1 else None, 
                                              is_similar=True)
                    else:
                        st.info("لم يتم العثور على نتائج مناسبة. جرب كلمات أو عبارات مختلفة.")
        
        # Quick search section with React styling
        st.markdown("""
        <div class="quick-search-container">
            <h3 style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">🚀 بحث سريع</h3>
            <div class="quick-search-grid">
        """, unsafe_allow_html=True)
        
        quick_searches = [
            "الحماية والأمان", "الدعاء للوالدين", "الاستغفار والتوبة", 
            "الحمد والشكر", "طلب الهداية", "دعاء المريض"
        ]
        
        cols = st.columns(3)
        for i, quick_search in enumerate(quick_searches):
            with cols[i % 3]:
                if st.button(quick_search, key=f"quick_{i}", use_container_width=True):
                    with st.spinner(f"🔍 جاري البحث عن: {quick_search}"):
                        semantic_results, similarities = semantic_search(
                            quick_search, vectorizer, df, top_k=3
                        )
                        if not semantic_results.empty:
                            st.markdown(f"**نتائج: {quick_search}**")
                            for idx, (_, row) in enumerate(semantic_results.iterrows()):
                                display_adhkar_card(row, 
                                                  similarity_score=similarities[idx],
                                                  is_similar=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Favorites Tab
    elif st.session_state.active_tab == 'favorites':
        st.markdown("### ❤️ الأذكار المفضلة")
        
        if st.session_state.favorite_adhkar:
            st.success(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
            
            # AI-powered similar favorites (only if sklearn available)
            if SKLEARN_AVAILABLE and vectorizer is not None:
                if st.button("🤖 اقتراحات ذكية بناءً على المفضلة", key="smart_suggestions"):
                    all_suggestions = []
                    for fav_adhkar in st.session_state.favorite_adhkar[:3]:
                        similar_results, similarities = find_similar_adhkar(
                            fav_adhkar, vectorizer, df, top_k=2
                        )
                        if not similar_results.empty:
                            for idx, (_, row) in enumerate(similar_results.iterrows()):
                                if row['text'] not in st.session_state.favorite_adhkar:
                                    all_suggestions.append((row, similarities[idx]))
                    
                    if all_suggestions:
                        st.markdown("### 🤖 اقتراحات ذكية بناءً على مفضلتك:")
                        for idx, (row, sim) in enumerate(all_suggestions[:5]):
                            display_adhkar_card(row, similarity_score=sim, is_similar=True)
                    else:
                        st.info("لا توجد اقتراحات ذكية متاحة حالياً")
            
            st.markdown("---")
            st.markdown("### 📚 أذكارك المفضلة:")
            
            for i, adhkar_text in enumerate(st.session_state.favorite_adhkar):
                # Find the adhkar in the dataframe
                adhkar_row = df[df['text'] == adhkar_text]
                if not adhkar_row.empty:
                    row = adhkar_row.iloc[0]
                    display_adhkar_card(row)
                else:
                    # Create a temporary row if not found in the dataframe
                    temp_row = pd.Series({
                        'text': adhkar_text,
                        'category': 'مفضلة'
                    }, name=f'fav_{i}')
                    display_adhkar_card(temp_row)
            
            if st.button("🗑️ مسح جميع المفضلة", key="clear_favorites"):
                st.session_state.favorite_adhkar = []
                st.success("تم مسح جميع الأذكار المفضلة")
                st.rerun()
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">❤️</div>
                <h3>لا توجد أذكار مفضلة</h3>
                <p>أضف أذكارك المفضلة من قسم البحث لتظهر هنا</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistics Tab
    elif st.session_state.active_tab == 'stats':
        st.markdown("### 📊 الإحصائيات والتحليلات")
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card blue">
                <div class="stat-number">{len(df['category'].unique())}</div>
                <div class="stat-label">عدد الفئات</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.daily_adhkar_count}</div>
                <div class="stat-label">أذكار اليوم</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card purple">
                <div class="stat-number">{len(st.session_state.favorite_adhkar)}</div>
                <div class="stat-label">المفضلة</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Category distribution
        st.markdown("### 📈 توزيع الأذكار حسب الفئات")
        category_counts = df['category'].value_counts()
        st.bar_chart(category_counts.head(10))
        
        # Most common categories
        st.markdown("### 🏆 أكثر الفئات شيوعاً")
        for i, (category, count) in enumerate(category_counts.head(5).items(), 1):
            st.write(f"{i}. **{category}**: {count} ذكر")
        
        # Text length analysis
        st.markdown("### 📏 تحليل أطوال النصوص")
        text_lengths = df['text'].str.len()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("متوسط الطول", f"{text_lengths.mean():.0f} حرف")
        with col2:
            st.metric("أقصر نص", f"{text_lengths.min()} حرف")
        with col3:
            st.metric("أطول نص", f"{text_lengths.max()} حرف")
        
        # Text length histogram
        st.markdown("### 📊 توزيع أطوال النصوص")
        hist_data = np.histogram(text_lengths, bins=20)
        chart_df = pd.DataFrame({
            'count': hist_data[0]
        })
        st.bar_chart(chart_df)
    
    # About Tab
    elif st.session_state.active_tab == 'about':
        st.markdown("## ℹ️ حول التطبيق")
        
        if not SKLEARN_AVAILABLE:
            show_installation_guide()
            st.markdown("---")
        
        st.markdown(f"""
        ### 🕌 تطبيق أذكار المسلم الذكي
        
        هذا التطبيق يحتوي على مجموعة شاملة من الأذكار والأدعية الإسلامية المأخوذة من القرآن الكريم والسنة النبوية الشريفة.
        
        #### 🌟 المميزات الأساسية:
        - 📖 أكثر من {len(df)} ذكر ودعاء
        - 🔍 بحث تقليدي في الأذكار
        - ⭐ إمكانية حفظ الأذكار المفضلة
        - 📊 تتبع عدد الأذكار المقروءة
        - 🎯 اقتراحات حسب الوقت
        - 📱 تصميم حديث مستوحى من واجهات حديثة
        """)
        
        if SKLEARN_AVAILABLE and vectorizer is not None:
            vocab_size = len(vectorizer.get_feature_names_out())
            st.markdown(f"""
            #### 🤖 المميزات الذكية (مفعلة):
            - 🧠 بحث ذكي بالمعنى باستخدام TF-IDF
            - 🎯 البحث عن الدعاء المناسب لحالتك
            - 🔍 العثور على أذكار مشابهة
            - 📊 تحليلات ذكية للفئات
            - 🎯 توصيات مخصصة
            - 📈 تحليل النصوص بـ {vocab_size:,} كلمة
            - 🤲 خوارزمية التشابه المحسنة
            """)
        elif SKLEARN_AVAILABLE:
            st.markdown("""
            #### ⚠️ المميزات الذكية (غير مفعلة):
            - النموذج غير محمل - تأكد من وجود ملف `tfidf_vectorizer.pkl`
            """)
        else:
            st.markdown("""
            #### ❌ المميزات الذكية (غير متاحة):
            - يتطلب تثبيت scikit-learn و joblib
            - راجع دليل التثبيت أعلاه
            """)
        
        st.markdown("""
        #### 🚀 الميزات الجديدة:
        - **البحث الذكي المحسن** باستخدام خوارزمية التشابه المخصصة
        - **البحث عن الدعاء المناسب** - أدخل حالتك واحصل على الدعاء الأنسب
        - **تصميم حديث** مع تأثيرات بصرية متقدمة
        - **واجهة متجاوبة** تعمل بسلاسة على جميع الأجهزة
        - **تحليلات ذكية** للفئات والنصوص
        - **توصيات مخصصة** بناءً على تفضيلاتك
        
        #### 🤲 دعاء
        
        *"اللهم اجعل هذا العمل خالصاً لوجهك الكريم، وانفع به المسلمين في كل مكان"*
        
        **تذكر:** المداومة على الأذكار خير من الانقطاع عنها
        """)

if __name__ == "__main__":
    main()
