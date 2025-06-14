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

# Modern React-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Modern React-style Variables */
    :root {
        --emerald-50: #ecfdf5;
        --emerald-100: #d1fae5;
        --emerald-500: #10b981;
        --emerald-600: #059669;
        --emerald-700: #047857;
        --emerald-800: #065f46;
        --teal-50: #f0fdfa;
        --teal-500: #14b8a6;
        --teal-600: #0d9488;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --blue-500: #3b82f6;
        --purple-500: #8b5cf6;
        --yellow-100: #fef3c7;
        --yellow-800: #92400e;
        --red-100: #fee2e2;
        --red-500: #ef4444;
        --white: #ffffff;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--emerald-50) 0%, var(--white) 50%, var(--teal-50) 100%);
        min-height: 100vh;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, var(--emerald-600) 0%, var(--teal-600) 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
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
        background: var(--emerald-600);
        color: white;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }
    
    .tab-button:hover:not(.active) {
        background: var(--emerald-50);
        color: var(--emerald-700);
    }
    
    /* Modern Search */
    .search-container {
        position: relative;
        margin-bottom: 2rem;
    }
    
    .search-icon {
        position: absolute;
        left: 16px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--gray-400);
        font-size: 1.2rem;
        z-index: 2;
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
        border-color: var(--emerald-200);
    }
    
    .featured-card {
        border: 2px solid var(--emerald-500);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
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
    
    .badge-morning { background: var(--yellow-100); color: var(--yellow-800); }
    .badge-evening { background: #f3e8ff; color: #7c3aed; }
    .badge-general { background: #dbeafe; color: #1d4ed8; }
    .badge-istighfar { background: var(--emerald-100); color: var(--emerald-800); }
    .badge-protection { background: var(--red-100); color: #dc2626; }
    
    /* Arabic Text */
    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 1.8rem;
        line-height: 1.8;
        color: var(--emerald-800);
        margin-bottom: 1rem;
        text-align: right;
        direction: rtl;
    }
    
    .transliteration {
        font-style: italic;
        color: var(--gray-600);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
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
        background: var(--emerald-600);
        color: white;
    }
    
    .btn-primary:hover {
        background: var(--emerald-700);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
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
        color: var(--red-500);
        border-color: var(--red-200);
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, var(--emerald-500) 0%, var(--teal-500) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stat-card.blue {
        background: linear-gradient(135deg, var(--blue-500) 0%, #6366f1 100%);
    }
    
    .stat-card.purple {
        background: linear-gradient(135deg, var(--purple-500) 0%, #ec4899 100%);
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
        background: var(--emerald-50);
        border: 1px solid var(--emerald-100);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .reward-text {
        color: var(--emerald-800);
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
        background: var(--emerald-600) !important;
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
        background: var(--emerald-700) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Active tab styling */
    .stButton > button:focus {
        background: var(--emerald-700) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid var(--gray-200) !important;
        padding: 12px 16px 12px 40px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--emerald-500) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- AI MODEL UTILITIES ----------------------
@st.cache_resource
def load_model_and_vectorizer():
    """Load model and data using joblib if available"""
    try:
        if JOBLIB_AVAILABLE:
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
        else:
            # Fallback to pickle if joblib not available
            with open("tfidf_vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج أو البيانات: {e}")
        return None, pd.DataFrame()

def semantic_search(query, vectorizer, df, top_k=5):
    """Perform semantic search using TF-IDF similarity"""
    try:
        if vectorizer is None or 'text' not in df.columns:
            raise ValueError("النموذج أو البيانات غير صالحة")
        
        # Transform query and data
        query_vec = vectorizer.transform([query])
        tfidf_matrix = vectorizer.transform(df['text'].astype(str))
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[::-1][:top_k]
        return df.iloc[top_indices], cosine_similarities[top_indices]
    except Exception as e:
        st.error(f"خطأ في البحث الذكي: {str(e)}")
        return pd.DataFrame(), []

def find_similar_dua(query, vectorizer, df):
    """Find the most similar dua to a query"""
    try:
        df_results, similarities = semantic_search(query, vectorizer, df, top_k=1)
        if not df_results.empty and similarities[0] > 0.1:
            return df_results.iloc[0]['category'], df_results.iloc[0]['text']
    except Exception:
        pass
    return None, None

# Sample Adhkar data (enhanced)
@st.cache_data
def load_adhkar_data():
    return [
        {
            "id": 1,
            "arabic": "سُبْحَانَ اللَّهِ وَبِحَمْدِهِ",
            "transliteration": "Subhan Allah wa bihamdihi",
            "translation": "Glory is to Allah and praise is to Him",
            "category": "morning",
            "source": "Sahih Bukhari",
            "reward": "Whoever says this 100 times, his sins will be forgiven even if they are like the foam of the sea",
            "count": 100,
        },
        {
            "id": 2,
            "arabic": "لَا إِلَهَ إِلَّا اللَّهُ وَحْدَهُ لَا شَرِيكَ لَهُ، لَهُ الْمُلْكُ وَلَهُ الْحَمْدُ وَهُوَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ",
            "transliteration": "La ilaha illa Allah wahdahu la sharika lahu, lahu al-mulku wa lahu al-hamdu wa huwa 'ala kulli shay'in qadir",
            "translation": "There is no god but Allah alone, with no partner. His is the dominion and His is the praise, and He is able to do all things",
            "category": "evening",
            "source": "Sahih Muslim",
            "reward": "Whoever says this 10 times, it is as if he freed four slaves from the children of Isma'il",
            "count": 10,
        },
        {
            "id": 3,
            "arabic": "اللَّهُمَّ أَعِنِّي عَلَى ذِكْرِكَ وَشُكْرِكَ وَحُسْنِ عِبَادَتِكَ",
            "transliteration": "Allahumma a'inni 'ala dhikrika wa shukrika wa husni 'ibadatika",
            "translation": "O Allah, help me to remember You, thank You, and worship You in the best manner",
            "category": "general",
            "source": "Abu Dawud",
            "reward": "A comprehensive du'a for spiritual improvement",
            "count": 1,
        },
        {
            "id": 4,
            "arabic": "أَسْتَغْفِرُ اللَّهَ الْعَظِيمَ الَّذِي لَا إِلَهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ وَأَتُوبُ إِلَيْهِ",
            "transliteration": "Astaghfir Allah al-'Azeem alladhi la ilaha illa huwa al-Hayy al-Qayyum wa atubu ilayhi",
            "translation": "I seek forgiveness from Allah the Mighty, whom there is no god but He, the Living, the Eternal, and I repent to Him",
            "category": "istighfar",
            "source": "At-Tirmidhi",
            "reward": "Whoever says this, Allah will forgive him even if he fled from battle",
            "count": 3,
        },
        {
            "id": 5,
            "arabic": "بِسْمِ اللَّهِ الَّذِي لَا يَضُرُّ مَعَ اسْمِهِ شَيْءٌ فِي الْأَرْضِ وَلَا فِي السَّمَاءِ وَهُوَ السَّمِيعُ الْعَلِيمُ",
            "transliteration": "Bismillah alladhi la yadurru ma'a ismihi shay'un fi al-ardi wa la fi as-sama'i wa huwa as-Sami' al-'Alim",
            "translation": "In the name of Allah, with whose name nothing on earth or in heaven can cause harm, and He is the All-Hearing, All-Knowing",
            "category": "protection",
            "source": "Abu Dawud",
            "reward": "Protection from harm when said 3 times in morning and evening",
            "count": 3,
        },
    ]

def initialize_session_state():
    """Initialize session state variables"""
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'read_counts' not in st.session_state:
        st.session_state.read_counts = {}
    if 'daily_adhkar' not in st.session_state:
        adhkar_data = load_adhkar_data()
        st.session_state.daily_adhkar = random.choice(adhkar_data)
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'search'
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'daily_adhkar_count' not in st.session_state:
        st.session_state.daily_adhkar_count = 0

def get_category_class(category):
    category_classes = {
        'morning': 'badge-morning',
        'evening': 'badge-evening',
        'general': 'badge-general',
        'istighfar': 'badge-istighfar',
        'protection': 'badge-protection',
    }
    return category_classes.get(category, 'badge-general')

def display_adhkar_card(adhkar, featured=False, similarity_score=None):
    """Display a modern adhkar card"""
    card_class = "modern-card featured-card" if featured else "modern-card"
    category_class = get_category_class(adhkar['category'])
    
    is_favorite = adhkar['id'] in st.session_state.favorites
    read_count = st.session_state.read_counts.get(adhkar['id'], 0)
    
    similarity_badge = ""
    if similarity_score is not None:
        similarity_percentage = int(similarity_score * 100)
        similarity_badge = f'<span style="background: var(--emerald-100); color: var(--emerald-700); padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; margin-left: 8px;">تشابه: {similarity_percentage}%</span>'
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <span class="category-badge {category_class}">{adhkar['category']} {similarity_badge}</span>
                <div style="display: flex; gap: 8px;">
                    {'❤️' if is_favorite else '🤍'}
                </div>
            </div>
            
            <div class="arabic-text">{adhkar['arabic']}</div>
            <div class="transliteration">{adhkar['transliteration']}</div>
            <div class="translation">{adhkar['translation']}</div>
        </div>
        
        <div class="card-content">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--gray-600); margin-bottom: 1rem;">
                <span>المصدر: {adhkar['source']}</span>
                <span>العدد المستحب: {adhkar['count']}</span>
            </div>
            
            <div class="reward-box">
                <div class="reward-text">
                    <strong>الفضل:</strong> {adhkar['reward']}
                </div>
            </div>
            
            <div class="card-actions">
                <div style="display: flex; gap: 8px;">
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("📖 قرأت هذا الذكر", key=f"read_{adhkar['id']}"):
            st.session_state.read_counts[adhkar['id']] = st.session_state.read_counts.get(adhkar['id'], 0) + 1
            st.session_state.counter += 1
            st.session_state.daily_adhkar_count += 1
            st.success("✅ تم احتساب القراءة!")
    
    with col2:
        fav_text = "💔 إزالة من المفضلة" if is_favorite else "❤️ إضافة للمفضلة"
        if st.button(fav_text, key=f"fav_{adhkar['id']}"):
            if is_favorite:
                st.session_state.favorites.remove(adhkar['id'])
                st.success("تم إزالة الذكر من المفضلة")
            else:
                st.session_state.favorites.append(adhkar['id'])
                st.success("✅ تم إضافة الذكر للمفضلة!")
    
    with col3:
        if st.button("📋 نسخ", key=f"copy_{adhkar['id']}"):
            copy_text = f"{adhkar['arabic']}\n\n{adhkar['transliteration']}\n\n{adhkar['translation']}"
            st.code(copy_text, language="text")
    
    with col4:
        if st.button("🔍 مشابه", key=f"similar_{adhkar['id']}"):
            st.session_state.current_adhkar_for_similarity = adhkar['arabic']
            st.rerun()
    
    # Display read count if any
    if read_count > 0:
        st.markdown(f"""
                <div style="margin-top: 8px;">
                    <span style="background: var(--emerald-100); color: var(--emerald-700); padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                        قُرئ {read_count} مرة
                    </span>
                </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div></div>", unsafe_allow_html=True)

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

def main():
    # Show warnings after page config if needed
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn not installed. AI features will be disabled. Install with: pip install scikit-learn")
    
    if not JOBLIB_AVAILABLE:
        st.info("ℹ️ joblib not installed. Using pickle as fallback for model loading. Install with: pip install joblib")
    
    # Initialize session state
    initialize_session_state()
    
    # Load AI model and data
    vectorizer, model_df = load_model_and_vectorizer()
    
    # Load sample data as fallback
    adhkar_data = load_adhkar_data()
    
    # Use model data if available, otherwise use sample data
    ai_enabled = False
    if not model_df.empty and vectorizer is not None:
        st.success("🤖 تم تحميل النموذج الذكي بنجاح!")
        # Convert model data to format compatible with display
        converted_data = []
        for idx, row in model_df.iterrows():
            converted_data.append({
                "id": idx + 1,
                "arabic": row.get('text', row.get('clean_text', '')),
                "transliteration": f"Dhikr {idx + 1}",
                "translation": f"Islamic remembrance from {row.get('category', 'general')} category",
                "category": row.get('category', 'general'),
                "source": "Islamic Sources",
                "reward": "Great reward from Allah",
                "count": 1,
            })
        adhkar_data = converted_data
        ai_enabled = True
    else:
        st.warning("⚠️ لم يتم العثور على النموذج الذكي، سيتم استخدام البيانات التجريبية")
    
    # Modern Header
    ai_status = "🤖 مفعل" if ai_enabled else "❌ غير متاح"
    st.markdown(f"""
    <div class="modern-header">
        <div class="header-content">
            <h1 class="header-title">أذكار المسلم الذكي</h1>
            <p class="header-subtitle">Muslim Adhkar AI - الذكاء الاصطناعي: {ai_status}</p>
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 البحث", key="tab_search", use_container_width=True):
            st.session_state.active_tab = 'search'
    
    with col2:
        if st.button("🤖 الذكي", key="tab_ai", use_container_width=True):
            st.session_state.active_tab = 'ai'
    
    with col3:
        if st.button("❤️ المفضلة", key="tab_favorites", use_container_width=True):
            st.session_state.active_tab = 'favorites'
    
    with col4:
        if st.button("📊 الإحصائيات", key="tab_stats", use_container_width=True):
            st.session_state.active_tab = 'stats'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display content based on active tab
    if st.session_state.active_tab == 'search':
        # Search Tab Content
        st.markdown("### 🔍 البحث في الأذكار")
        
        # Modern Search Bar
        search_query = st.text_input(
            "",
            placeholder="ابحث في الأذكار...",
            label_visibility="collapsed",
            key="search_input"
        )
        
        # Filter data based on search
        filtered_adhkar = []
        if search_query.strip():
            for adhkar in adhkar_data:
                searchable_text = f"{adhkar['arabic']} {adhkar['transliteration']} {adhkar['translation']} {adhkar['category']}".lower()
                if search_query.lower() in searchable_text:
                    filtered_adhkar.append(adhkar)
        else:
            filtered_adhkar = adhkar_data
        
        # Display results
        st.markdown(f"**عدد النتائج: {len(filtered_adhkar)}**")
        
        # Display adhkar cards
        for adhkar in filtered_adhkar:
            display_adhkar_card(adhkar)
    
    elif st.session_state.active_tab == 'ai':
        # AI Search Tab Content
        st.markdown("### 🤖 البحث الذكي بالذكاء الاصطناعي")
        
        if ai_enabled:
            st.success("✅ النموذج الذكي جاهز للاستخدام")
        else:
            st.warning("⚠️ الميزات الذكية غير متاحة. تأكد من تثبيت المكتبات المطلوبة.")
        
        # Smart Dua Finder
        st.markdown("#### 🎯 البحث عن الدعاء المناسب")
        user_dua = st.text_input(
            "🤲 أدخل دعاءك أو وصف حالتك:", 
            placeholder="مثال: اللهم اغفر لي، أريد الحماية، أشعر بالخوف...",
            key="dua_input"
        )
        
        if user_dua:
            if ai_enabled:
                with st.spinner("🤖 جاري البحث عن الدعاء المناسب..."):
                    category, similar_text = find_similar_dua(user_dua, vectorizer, model_df)
                    
                    if similar_text:
                        st.success(f"✨ تم العثور على دعاء مناسب في فئة: **{category}**")
                        # Create a temporary adhkar entry for display
                        similar_adhkar = {
                            "id": 9999,
                            "arabic": similar_text,
                            "transliteration": "AI Recommended",
                            "translation": "Smart match based on your description",
                            "category": category,
                            "source": "AI Semantic Search",
                            "reward": "Recommended by our AI model",
                            "count": 1
                        }
                        display_adhkar_card(similar_adhkar, featured=True)
                    else:
                        st.info("❌ لم يتم العثور على دعاء مناسب. جرب وصفًا آخر.")
            else:
                st.warning("الميزات الذكية غير متاحة. يرجى تثبيت المكتبات المطلوبة.")
        
        st.markdown("---")
        
        # Semantic search
        st.markdown("#### 🧠 البحث الدلالي المتقدم")
        semantic_query = st.text_input(
            "ابحث بالمعنى", 
            placeholder="مثال: الحماية من الشر، الدعاء للوالدين، الاستغفار...",
            key="semantic_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            search_depth = st.selectbox("عدد النتائج", [3, 5, 8, 10], index=1, key="depth_select")
        with col2:
            min_similarity = st.slider("حد التشابه الأدنى", 0.1, 0.8, 0.2, 0.1, key="similarity_slider")
        
        if semantic_query:
            if ai_enabled:
                with st.spinner("🤖 جاري البحث الذكي..."):
                    semantic_results, similarities = semantic_search(
                        semantic_query, vectorizer, model_df, top_k=search_depth
                    )
                    
                    if not semantic_results.empty:
                        # Filter by minimum similarity
                        valid_results = []
                        for idx, (_, row) in enumerate(semantic_results.iterrows()):
                            if similarities[idx] >= min_similarity:
                                valid_results.append({
                                    "id": row.name,
                                    "arabic": row.get('text', ''),
                                    "transliteration": f"نتيجة ذكية #{idx+1}",
                                    "translation": f"التشابه: {similarities[idx]*100:.1f}%",
                                    "category": row.get('category', 'unknown'),
                                    "source": "AI Semantic Search",
                                    "reward": "Recommended by our AI model",
                                    "count": 1,
                                    "similarity": similarities[idx]
                                })
                        
                        if valid_results:
                            st.success(f"🎯 تم العثور على {len(valid_results)} نتيجة ذكية")
                            for result in valid_results:
                                display_adhkar_card(result, similarity_score=result['similarity'])
                        else:
                            st.warning("لم يتم العثور على نتائج تتجاوز حد التشابه المحدد")
                    else:
                        st.info("لم يتم العثور على نتائج. جرب كلمات مختلفة.")
            else:
                st.warning("الميزات الذكية غير متاحة. يرجى تثبيت المكتبات المطلوبة.")
    
    elif st.session_state.active_tab == 'favorites':
        # Favorites Tab Content
        st.markdown("### ❤️ الأذكار المفضلة")
        
        favorite_adhkar = [adhkar for adhkar in adhkar_data if adhkar['id'] in st.session_state.favorites]
        
        if not favorite_adhkar:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">❤️</div>
                <h3>لا توجد أذكار مفضلة</h3>
                <p>أضف أذكارك المفضلة من قسم البحث لتظهر هنا</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"لديك {len(favorite_adhkar)} ذكر في المفضلة")
            for adhkar in favorite_adhkar:
                display_adhkar_card(adhkar)
    
    elif st.session_state.active_tab == 'stats':
        # Statistics Tab Content
        st.markdown("### 📊 الإحصائيات والتحليلات")
        
        total_reads = sum(st.session_state.read_counts.values())
        favorite_count = len(st.session_state.favorites)
        total_adhkar = len(adhkar_data)
        
        # Stats Cards
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{st.session_state.counter}</div>
                <div class="stat-label">إجمالي القراءات</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-number">{favorite_count}</div>
                <div class="stat-label">الأذكار المفضلة</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{st.session_state.daily_adhkar_count}</div>
                <div class="stat-label">قراءات اليوم</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-number">{total_adhkar}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Most Read Adhkar
        if st.session_state.read_counts:
            st.markdown("#### 📈 الأذكار الأكثر قراءة")
            
            # Sort by read count
            sorted_reads = sorted(st.session_state.read_counts.items(), key=lambda x: x[1], reverse=True)
            
            for adhkar_id, count in sorted_reads[:5]:
                adhkar = next((a for a in adhkar_data if a['id'] == adhkar_id), None)
                if adhkar:
                    st.markdown(f"""
                    <div class="modern-card" style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem;">
                            <div style="flex: 1;">
                                <p class="arabic-text" style="font-size: 1.3rem; margin-bottom: 0.5rem;">{adhkar['arabic']}</p>
                                <p style="color: var(--gray-600); font-size: 0.9rem; margin: 0;">{adhkar['translation']}</p>
                            </div>
                            <div style="margin-left: 1rem;">
                                <span style="background: var(--emerald-100); color: var(--emerald-700); padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                                    {count} مرة
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📖 ابدأ بقراءة بعض الأذكار لترى الإحصائيات هنا")

if __name__ == "__main__":
    main()
