import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="أذكار المسلم - Islamic Adhkar AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Modern iOS-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: #1d1d1f;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-left: 12px;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .adhkar-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        direction: rtl;
        text-align: right;
        transition: all 0.3s ease;
    }
    
    .adhkar-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .similar-adhkar-card {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1), rgba(255, 59, 48, 0.1));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(255, 149, 0, 0.2);
        margin-bottom: 16px;
        border: 1px solid rgba(255, 149, 0, 0.3);
        direction: rtl;
        text-align: right;
        transition: all 0.3s ease;
    }
    
    .similar-adhkar-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(255, 149, 0, 0.3);
    }
    
    .adhkar-text {
        font-size: 18px;
        line-height: 1.8;
        color: #1d1d1f;
        font-family: 'Amiri', serif;
        margin-bottom: 16px;
        font-weight: 400;
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #34C759, #30D158);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    .category-tag {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white;
        padding: 8px 16px;
        border-radius: 16px;
        font-size: 14px;
        font-weight: 500;
        display: inline-block;
        margin-top: 12px;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    .search-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .ai-search-container {
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.1), rgba(88, 86, 214, 0.1));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #1d1d1f;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(0, 122, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 122, 255, 0.1);
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .time-based-greeting {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1), rgba(255, 204, 0, 0.1));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        color: #1d1d1f;
        border: 1px solid rgba(255, 149, 0, 0.2);
        box-shadow: 0 4px 20px rgba(255, 149, 0, 0.1);
    }
    
    .counter-display {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin: 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .counter-number {
        font-size: 48px;
        font-weight: 700;
        color: #007AFF;
        margin: 16px 0;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    .random-adhkar {
        background: linear-gradient(135deg, rgba(175, 82, 222, 0.1), rgba(255, 45, 85, 0.1));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        margin: 16px 0;
        text-align: center;
        color: #1d1d1f;
        border: 1px solid rgba(175, 82, 222, 0.2);
        box-shadow: 0 4px 20px rgba(175, 82, 222, 0.1);
    }
    
    .stat-box {
        background: linear-gradient(135deg, #FF2D92, #FF6B35);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 45, 146, 0.3);
        min-width: 150px;
        font-family: 'SF Pro Display', sans-serif;
    }
    
    .installation-guide {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1), rgba(255, 59, 48, 0.1));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 16px;
        margin: 16px 0;
        border-left: 4px solid #FF9500;
        border: 1px solid rgba(255, 149, 0, 0.2);
        box-shadow: 0 4px 20px rgba(255, 149, 0, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'SF Pro Display', sans-serif;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 122, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 122, 255, 0.4);
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=SF+Pro+Display:wght@300;400;500;600;700&display=swap" rel="stylesheet">
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

def display_adhkar_card(adhkar_text, category, index, similarity_score=None, is_similar=False):
    """Display a single adhkar card with optional similarity score"""
    card_class = "similar-adhkar-card" if is_similar else "adhkar-card"
    
    similarity_badge = ""
    if similarity_score is not None:
        similarity_percentage = int(similarity_score * 100)
        similarity_badge = f'<span class="similarity-score">تشابه: {similarity_percentage}%</span>'
    
    with st.container():
        st.markdown(f"""
        <div class="{card_class}">
            <div class="adhkar-text">{adhkar_text}</div>
            <div class="category-tag">{category}{similarity_badge}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("📖 قراءة", key=f"read_{index}"):
                st.session_state.counter += 1
                st.session_state.daily_adhkar_count += 1
                st.success("✅ تم احتساب القراءة")
        
        with col2:
            if st.button("❤️ مفضلة", key=f"fav_{index}"):
                if adhkar_text not in st.session_state.favorite_adhkar:
                    st.session_state.favorite_adhkar.append(adhkar_text)
                    st.success("✅ تم إضافة الذكر للمفضلة")
                else:
                    st.info("هذا الذكر موجود بالفعل في المفضلة")
        
        with col3:
            if SKLEARN_AVAILABLE and st.button("🔍 مشابه", key=f"similar_{index}"):
                st.session_state.current_adhkar_for_similarity = adhkar_text
                st.rerun()
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{index}"):
                st.code(adhkar_text, language="text")

def show_installation_guide():
    """Show installation guide for missing dependencies"""
    st.markdown("""
    <div class="installation-guide">
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
    - 🤖 البحث الذكي بالمعنى
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
    
    # Main header
    ai_status = "🤖 مفعل" if (SKLEARN_AVAILABLE and vectorizer is not None) else "❌ غير متاح"
    st.markdown(f"""
    <div class="main-header">
        <h1>🕌 أذكار المسلم الذكي</h1>
        <h2>Islamic Adhkar AI</h2>
        <p style="font-size: 18px; margin-top: 16px;">اذكروا الله كثيراً لعلكم تفلحون</p>
        <span class="ai-badge">الذكاء الاصطناعي: {ai_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Time-based greeting
    greeting, time_period = get_time_based_greeting()
    st.markdown(f"""
    <div class="time-based-greeting">
        <h3>{greeting}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🤖 النموذج الذكي</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if SKLEARN_AVAILABLE and vectorizer is not None:
            st.success("✅ النموذج جاهز للاستخدام")
            vocab_size = len(vectorizer.get_feature_names_out())
            st.info(f"📊 حجم المفردات: {vocab_size:,} كلمة")
            if JOBLIB_AVAILABLE:
                st.info("🔧 تم التحميل باستخدام joblib")
            else:
                st.info("🔧 تم التحميل باستخدام pickle")
        elif SKLEARN_AVAILABLE:
            st.warning("⚠️ النموذج غير محمل")
        else:
            st.error("❌ scikit-learn غير مثبت")
        
        st.markdown("""
        <div class="sidebar-content">
            <h2>📊 إحصائياتك اليومية</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Daily statistics
        st.markdown(f"""
        <div class="counter-display">
            <h3>🎯 عداد الأذكار اليوم</h3>
            <div class="counter-number">{st.session_state.daily_adhkar_count}</div>
            <p>ذكر مقروء اليوم</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="counter-display">
            <h3>📈 إجمالي الأذكار</h3>
            <div class="counter-number">{st.session_state.counter}</div>
            <p>إجمالي الأذكار المقروءة</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset counter
        if st.button("🔄 إعادة تعيين العداد"):
            st.session_state.counter = 0
            st.session_state.daily_adhkar_count = 0
            st.success("تم إعادة تعيين العداد")
        
        # Random adhkar
        st.markdown("""
        <div class="sidebar-content">
            <h3>🎯 ذكر مقترح</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 احصل على ذكر"):
            if SKLEARN_AVAILABLE and vectorizer is not None:
                # Get time-based recommendation
                current_hour = datetime.now().hour
                if 5 <= current_hour < 12:
                    query = "صباح"
                elif 18 <= current_hour < 22:
                    query = "مساء"
                else:
                    query = "نوم"
                
                smart_results, similarities = semantic_search(query, vectorizer, df, top_k=1)
                if not smart_results.empty:
                    smart_adhkar = smart_results.iloc[0]
                    st.markdown(f"""
                    <div class="random-adhkar">
                        <div class="adhkar-text">{smart_adhkar['clean_text']}</div>
                        <div class="category-tag">{smart_adhkar['category']} 
                        <span class="similarity-score">ذكي 🤖</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    random_adhkar = df.sample(1).iloc[0]
                    st.markdown(f"""
                    <div class="random-adhkar">
                        <div class="adhkar-text">{random_adhkar['clean_text']}</div>
                        <div class="category-tag">{random_adhkar['category']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                random_adhkar = df.sample(1).iloc[0]
                st.markdown(f"""
                <div class="random-adhkar">
                    <div class="adhkar-text">{random_adhkar['clean_text']}</div>
                    <div class="category-tag">{random_adhkar['category']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content tabs
    if SKLEARN_AVAILABLE and vectorizer is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 البحث الذكي", 
            "🔍 البحث التقليدي", 
            "⭐ المفضلة", 
            "📊 الإحصائيات",
            "ℹ️ حول التطبيق"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 البحث", 
            "⭐ المفضلة", 
            "📊 الإحصائيات",
            "ℹ️ حول التطبيق"
        ])
    
    # AI Search Tab (only if sklearn is available)
    if SKLEARN_AVAILABLE and vectorizer is not None:
        with tab1:
            st.markdown("""
            <div class="ai-search-container">
                <h3>🤖 البحث الذكي بالذكاء الاصطناعي</h3>
                <p>ابحث بالمعنى وليس فقط بالكلمات الدقيقة</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Smart Dua Finder
            st.markdown("### 🎯 البحث عن الدعاء المناسب")
            user_dua = st.text_input(
                "🤲 أدخل دعاءك أو وصف حالتك:", 
                placeholder="مثال: اللهم اغفر لي، أريد الحماية، أشعر بالخوف...",
                help="النموذج سيجد الدعاء الأنسب لحالتك"
            )
            
            if user_dua:
                with st.spinner("🤖 جاري البحث عن الدعاء المناسب..."):
                    category, similar_text = find_similar_dua(user_dua, vectorizer, df)
                    
                    if similar_text:
                        st.success(f"✨ تم العثور على دعاء مناسب في فئة: **{category}**")
                        st.markdown(f"""
                        <div class="similar-adhkar-card">
                            <div class="adhkar-text">{similar_text}</div>
                            <div class="category-tag">{category} <span class="similarity-score">مناسب 🎯</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(category)
            
            st.markdown("---")
            
            # Semantic search
            semantic_query = st.text_input(
                "🧠 البحث الدلالي المتقدم", 
                placeholder="مثال: الحماية من الشر، الدعاء للوالدين، الاستغفار...",
                help="ابحث بالمعنى - النموذج سيفهم قصدك"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                search_depth = st.selectbox("عمق البحث", [3, 5, 8, 10], index=1)
            with col2:
                min_similarity = st.slider("حد التشابه الأدنى", 0.1, 0.8, 0.2, 0.1)
            
            if semantic_query:
                with st.spinner("🤖 جاري البحث الذكي..."):
                    semantic_results, similarities = semantic_search(
                        semantic_query, vectorizer, df, top_k=search_depth
                    )
                    
                    if not semantic_results.empty:
                        # Filter by minimum similarity
                        valid_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
                        if valid_indices:
                            filtered_results = semantic_results.iloc[valid_indices]
                            filtered_similarities = [similarities[i] for i in valid_indices]
                            
                            st.success(f"🎯 تم العثور على {len(filtered_results)} نتيجة ذكية")
                            
                            for idx, (_, row) in enumerate(filtered_results.iterrows()):
                                display_adhkar_card(
                                    row['clean_text'], 
                                    row['category'], 
                                    f"semantic_{idx}",
                                    similarity_score=filtered_similarities[idx],
                                    is_similar=True
                                )
                        else:
                            st.warning("لم يتم العثور على نتائج تتجاوز حد التشابه المحدد")
                    else:
                        st.info("لم يتم العثور على نتائج. جرب كلمات مختلفة.")
            
            # Quick semantic search buttons
            st.markdown("### 🚀 بحث سريع")
            quick_searches = [
                "الحماية والأمان", "الدعاء للوالدين", "الاستغفار والتوبة", 
                "الحمد والشكر", "طلب الهداية", "دعاء المريض"
            ]
            
            cols = st.columns(3)
            for i, quick_search in enumerate(quick_searches):
                with cols[i % 3]:
                    if st.button(quick_search, key=f"quick_{i}"):
                        semantic_results, similarities = semantic_search(
                            quick_search, vectorizer, df, top_k=3
                        )
                        if not semantic_results.empty:
                            st.write(f"**نتائج: {quick_search}**")
                            for idx, (_, row) in enumerate(semantic_results.iterrows()):
                                display_adhkar_card(
                                    row['clean_text'], 
                                    row['category'], 
                                    f"quick_{i}_{idx}",
                                    similarity_score=similarities[idx],
                                    is_similar=True
                                )
        
        traditional_tab = tab2
        favorites_tab = tab3
        analytics_tab = tab4
        about_tab = tab5
    else:
        traditional_tab = tab1
        favorites_tab = tab2
        analytics_tab = tab3
        about_tab = tab4
    
    # Traditional Search Tab
    with traditional_tab:
        st.markdown("""
        <div class="search-container">
            <h3>🔍 البحث في الأذكار</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Traditional search and filter options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("🔍 ابحث في الأذكار", placeholder="اكتب كلمة للبحث...")
        
        with col2:
            categories = ['الكل'] + list(df['category'].unique())
            selected_category = st.selectbox("📂 اختر الفئة", categories)
        
        # Filter data based on search and category
        filtered_df = df.copy()
        
        if search_query:
            filtered_df = filtered_df[
                filtered_df['clean_text'].str.contains(search_query, na=False) |
                filtered_df['category'].str.contains(search_query, na=False)
            ]
        
        if selected_category != 'الكل':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        # Display results
        st.markdown(f"**عدد النتائج: {len(filtered_df)}**")
        
        # Pagination
        items_per_page = 5
        total_pages = max(1, len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0))
        
        if total_pages > 1:
            page = st.selectbox("📄 الصفحة", range(1, total_pages + 1))
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = filtered_df.iloc[start_idx:end_idx]
        else:
            page_df = filtered_df
        
        # Display adhkar cards
        for idx, row in page_df.iterrows():
            display_adhkar_card(row['clean_text'], row['category'], f"trad_{idx}")
    
    # Favorites Tab
    with favorites_tab:
        st.markdown("## ⭐ الأذكار المفضلة")
        
        if st.session_state.favorite_adhkar:
            st.success(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
            
            # AI-powered similar favorites (only if sklearn available)
            if SKLEARN_AVAILABLE and vectorizer is not None:
                if st.button("🤖 اقتراحات ذكية بناءً على المفضلة"):
                    all_suggestions = []
                    for fav_adhkar in st.session_state.favorite_adhkar[:3]:
                        similar_results, similarities = find_similar_adhkar(
                            fav_adhkar, vectorizer, df, top_k=2
                        )
                        if not similar_results.empty:
                            for idx, (_, row) in enumerate(similar_results.iterrows()):
                                if row['clean_text'] not in st.session_state.favorite_adhkar:
                                    all_suggestions.append((row, similarities[idx]))
                    
                    if all_suggestions:
                        st.markdown("### 🤖 اقتراحات ذكية بناءً على مفضلتك:")
                        for idx, (row, sim) in enumerate(all_suggestions[:5]):
                            display_adhkar_card(
                                row['clean_text'], 
                                row['category'], 
                                f"fav_suggest_{idx}",
                                similarity_score=sim,
                                is_similar=True
                            )
                    else:
                        st.info("لا توجد اقتراحات ذكية متاحة حالياً")
            
            st.markdown("---")
            st.markdown("### 📚 أذكارك المفضلة:")
            
            for i, adhkar in enumerate(st.session_state.favorite_adhkar):
                st.markdown(f"""
                <div class="adhkar-card">
                    <div class="adhkar-text">{adhkar}</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"🗑️ حذف من المفضلة", key=f"del_fav_{i}"):
                        st.session_state.favorite_adhkar.remove(adhkar)
                        st.rerun()
                
                with col2:
                    if SKLEARN_AVAILABLE and vectorizer is not None:
                        if st.button(f"🔍 أذكار مشابهة", key=f"similar_fav_{i}"):
                            similar_results, similarities = find_similar_adhkar(
                                adhkar, vectorizer, df, top_k=3
                            )
                            if not similar_results.empty:
                                st.markdown(f"**أذكار مشابهة ل:** {adhkar[:50]}...")
                                for idx, (_, row) in enumerate(similar_results.iterrows()):
                                    display_adhkar_card(
                                        row['clean_text'], 
                                        row['category'], 
                                        f"similar_to_fav_{i}_{idx}",
                                        similarity_score=similarities[idx],
                                        is_similar=True
                                    )
            
            if st.button("🗑️ مسح جميع المفضلة"):
                st.session_state.favorite_adhkar = []
                st.success("تم مسح جميع الأذكار المفضلة")
                st.rerun()
        else:
            st.info("لا توجد أذكار مفضلة حتى الآن. أضف بعض الأذكار من قسم البحث!")
    
    # Analytics Tab
    with analytics_tab:
        st.markdown("## 📊 إحصائيات وتحليلات")
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <h3>{len(df)}</h3>
                <p>إجمالي الأذكار</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <h3>{len(df['category'].unique())}</h3>
                <p>عدد الفئات</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <h3>{st.session_state.daily_adhkar_count}</h3>
                <p>أذكار اليوم</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <h3>{len(st.session_state.favorite_adhkar)}</h3>
                <p>المفضلة</p>
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
        text_lengths = df['clean_text'].str.len()
        
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
    with about_tab:
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
        - 📱 تصميم حديث مستوحى من iOS
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
        
        st.markdown(f"""
        #### 📚 الفئات المتاحة ({len(df['category'].unique())} فئة):
        """)
        
        categories_list = df['category'].unique()
        for i, category in enumerate(categories_list, 1):
            count = len(df[df['category'] == category])
            st.write(f"{i}. **{category}** ({count} ذكر)")
        
        st.markdown("""
        ---
        
        #### 🛠️ المتطلبات التقنية:
        ```
        streamlit>=1.28.0
        pandas>=1.5.0
        numpy>=1.24.0
        scikit-learn>=1.3.0
        joblib>=1.3.0
        ```
        
        #### 📁 الملفات المطلوبة:
        - `adhkar_df.csv` - بيانات الأذكار
        - `tfidf_vectorizer.pkl` - النموذج الذكي المدرب
        
        #### 🎨 التصميم:
        - تصميم حديث مستوحى من iOS
        - واجهة عربية RTL مع خطوط Amiri
        - تأثيرات Glassmorphism و Backdrop Blur
        - ألوان متدرجة وانتقالات سلسة
        - تجربة مستخدم محسنة للأجهزة المحمولة
        
        ---
        ### 🤲 دعاء
        
        *"اللهم اجعل هذا العمل خالصاً لوجهك الكريم، وانفع به المسلمين في كل مكان"*
        
        **تذكر:** المداومة على الأذكار خير من الانقطاع عنها
        
        ---
        
        #### 🚀 الميزات الجديدة:
        - **البحث الذكي المحسن** باستخدام خوارزمية التشابه المخصصة
        - **البحث عن الدعاء المناسب** - أدخل حالتك واحصل على الدعاء الأنسب
        - **تصميم iOS الحديث** مع تأثيرات بصرية متقدمة
        - **واجهة متجاوبة** تعمل بسلاسة على جميع الأجهزة
        - **تحليلات ذكية** للفئات والنصوص
        - **توصيات مخصصة** بناءً على تفضيلاتك
        
        #### 📞 الدعم التقني:
        - إذا واجهت مشاكل في التثبيت، تأكد من إصدارات المكتبات
        - للمساعدة في النموذج الذكي، تأكد من وجود ملف النموذج
        - التطبيق يعمل بدون الميزات الذكية إذا لم تكن متاحة
        - التصميم محسن للعرض على الأجهزة المحمولة وأجهزة سطح المكتب
        
        #### 🔧 كيفية الاستخدام:
        1. **للبحث العادي**: استخدم تبويب "البحث التقليدي"
        2. **للبحث الذكي**: استخدم تبويب "البحث الذكي" وأدخل وصف حالتك
        3. **للحصول على توصيات**: استخدم الشريط الجانبي للحصول على اقتراحات
        4. **لحفظ المفضلة**: اضغط على زر "مفضلة" في أي ذكر تريد حفظه
        5. **للأذكار المشابهة**: اضغط على زر "مشابه" للعثور على أذكار ذات معنى قريب
        """)

if __name__ == "__main__":
    main()
