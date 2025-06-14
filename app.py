import streamlit as st

# Page configuration
st.set_page_config(
    page_title="أذكار المسلم - Islamic Adhkar",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
import re

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Modern app-like CSS with specified colors
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    :root {{
        --bg-primary: #333446;
        --bg-secondary: #7F8CAA;
        --accent: #B8CFCE;
        --text-primary: #FFFFFF;
        --text-secondary: #B8CFCE;
        --card-bg: rgba(127, 140, 170, 0.1);
        --card-border: rgba(184, 207, 206, 0.3);
    }}
    
    /* Main App Container */
    .stApp {{
        background: #333446;
        max-width: 500px;
        margin: 0 auto;
        padding: 0;
        min-height: 100vh;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Top Header with Search */
    .app-header {{
        background: #333446;
        position: sticky;
        top: 0;
        z-index: 1000;
        padding: 1rem 1rem 0.5rem 1rem;
        border-bottom: 1px solid rgba(184, 207, 206, 0.2);
    }}
    
    .app-title-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }}
    
    .app-logo {{
        width: 40px;
        height: 40px;
        background: #7F8CAA;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #333446;
        font-weight: bold;
        font-size: 1.2rem;
    }}
    
    .app-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: #B8CFCE;
        font-family: 'Amiri', serif;
        margin: 0;
        flex: 1;
        text-align: center;
    }}
    
    .menu-icon {{
        width: 40px;
        height: 40px;
        background: transparent;
        border: 1px solid #B8CFCE;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #B8CFCE;
        cursor: pointer;
    }}
    
    /* Search Bar */
    .search-container {{
        background: rgba(127, 140, 170, 0.2);
        border-radius: 16px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border: 1px solid rgba(184, 207, 206, 0.2);
        margin-bottom: 0.5rem;
    }}
    
    .search-icon {{
        color: #B8CFCE;
        font-size: 1.1rem;
    }}
    
    .search-input {{
        background: transparent;
        border: none;
        color: #FFFFFF;
        flex: 1;
        outline: none;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }}
    
    .search-input::placeholder {{
        color: rgba(184, 207, 206, 0.6);
    }}
    
    /* Content Area */
    .content-area {{
        padding: 1rem;
        padding-bottom: 80px;
        overflow-y: auto;
    }}
    
    /* Cards */
    .card {{
        background: rgba(127, 140, 170, 0.15);
        padding: 1.25rem;
        border-radius: 20px;
        margin: 0.75rem 0;
        border: 1px solid rgba(184, 207, 206, 0.2);
        direction: rtl;
        text-align: right;
        backdrop-filter: blur(10px);
    }}
    
    .card-active {{
        background: rgba(127, 140, 170, 0.25);
        border: 1px solid rgba(184, 207, 206, 0.4);
    }}
    
    /* Adhkar Text */
    .adhkar-text {{
        font-size: 1.1rem;
        line-height: 1.8;
        color: #FFFFFF;
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
        font-weight: 400;
    }}
    
    /* Category Badge */
    .category-badge {{
        background: #7F8CAA;
        color: #333446;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        font-family: 'Inter', sans-serif;
        margin-top: 0.5rem;
    }}
    
    /* Action Buttons */
    .action-buttons {{
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        direction: ltr;
    }}
    
    .action-btn {{
        flex: 1;
        background: rgba(184, 207, 206, 0.2);
        border: 1px solid rgba(184, 207, 206, 0.3);
        color: #B8CFCE;
        padding: 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        text-align: center;
        transition: all 0.2s ease;
    }}
    
    .action-btn:hover {{
        background: rgba(184, 207, 206, 0.3);
        border-color: #B8CFCE;
        color: #FFFFFF;
    }}
    
    /* Counter Cards */
    .counter-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 1rem 0;
    }}
    
    .counter-card {{
        background: rgba(127, 140, 170, 0.15);
        padding: 1.25rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(184, 207, 206, 0.2);
    }}
    
    .counter-label {{
        color: #B8CFCE;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.25rem;
    }}
    
    .counter-number {{
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        margin: 0.25rem 0;
    }}
    
    .counter-desc {{
        color: #B8CFCE;
        font-size: 0.7rem;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Feature Card */
    .feature-card {{
        background: linear-gradient(135deg, rgba(127, 140, 170, 0.2), rgba(184, 207, 206, 0.1));
        padding: 1.25rem;
        border-radius: 20px;
        margin: 0.75rem 0;
        border: 1px solid rgba(184, 207, 206, 0.3);
    }}
    
    /* Bottom Navigation */
    .nav-container {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #333446;
        border-top: 1px solid rgba(184, 207, 206, 0.2);
        padding: 0.5rem;
        display: flex;
        justify-content: space-around;
        z-index: 1000;
    }}
    
    .nav-item {{
        flex: 1;
        text-align: center;
        padding: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border-radius: 12px;
    }}
    
    .nav-item:hover {{
        background: rgba(127, 140, 170, 0.2);
    }}
    
    .nav-item.active {{
        background: rgba(127, 140, 170, 0.3);
    }}
    
    .nav-icon {{
        font-size: 1.2rem;
        color: #B8CFCE;
        display: block;
        margin-bottom: 0.25rem;
    }}
    
    .nav-label {{
        font-size: 0.7rem;
        color: #B8CFCE;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Streamlit Overrides */
    .stButton > button {{
        background: rgba(184, 207, 206, 0.2);
        color: #B8CFCE;
        border: 1px solid rgba(184, 207, 206, 0.3);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        width: 100%;
    }}
    
    .stButton > button:hover {{
        background: rgba(184, 207, 206, 0.3);
        border-color: #B8CFCE;
        color: #FFFFFF;
    }}
    
    .stTextInput > div > div > input {{
        background: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }}
    
    .stTextInput > div {{
        background: rgba(127, 140, 170, 0.2) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(184, 207, 206, 0.2) !important;
    }}
    
    .stSelectbox > div > div {{
        background: rgba(127, 140, 170, 0.2) !important;
        border: 1px solid rgba(184, 207, 206, 0.2) !important;
        color: #FFFFFF !important;
    }}
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {{
        background: rgba(127, 140, 170, 0.2) !important;
        color: #B8CFCE !important;
        border: 1px solid rgba(184, 207, 206, 0.3) !important;
        border-radius: 12px !important;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        color: #B8CFCE !important;
    }}
    
    p, div, span {{
        color: #B8CFCE;
    }}
    
    /* Greeting Card */
    .greeting-card {{
        background: linear-gradient(135deg, rgba(127, 140, 170, 0.25), rgba(184, 207, 206, 0.15));
        padding: 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.75rem 0;
        border: 1px solid rgba(184, 207, 206, 0.3);
    }}
    
    .greeting-card h3 {{
        color: #FFFFFF !important;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }}
    
    /* Section Headers */
    .section-header {{
        color: #B8CFCE;
        font-size: 1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }}
</style>
""", unsafe_allow_html=True)

# Clean Arabic text
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
    """Load model and data"""
    try:
        if JOBLIB_AVAILABLE:
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            df = pd.read_csv("adhkar_df.csv")
            return vectorizer, df
        else:
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
    """Perform semantic search"""
    try:
        if vectorizer is None:
            return pd.DataFrame(), []
        
        query_vector = vectorizer.transform([query])
        tfidf_matrix = vectorizer.transform(df['clean_text'])
        similarities = manual_cosine_similarity(query_vector, tfidf_matrix)
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        meaningful_indices = [idx for idx, sim in zip(top_indices, top_similarities) if sim > 0.1]
        meaningful_similarities = [sim for sim in top_similarities if sim > 0.1]
        
        if not meaningful_indices:
            return pd.DataFrame(), []
            
        result_df = df.iloc[meaningful_indices].copy()
        return result_df, meaningful_similarities
        
    except Exception as e:
        st.error(f"خطأ في البحث الدلالي: {e}")
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
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Reset daily counter if it's a new day
    if st.session_state.last_date != datetime.now().date():
        st.session_state.daily_adhkar_count = 0
        st.session_state.last_date = datetime.now().date()

def display_adhkar_card(adhkar_text, category, index, similarity_score=None, is_similar=False):
    """Display a single adhkar card"""
    card_class = "card-active" if is_similar else ""
    
    similarity_badge = ""
    if similarity_score is not None:
        similarity_percentage = int(similarity_score * 100)
        similarity_badge = f' • تشابه {similarity_percentage}%'
    
    st.markdown(f"""
    <div class="card {card_class}">
        <div class="adhkar-text">{adhkar_text}</div>
        <div class="category-badge">{category}{similarity_badge}</div>
        <div class="action-buttons">
            <div class="action-btn" onclick="console.log('read')">📖 قراءة</div>
            <div class="action-btn" onclick="console.log('favorite')">⭐ مفضلة</div>
            <div class="action-btn" onclick="console.log('copy')">📋 نسخ</div>
            <div class="action-btn" onclick="console.log('similar')">🔍 مشابه</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hidden buttons for functionality
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("", key=f"read_{index}", help="قراءة"):
            st.session_state.counter += 1
            st.session_state.daily_adhkar_count += 1
            st.success("✅ تم احتساب القراءة")
    
    with col2:
        if st.button("", key=f"fav_{index}", help="مفضلة"):
            if adhkar_text not in st.session_state.favorite_adhkar:
                st.session_state.favorite_adhkar.append(adhkar_text)
                st.success("✅ تم إضافة للمفضلة")
    
    with col3:
        if st.button("", key=f"copy_{index}", help="نسخ"):
            st.code(adhkar_text, language="text")
    
    with col4:
        if SKLEARN_AVAILABLE and st.button("", key=f"similar_{index}", help="مشابه"):
            st.session_state.current_adhkar_for_similarity = adhkar_text
            st.rerun()

def render_header(df):
    """Render the app header with search"""
    st.markdown("""
    <div class="app-header">
        <div class="app-title-bar">
            <div class="app-logo">ب</div>
            <h1 class="app-title">بيان - أذكار المسلم</h1>
            <div class="menu-icon">☰</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    search_container = st.container()
    with search_container:
        search_query = st.text_input(
            "",
            placeholder="🔍 ابحث في الأذكار...",
            key="global_search",
            label_visibility="collapsed"
        )
        
        if search_query:
            st.session_state.search_query = search_query
            return search_query
    
    return ""

def render_navigation():
    """Render bottom navigation"""
    current_page = st.session_state.current_page
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 الرئيسية", key="nav_home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        if st.button("⭐ المفضلة", key="nav_favorites", use_container_width=True):
            st.session_state.current_page = 'favorites'
            st.rerun()
    
    with col3:
        if st.button("⚙️ الإعدادات", key="nav_settings", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()

def home_page(df, vectorizer, search_query=""):
    """Home page content"""
    # Filter by search if query exists
    if search_query:
        filtered_df = df[
            df['clean_text'].str.contains(search_query, na=False) |
            df['category'].str.contains(search_query, na=False)
        ]
        st.markdown(f'<div class="section-header">نتائج البحث ({len(filtered_df)})</div>', unsafe_allow_html=True)
        
        for idx, row in filtered_df.head(10).iterrows():
            display_adhkar_card(row['clean_text'], row['category'], f"search_{idx}")
        
        return
    
    # Time-based greeting
    greeting, time_period = get_time_based_greeting()
    st.markdown(f"""
    <div class="greeting-card">
        <h3>{greeting}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Counters
    st.markdown("""
    <div class="counter-grid">
        <div class="counter-card">
            <div class="counter-label">📅 اليوم</div>
            <div class="counter-number">{}</div>
            <div class="counter-desc">أذكار مقروءة</div>
        </div>
        <div class="counter-card">
            <div class="counter-label">📊 الإجمالي</div>
            <div class="counter-number">{}</div>
            <div class="counter-desc">أذكار كاملة</div>
        </div>
    </div>
    """.format(st.session_state.daily_adhkar_count, st.session_state.counter), unsafe_allow_html=True)
    
    # Random adhkar
    st.markdown('<div class="section-header">ذكر اليوم</div>', unsafe_allow_html=True)
    
    if st.button("🎯 احصل على ذكر جديد", use_container_width=True):
        st.session_state.random_adhkar = df.sample(1).iloc[0]
    
    if 'random_adhkar' not in st.session_state:
        st.session_state.random_adhkar = df.sample(1).iloc[0]
    
    adhkar = st.session_state.random_adhkar
    display_adhkar_card(adhkar['clean_text'], adhkar['category'], "daily_adhkar")
    
    # AI search if available
    if SKLEARN_AVAILABLE and vectorizer is not None:
        st.markdown('<div class="section-header">🤖 البحث الذكي</div>', unsafe_allow_html=True)
        user_dua = st.text_input(
            "",
            placeholder="اكتب حالتك أو طلبك...",
            key="ai_search_home",
            label_visibility="collapsed"
        )
        
        if user_dua:
            with st.spinner("جاري البحث..."):
                category, similar_text = find_similar_dua(user_dua, vectorizer, df)
                
                if similar_text:
                    st.success(f"تم العثور على دعاء مناسب")
                    display_adhkar_card(similar_text, category, "ai_result", is_similar=True)

def favorites_page(df, vectorizer):
    """Favorites page content"""
    st.markdown('<div class="section-header">⭐ الأذكار المفضلة</div>', unsafe_allow_html=True)
    
    if st.session_state.favorite_adhkar:
        st.info(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
        
        for i, adhkar in enumerate(st.session_state.favorite_adhkar):
            display_adhkar_card(adhkar, "مفضلة", f"fav_item_{i}")
        
        if st.button("🗑️ مسح جميع المفضلة", use_container_width=True):
            st.session_state.favorite_adhkar = []
            st.success("تم مسح جميع الأذكار المفضلة")
            st.rerun()
    else:
        st.info("لا توجد أذكار مفضلة حتى الآن")

def settings_page():
    """Settings page content"""
    st.markdown('<div class="section-header">⚙️ الإعدادات</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <div class="adhkar-text">تطبيق بيان - أذكار المسلم</div>
        <div class="category-badge">الإصدار 1.0</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset counters
    if st.button("🔄 إعادة تعيين العدادات", use_container_width=True):
        st.session_state.counter = 0
        st.session_state.daily_adhkar_count = 0
        st.success("تم إعادة تعيين العدادات")
        st.rerun()

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data and model
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()
        if df.empty:
            st.error("لا يمكن تحميل البيانات")
            return
    
    # Render header with search
    search_query = render_header(df)
    
    # Main content area
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    # Page routing
    if st.session_state.current_page == 'home':
        home_page(df, vectorizer, search_query)
    elif st.session_state.current_page == 'favorites':
        favorites_page(df, vectorizer)
    elif st.session_state.current_page == 'settings':
        settings_page()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom navigation
    render_navigation()

if __name__ == "__main__":
    main()
