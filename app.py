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

# Modern app-like CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    :root {{
        --primary: #2563eb;
        --primary-light: #dbeafe;
        --secondary: #8b5cf6;
        --accent: #10b981;
        --light: #f8fafc;
        --dark: #1e293b;
        --text: #334155;
        --border: #e2e8f0;
    }}
    
    .stApp {{
        background: #f8fafc;
        max-width: 500px;
        margin: 0 auto;
        padding: 0;
    }}
    
    .header {{
        background: white;
        padding: 1rem;
        text-align: center;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        padding: 0.5rem 0;
    }}
    
    .logo-img {{
        width: 80px;
        height: auto;
        border-radius: 16px;
    }}
    
    .app-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.25rem 0;
        color: var(--dark);
        font-family: 'Amiri', serif;
    }}
    
    .app-subtitle {{
        font-size: 0.9rem;
        font-weight: 400;
        margin: 0.25rem 0;
        color: #64748b;
        font-family: 'Inter', sans-serif;
    }}
    
    .card {{
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        direction: rtl;
        text-align: right;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    
    .highlight-card {{
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-left: 4px solid var(--primary);
    }}
    
    .adhkar-text {{
        font-size: 1.15rem;
        line-height: 1.8;
        color: var(--dark);
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
        font-weight: 400;
    }}
    
    .badge {{
        background: var(--accent);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 6px;
        font-family: 'Inter', sans-serif;
    }}
    
    .tag {{
        background: var(--primary);
        color: white;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 0.5rem;
        font-family: 'Inter', sans-serif;
    }}
    
    .section {{
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
    }}
    
    .section-highlight {{
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 4px solid var(--accent);
    }}
    
    .greeting-card {{
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.75rem 0;
        color: #854d0e;
        border-left: 4px solid #f59e0b;
    }}
    
    .counter-card {{
        background: white;
        padding: 1rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid var(--border);
    }}
    
    .counter-number {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }}
    
    .feature-card {{
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        padding: 1rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        color: var(--secondary);
        border-left: 4px solid var(--secondary);
    }}
    
    /* Button Styles */
    .stButton > button {{
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 8px 16px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        transition: all 0.2s ease;
        width: 100%;
        margin: 0.25rem 0;
    }}
    
    .stButton > button:hover {{
        background: #1d4ed8;
        transform: translateY(-1px);
    }}
    
    .nav-button {{
        background: white !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
    }}
    
    /* Navigation */
    .nav-container {{
        display: flex;
        justify-content: space-around;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 0.75rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        z-index: 100;
    }}
    
    .nav-button-container {{
        flex: 1;
        text-align: center;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--dark);
        margin: 0.5rem 0;
    }}
    
    p, div, span {{
        font-family: 'Inter', sans-serif;
        color: var(--text);
        font-size: 0.9rem;
    }}
    
    /* Utility */
    .text-center {{
        text-align: center;
    }}
    
    .mb-1 {{
        margin-bottom: 0.5rem;
    }}
    
    .mt-1 {{
        margin-top: 0.5rem;
    }}
    
    .content-area {{
        padding-bottom: 70px;
    }}
</style>

<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
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
    
    # Reset daily counter if it's a new day
    if st.session_state.last_date != datetime.now().date():
        st.session_state.daily_adhkar_count = 0
        st.session_state.last_date = datetime.now().date()

def display_adhkar_card(adhkar_text, category, index, similarity_score=None, is_similar=False):
    """Display a single adhkar card"""
    card_class = "card highlight-card" if is_similar else "card"
    
    similarity_badge = ""
    if similarity_score is not None:
        similarity_percentage = int(similarity_score * 100)
        similarity_badge = f'<span class="badge">تشابه: {similarity_percentage}%</span>'
    
    with st.container():
        st.markdown(f"""
        <div class="{card_class}">
            <div class="adhkar-text">{adhkar_text}</div>
            <div class="tag">{category}{similarity_badge}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
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
        
        col3, col4 = st.columns(2)
        with col3:
            if SKLEARN_AVAILABLE and st.button("🔍 مشابه", key=f"similar_{index}"):
                st.session_state.current_adhkar_for_similarity = adhkar_text
                st.rerun()
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{index}"):
                st.code(adhkar_text, language="text")

def navigation():
    """Bottom navigation bar"""
    st.markdown("""
    <div class="nav-container">
        <div class="nav-button-container">
            <button class="nav-button" onclick="setPage('home')">🏠 الرئيسية</button>
        </div>
        <div class="nav-button-container">
            <button class="nav-button" onclick="setPage('search')">🔍 بحث</button>
        </div>
        <div class="nav-button-container">
            <button class="nav-button" onclick="setPage('favorites')">⭐ مفضلة</button>
        </div>
    </div>
    
    <script>
    function setPage(page) {{
        Streamlit.setComponentValue(page);
    }}
    </script>
    """, unsafe_allow_html=True)
    
    # Handle navigation
    if 'nav_click' not in st.session_state:
        st.session_state.nav_click = None
        
    nav_click = st.session_state.get('nav_click', None)
    if nav_click:
        st.session_state.current_page = nav_click
        st.session_state.nav_click = None
        st.rerun()

def home_page(df, vectorizer):
    """Home page content"""
    # Time-based greeting
    greeting, time_period = get_time_based_greeting()
    st.markdown(f"""
    <div class="greeting-card">
        <h3>{greeting}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Counters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="counter-card">
            <div>📅 اليوم</div>
            <div class="counter-number">{st.session_state.daily_adhkar_count}</div>
            <div>أذكار مقروءة</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="counter-card">
            <div>📊 الإجمالي</div>
            <div class="counter-number">{st.session_state.counter}</div>
            <div>أذكار مقروءة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Random adhkar
    st.markdown("### ذكر اليوم")
    if st.button("🎯 احصل على ذكر جديد"):
        st.session_state.random_adhkar = df.sample(1).iloc[0]
    
    if 'random_adhkar' not in st.session_state:
        st.session_state.random_adhkar = df.sample(1).iloc[0]
    
    adhkar = st.session_state.random_adhkar
    st.markdown(f"""
    <div class="feature-card">
        <div class="adhkar-text">{adhkar['clean_text']}</div>
        <div class="tag">{adhkar['category']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI search if available
    if SKLEARN_AVAILABLE and vectorizer is not None:
        st.markdown("### 🤖 البحث الذكي")
        user_dua = st.text_input(
            "اكتب حالتك أو طلبك:", 
            placeholder="مثال: اللهم اغفر لي، أريد الحماية...",
            key="ai_search_home"
        )
        
        if user_dua:
            with st.spinner("جاري البحث عن الدعاء المناسب..."):
                category, similar_text = find_similar_dua(user_dua, vectorizer, df)
                
                if similar_text:
                    st.success(f"تم العثور على دعاء مناسب في فئة: **{category}**")
                    st.markdown(f"""
                    <div class="card highlight-card">
                        <div class="adhkar-text">{similar_text}</div>
                        <div class="tag">{category} <span class="badge">مناسب 🎯</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(category)

def search_page(df, vectorizer):
    """Search page content"""
    st.markdown("### 🔍 بحث في الأذكار")
    
    # Search options
    search_query = st.text_input("ابحث في الأذكار", placeholder="اكتب كلمة للبحث...")
    
    categories = ['الكل'] + list(df['category'].unique())
    selected_category = st.selectbox("اختر الفئة", categories, index=0)
    
    # Filter data
    filtered_df = df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['clean_text'].str.contains(search_query, na=False) |
            df['category'].str.contains(search_query, na=False)
        ]
    
    if selected_category != 'الكل':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Display results
    st.markdown(f"**عدد النتائج: {len(filtered_df)}**")
    
    # Display adhkar cards
    for idx, row in filtered_df.head(10).iterrows():
        display_adhkar_card(row['clean_text'], row['category'], f"search_{idx}")

def favorites_page(df, vectorizer):
    """Favorites page content"""
    st.markdown("## ⭐ الأذكار المفضلة")
    
    if st.session_state.favorite_adhkar:
        st.success(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
        
        # Display favorites
        for i, adhkar in enumerate(st.session_state.favorite_adhkar):
            st.markdown(f"""
            <div class="card">
                <div class="adhkar-text">{adhkar}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🗑️ حذف", key=f"del_fav_{i}"):
                    st.session_state.favorite_adhkar.remove(adhkar)
                    st.rerun()
            
            with col2:
                if SKLEARN_AVAILABLE and vectorizer is not None:
                    if st.button(f"🔍 مشابه", key=f"similar_fav_{i}"):
                        similar_results, similarities = semantic_search(adhkar, vectorizer, df, top_k=3)
                        if similar_results:
                            st.markdown("**أذكار مشابهة:**")
                            for idx, (_, row) in enumerate(similar_results.iterrows()):
                                display_adhkar_card(
                                    row['clean_text'], 
                                    row['category'], 
                                    f"similar_{i}_{idx}",
                                    similarity_score=similarities[idx],
                                    is_similar=True
                                )
        
        if st.button("🗑️ مسح جميع المفضلة"):
            st.session_state.favorite_adhkar = []
            st.success("تم مسح جميع الأذكار المفضلة")
            st.rerun()
    else:
        st.info("لا توجد أذكار مفضلة حتى الآن. أضف بعض الأذكار من قسم البحث!")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data and model
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()
        if df.empty:
            st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
            return
    
    # App header
    st.markdown(f"""
    <div class="header">
        <div class="logo-container">
            <img src="https://via.placeholder.com/80x80/2563eb/ffffff?text=BAYAAN" class="logo-img" alt="Bayaan Logo">
        </div>
        <div class="app-title">بيان - أذكار المسلم</div>
        <div class="app-subtitle">اذكروا الله كثيراً لعلكم تفلحون</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    # Page routing
    if st.session_state.current_page == 'home':
        home_page(df, vectorizer)
    elif st.session_state.current_page == 'search':
        search_page(df, vectorizer)
    elif st.session_state.current_page == 'favorites':
        favorites_page(df, vectorizer)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom navigation
    navigation()

if __name__ == "__main__":
    main()
