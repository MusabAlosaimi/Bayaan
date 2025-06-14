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
    
    /* Modern Color Scheme */
    :root {{
        --primary: #2563eb;
        --primary-light: #dbeafe;
        --secondary: #8b5cf6;
        --accent: #10b981;
        --light: #f8fafc;
        --dark: #1e293b;
        --text: #334155;
        --border: #e2e8f0;
        --card-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}
    
    .stApp {{
        background: #f8fafc;
        max-width: 500px;
        margin: 0 auto;
        padding: 0;
        font-family: 'Inter', sans-serif;
    }}
    
    .header {{
        background: white;
        padding: 1rem;
        text-align: center;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: var(--card-shadow);
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
    }}
    
    .card {{
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        direction: rtl;
        text-align: right;
        box-shadow: var(--card-shadow);
        transition: all 0.2s ease;
    }}
    
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
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
    }}
    
    .section {{
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--card-shadow);
    }}
    
    .search-container {{
        background: white;
        padding: 1rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--card-shadow);
    }}
    
    .search-bar {{
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
    }}
    
    .search-input {{
        flex: 1;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 1rem;
        direction: rtl;
    }}
    
    .search-input:focus {{
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px var(--primary-light);
    }}
    
    .search-button {{
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0 20px;
        font-weight: 500;
        cursor: pointer;
    }}
    
    .filter-section {{
        display: flex;
        gap: 10px;
    }}
    
    .filter-select {{
        flex: 1;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        font-size: 1rem;
        direction: rtl;
        background: white;
    }}
    
    /* Button Styles */
    .action-button {{
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
    }}
    
    .action-button:hover {{
        background: #1d4ed8;
        transform: translateY(-1px);
    }}
    
    .secondary-button {{
        background: white;
        color: var(--primary);
        border: 1px solid var(--primary);
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
        box-shadow: 0 -4px 10px rgba(0,0,0,0.05);
        z-index: 100;
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
    }}
    
    .nav-button {{
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 0.8rem;
        color: #64748b;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 10px;
        transition: all 0.2s ease;
    }}
    
    .nav-button.active {{
        background: var(--primary-light);
        color: var(--primary);
    }}
    
    .nav-icon {{
        font-size: 1.2rem;
        margin-bottom: 3px;
    }}
    
    /* Content Area */
    .content-area {{
        padding: 1rem;
        padding-bottom: 70px;
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
    
    .flex {{
        display: flex;
        gap: 10px;
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
            st.button("📖 قراءة", key=f"read_{index}", use_container_width=True)
        
        with col2:
            if st.button("❤️ مفضلة", key=f"fav_{index}", use_container_width=True):
                if 'favorite_adhkar' not in st.session_state:
                    st.session_state.favorite_adhkar = []
                if adhkar_text not in st.session_state.favorite_adhkar:
                    st.session_state.favorite_adhkar.append(adhkar_text)
                    st.success("✅ تم إضافة الذكر للمفضلة")
        
        col3, col4 = st.columns(2)
        with col3:
            if SKLEARN_AVAILABLE:
                st.button("🔍 مشابه", key=f"similar_{index}", use_container_width=True)
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{index}", use_container_width=True):
                st.code(adhkar_text, language="text")

def navigation():
    """Bottom navigation bar"""
    st.markdown("""
    <div class="nav-container">
        <div class="nav-button active" onclick="setPage('home')">
            <div class="nav-icon">🏠</div>
            <div>الرئيسية</div>
        </div>
        <div class="nav-button" onclick="setPage('search')">
            <div class="nav-icon">🔍</div>
            <div>بحث</div>
        </div>
        <div class="nav-button" onclick="setPage('favorites')">
            <div class="nav-icon">⭐</div>
            <div>المفضلة</div>
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
    st.markdown("### أذكار اليوم المختارة")
    
    # Display 3 random adhkar
    if 'random_adhkar' not in st.session_state:
        st.session_state.random_adhkar = df.sample(3)
    
    for idx, row in st.session_state.random_adhkar.iterrows():
        display_adhkar_card(
            row['clean_text'], 
            row['category'], 
            f"home_{idx}"
        )
    
    # Refresh button
    if st.button("🔄 تحديث الأذكار", use_container_width=True):
        st.session_state.random_adhkar = df.sample(3)
        st.rerun()
    
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
                # Transform query
                clean_dua = remove_tashkeel(user_dua.strip())
                if not clean_dua:
                    st.info("❗ الرجاء إدخال نص صحيح")
                    return
                
                user_vector = vectorizer.transform([clean_dua])
                tfidf_matrix = vectorizer.transform(df['clean_text'])
                similarities = manual_cosine_similarity(user_vector, tfidf_matrix)
                best_idx = similarities.argmax()
                best_score = similarities[best_idx]
                
                if best_score < 0.1:
                    st.info("❌ لم يتم العثور على دعاء مشابه")
                else:
                    row = df.iloc[best_idx]
                    st.success(f"✨ تم العثور على دعاء مناسب في فئة: **{row['category']}**")
                    display_adhkar_card(
                        row['clean_text'], 
                        row['category'], 
                        "ai_result",
                        similarity_score=best_score,
                        is_similar=True
                    )

def search_page(df, vectorizer):
    """Search page content"""
    st.markdown("### 🔍 بحث في الأذكار")
    
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        # Search input with button
        st.markdown('<div class="search-bar">', unsafe_allow_html=True)
        search_query = st.text_input("", placeholder="ابحث عن أذكار أو أدعية...", key="search_input", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Filters
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        categories = ['الكل'] + list(df['category'].unique())
        selected_category = st.selectbox("الفئة", categories, index=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df.copy()
    
    if search_query:
        # Use semantic search if available
        if SKLEARN_AVAILABLE and vectorizer is not None:
            with st.spinner("جاري البحث الذكي..."):
                semantic_results, similarities = semantic_search(search_query, vectorizer, df, top_k=10)
                if not semantic_results.empty:
                    st.success(f"🎯 تم العثور على {len(semantic_results)} نتيجة ذكية")
                    for idx, (_, row) in enumerate(semantic_results.iterrows()):
                        display_adhkar_card(
                            row['clean_text'], 
                            row['category'], 
                            f"semantic_{idx}",
                            similarity_score=similarities[idx],
                            is_similar=True
                        )
                else:
                    st.info("لم يتم العثور على نتائج مطابقة")
        else:
            # Traditional search
            filtered_df = filtered_df[
                filtered_df['clean_text'].str.contains(search_query, na=False) |
                filtered_df['category'].str.contains(search_query, na=False)
            ]
    
    if selected_category != 'الكل':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Display results if not using semantic search
    if not search_query or (search_query and (not SKLEARN_AVAILABLE or vectorizer is None)):
        st.markdown(f"**عدد النتائج: {len(filtered_df)}**")
        
        # Display adhkar cards
        for idx, row in filtered_df.head(10).iterrows():
            display_adhkar_card(row['clean_text'], row['category'], f"search_{idx}")

def favorites_page(df, vectorizer):
    """Favorites page content"""
    st.markdown("## ⭐ الأذكار المفضلة")
    
    if 'favorite_adhkar' in st.session_state and st.session_state.favorite_adhkar:
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
                if st.button(f"🗑️ حذف", key=f"del_fav_{i}", use_container_width=True):
                    st.session_state.favorite_adhkar.remove(adhkar)
                    st.rerun()
            
            with col2:
                if st.button(f"📋 نسخ", key=f"copy_fav_{i}", use_container_width=True):
                    st.code(adhkar, language="text")
        
        if st.button("🗑️ مسح جميع المفضلة", use_container_width=True):
            st.session_state.favorite_adhkar = []
            st.success("تم مسح جميع الأذكار المفضلة")
            st.rerun()
    else:
        st.info("لا توجد أذكار مفضلة حتى الآن. أضف بعض الأذكار من قسم البحث!")

def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
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
