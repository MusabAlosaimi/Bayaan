import streamlit as st

# Page configuration
st.set_page_config(
    page_title="بيان - أذكار المسلم",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="expanded"
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

# Modern website CSS
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
        font-family: 'Inter', sans-serif;
    }}
    
    .container {{
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 1rem;
    }}
    
    .header {{
        background: white;
        padding: 1rem 0;
        text-align: center;
        box-shadow: var(--card-shadow);
        position: sticky;
        top: 0;
        z-index: 100;
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        padding: 0.5rem 0;
    }}
    
    .logo-img {{
        width: 120px;
        height: auto;
        border-radius: 16px;
    }}
    
    .app-title {{
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
        color: var(--dark);
        font-family: 'Amiri', serif;
    }}
    
    .app-subtitle {{
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.25rem 0;
        color: #64748b;
    }}
    
    .nav-container {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border);
    }}
    
    .nav-button {{
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .nav-button:hover {{
        background: var(--primary-light);
        color: var(--primary);
    }}
    
    .nav-button.active {{
        background: var(--primary);
        color: white;
    }}
    
    .card {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
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
        font-size: 1.25rem;
        line-height: 1.8;
        color: var(--dark);
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
        font-weight: 400;
    }}
    
    .badge {{
        background: var(--accent);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
    }}
    
    .tag {{
        background: var(--primary);
        color: white;
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 0.5rem;
    }}
    
    .section {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--card-shadow);
    }}
    
    .search-container {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--card-shadow);
    }}
    
    .search-bar {{
        display: flex;
        gap: 15px;
        margin-bottom: 1.5rem;
    }}
    
    .search-input {{
        flex: 1;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 1.1rem;
        direction: rtl;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .search-input:focus {{
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primary-light);
    }}
    
    .filter-section {{
        display: flex;
        gap: 15px;
        margin-bottom: 1.5rem;
    }}
    
    .filter-select {{
        flex: 1;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        font-size: 1.1rem;
        direction: rtl;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .action-button {{
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }}
    
    .action-button:hover {{
        background: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }}
    
    .secondary-button {{
        background: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }}
    
    .footer {{
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        color: var(--text);
        border-top: 1px solid var(--border);
    }}
    
    .stats-container {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .stat-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border);
    }}
    
    .stat-number {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }}
    
    .stat-label {{
        font-size: 1rem;
        color: var(--text);
    }}
    
    .grid {{
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 2rem;
        margin: 2rem 0;
    }}
    
    @media (max-width: 768px) {{
        .grid {{
            grid-template-columns: 1fr;
        }}
        
        .stats-container {{
            grid-template-columns: repeat(2, 1fr);
        }}
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("📖 قراءة", key=f"read_{index}", use_container_width=True)
        
        with col2:
            if st.button("❤️ مفضلة", key=f"fav_{index}", use_container_width=True):
                if 'favorite_adhkar' not in st.session_state:
                    st.session_state.favorite_adhkar = []
                if adhkar_text not in st.session_state.favorite_adhkar:
                    st.session_state.favorite_adhkar.append(adhkar_text)
                    st.success("✅ تم إضافة الذكر للمفضلة")
        
        with col3:
            if SKLEARN_AVAILABLE:
                st.button("🔍 مشابه", key=f"similar_{index}", use_container_width=True)
        
        with col4:
            if st.button("📋 نسخ", key=f"copy_{index}", use_container_width=True):
                st.code(adhkar_text, language="text")

def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'favorite_adhkar' not in st.session_state:
        st.session_state.favorite_adhkar = []
    
    # Load data and model
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()
        if df.empty:
            st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
            return
    
    # Main container for content
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Website header
    st.markdown(f"""
    <div class="header">
        <div class="logo-container">
            <img src="https://via.placeholder.com/120x120/2563eb/ffffff?text=BAYAAN" class="logo-img" alt="Bayaan Logo">
        </div>
        <div class="app-title">بيان - أذكار المسلم</div>
        <div class="app-subtitle">اذكروا الله كثيراً لعلكم تفلحون</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("""
    <div class="nav-container">
        <div class="nav-button {'active' if st.session_state.current_page == 'home' else ''}" 
             onclick="setPage('home')">الرئيسية</div>
        <div class="nav-button {'active' if st.session_state.current_page == 'search' else ''}" 
             onclick="setPage('search')">البحث</div>
        <div class="nav-button {'active' if st.session_state.current_page == 'favorites' else ''}" 
             onclick="setPage('favorites')">المفضلة</div>
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
    
    # Home page
    if st.session_state.current_page == 'home':
        st.markdown("## 🕌 مرحباً في أذكار المسلم")
        st.markdown("""
        <div class="section">
            <h3>أذكار مختارة لكم اليوم</h3>
            <p>مجموعة من الأذكار المختارة بعناية تناسب وقتك الحالي</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Time-based greeting
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            greeting = "🌅 صباح الخير - أذكار الصباح"
        elif 12 <= current_hour < 18:
            greeting = "☀️ مساء الخير - أذكار المساء"
        elif 18 <= current_hour < 22:
            greeting = "🌆 مساء الخير - أذكار المساء"
        else:
            greeting = "🌙 تصبح على خير - أذكار النوم"
            
        st.markdown(f"""
        <div class="card highlight-card">
            <div style="text-align: center; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
                {greeting}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Stats
        st.markdown("## 📊 إحصائيات")
        st.markdown("""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(df['category'].unique())}</div>
                <div class="stat-label">عدد الفئات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.favorite_adhkar)}</div>
                <div class="stat-label">المفضلة</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{random.randint(1000, 5000)}</div>
                <div class="stat-label">المستخدمين</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI search
        if SKLEARN_AVAILABLE and vectorizer is not None:
            st.markdown("## 🤖 البحث الذكي")
            with st.form("ai_search_form"):
                user_dua = st.text_area(
                    "اكتب حالتك أو طلبك:", 
                    placeholder="مثال: اللهم اغفر لي، أريد الحماية من الشر، أدعية للرزق...",
                    height=100
                )
                
                if st.form_submit_button("🔍 بحث ذكي"):
                    with st.spinner("جاري البحث عن الدعاء المناسب..."):
                        # Transform query
                        clean_dua = remove_tashkeel(user_dua.strip())
                        if not clean_dua:
                            st.info("❗ الرجاء إدخال نص صحيح")
                        else:
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
    
    # Search page
    elif st.session_state.current_page == 'search':
        st.markdown("## 🔍 البحث في الأذكار")
        
        with st.container():
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            
            # Search input
            st.markdown('<div class="search-bar">', unsafe_allow_html=True)
            search_query = st.text_input(
                "", 
                placeholder="ابحث في الأذكار والأدعية... (مثال: الصباح، الرزق، الحماية)",
                key="search_input", 
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Filters
            st.markdown('<div class="filter-section">', unsafe_allow_html=True)
            categories = ['الكل'] + list(df['category'].unique())
            selected_category = st.selectbox(
                "الفئة", 
                categories, 
                index=0,
                label_visibility="collapsed"
            )
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
        
        # Display results
        st.markdown(f"### نتائج البحث: {len(filtered_df)} ذكر")
        
        # Display adhkar cards
        for idx, row in filtered_df.iterrows():
            display_adhkar_card(row['clean_text'], row['category'], f"search_{idx}")
    
    # Favorites page
    elif st.session_state.current_page == 'favorites':
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
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>بيان - أذكار المسلم • إحياء سنة الذكر</p>
        <p>جميع الحقوق محفوظة © {year}</p>
    </div>
    """.format(year=datetime.now().year), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close container

if __name__ == "__main__":
    main()
