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

# Improved CSS - keeping original design but enhanced
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Your Color Scheme - Enhanced */
    :root {{
        --bg-color: #333446;
        --text-color: #7F8CAA;
        --accent-color: #B8CFCE;
        --white: #ffffff;
        --light-bg: #f5f7fa;
    }}
    
    .stApp {{
        background: var(--light-bg);
    }}
    
    /* Enhanced Header Section */
    .main-header {{
        background: var(--white);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--accent-color);
    }}
    
    .logo-container {{
        margin-bottom: 1rem;
        display: flex;
        justify-content: center;
    }}
    
    .logo-img {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 3px solid var(--accent-color);
        padding: 8px;
        background: var(--white);
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.05);
    }}
    
    .main-header h1 {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem 0;
        color: var(--bg-color);
        font-family: 'Amiri', serif;
    }}
    
    .main-header p {{
        font-size: 1.1rem;
        color: var(--text-color);
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    
    /* Enhanced Search Section */
    .search-section {{
        background: linear-gradient(135deg, var(--white), #f8fafb);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 2px solid var(--accent-color);
    }}
    
    .search-title {{
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--bg-color);
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
        text-align: center;
    }}
    
    /* Enhanced Cards */
    .adhkar-card {{
        background: linear-gradient(135deg, var(--white), #fafbfc);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.08);
        direction: rtl;
        text-align: right;
        border: 1px solid var(--accent-color);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    
    .adhkar-card::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent-color), var(--text-color));
    }}
    
    .adhkar-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        border-color: var(--text-color);
    }}
    
    .adhkar-text {{
        font-size: 1.3rem;
        line-height: 2;
        color: var(--bg-color);
        font-family: 'Amiri', serif;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }}
    
    .adhkar-category {{
        background: linear-gradient(135deg, var(--accent-color), #c8dedd);
        color: var(--bg-color);
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 2px 8px rgba(184, 207, 206, 0.3);
        border: 1px solid rgba(184, 207, 206, 0.5);
    }}
    
    /* Enhanced Buttons */
    .stButton > button {{
        background: var(--text-color) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(127, 140, 170, 0.3) !important;
        margin: 0 4px !important;
    }}
    
    .stButton > button:hover {{
        background: var(--bg-color) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(127, 140, 170, 0.4) !important;
    }}
    
    /* Enhanced Sidebar */
    .sidebar-card {{
        background: var(--white);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--accent-color);
        transition: all 0.3s ease;
    }}
    
    .sidebar-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }}
    
    .counter-display {{
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: var(--bg-color);
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Enhanced Stats */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    
    .stat-card {{
        background: var(--white);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--accent-color);
        transition: all 0.3s ease;
    }}
    
    .stat-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }}
    
    .stat-number {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--bg-color);
        font-family: 'Inter', sans-serif;
    }}
    
    .stat-label {{
        font-size: 0.9rem;
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }}
    
    /* Enhanced Input styling */
    .stTextInput > div > div > input {{
        border-radius: 12px !important;
        border: 2px solid var(--accent-color) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        background: var(--white) !important;
        color: var(--bg-color) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--text-color) !important;
        box-shadow: 0 0 0 3px rgba(184, 207, 206, 0.2) !important;
    }}
    
    .stSelectbox > div > div > select {{
        border-radius: 12px !important;
        border: 2px solid var(--accent-color) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.8rem !important;
        background: var(--white) !important;
        color: var(--bg-color) !important;
    }}
    
    /* Enhanced Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: var(--accent-color);
        color: var(--bg-color);
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: var(--bg-color);
        color: var(--white);
    }}
    
    /* Enhanced Success/Info messages */
    .stSuccess {{
        background: var(--accent-color);
        color: var(--bg-color);
        border-radius: 8px;
    }}
    
    .stInfo {{
        background: var(--text-color);
        color: var(--white);
        border-radius: 8px;
    }}
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--light-bg);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--accent-color);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-color);
    }}
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

def display_adhkar_card(adhkar_text, category, index, similarity_score=None):
    """Display a single adhkar card"""
    similarity_badge = ""
    if similarity_score is not None:
        similarity_percentage = int(similarity_score * 100)
        similarity_badge = f" • تطابق {similarity_percentage}%"
    
    with st.container():
        st.markdown(f"""
        <div class="adhkar-card">
            <div class="adhkar-text">{adhkar_text}</div>
            <span class="adhkar-category">{category}{similarity_badge}</span>
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
            if st.button("📋 نسخ", key=f"copy_{index}"):
                st.code(adhkar_text, language="text")
        
        with col4:
            if st.button("🔗 مشاركة", key=f"share_{index}"):
                st.info("تم تحضير الذكر للمشاركة")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data and model
    vectorizer, df = load_model_and_vectorizer()
    
    if df.empty:
        df = load_data()  # Fallback to regular data loading
        if df.empty:
            st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
            return
    
    # Main header with logo
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <img src="https://via.placeholder.com/80x80/B8CFCE/333446?text=🕌" class="logo-img" alt="Islamic Logo">
        </div>
        <h1>أذكار المسلم</h1>
        <p>اذكروا الله كثيراً لعلكم تفلحون</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search section at the top
    st.markdown("""
    <div class="search-section">
        <div class="search-title">🔍 البحث في الأذكار والأدعية</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "", 
            placeholder="ابحث في الأذكار والأدعية... (مثال: الحمد لله، اللهم اغفر لي، الاستغفار)", 
            label_visibility="collapsed"
        )
    
    with col2:
        categories = ['الكل'] + list(df['category'].unique())
        selected_category = st.selectbox("الفئة", categories, label_visibility="collapsed")
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-card">
            <h3 style="text-align: center; color: var(--bg-color);">📊 إحصائياتك</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Daily statistics
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="text-align: center;">
                <div style="color: var(--text-color); font-size: 0.9rem;">أذكار اليوم</div>
                <div class="counter-display">{st.session_state.daily_adhkar_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="text-align: center;">
                <div style="color: var(--text-color); font-size: 0.9rem;">إجمالي الأذكار</div>
                <div class="counter-display">{st.session_state.counter}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset counter
        if st.button("🔄 إعادة تعيين"):
            st.session_state.counter = 0
            st.session_state.daily_adhkar_count = 0
            st.success("تم إعادة تعيين العداد")
        
        # Random adhkar
        st.markdown("""
        <div class="sidebar-card">
            <h4 style="text-align: center; color: var(--bg-color);">🎯 ذكر عشوائي</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("احصل على ذكر"):
            random_adhkar = df.sample(1).iloc[0]
            st.markdown(f"""
            <div class="sidebar-card">
                <div style="direction: rtl; text-align: right; color: var(--bg-color); font-family: 'Amiri', serif; line-height: 1.6;">
                    {random_adhkar['clean_text']}
                </div>
                <div style="margin-top: 8px;">
                    <span class="adhkar-category">{random_adhkar['category']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 البحث", "⭐ المفضلة", "📊 الإحصائيات"])
    
    # Search Tab
    with tab1:
        # Filter data based on search and category
        filtered_df = df.copy()
        
        if search_query:
            # Use semantic search if available, otherwise traditional search
            if SKLEARN_AVAILABLE and vectorizer is not None:
                semantic_results, similarities = semantic_search(search_query, vectorizer, df, top_k=10)
                if not semantic_results.empty:
                    st.success(f"🎯 تم العثور على {len(semantic_results)} نتيجة ذكية")
                    for idx, (_, row) in enumerate(semantic_results.iterrows()):
                        display_adhkar_card(
                            row['clean_text'], 
                            row['category'], 
                            f"semantic_{idx}",
                            similarity_score=similarities[idx]
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
        
        # Display results for traditional search or when no search query
        if not search_query or (not SKLEARN_AVAILABLE or vectorizer is None):
            st.markdown(f"**عدد النتائج: {len(filtered_df)}**")
            
            # Display adhkar cards
            for idx, row in filtered_df.head(10).iterrows():
                display_adhkar_card(row['clean_text'], row['category'], f"trad_{idx}")
    
    # Favorites Tab
    with tab2:
        if st.session_state.favorite_adhkar:
            st.success(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
            
            for i, adhkar in enumerate(st.session_state.favorite_adhkar):
                st.markdown(f"""
                <div class="adhkar-card">
                    <div class="adhkar-text">{adhkar}</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"🗑️ حذف", key=f"del_fav_{i}"):
                        st.session_state.favorite_adhkar.remove(adhkar)
                        st.rerun()
                
                with col2:
                    if st.button(f"📋 نسخ", key=f"copy_fav_{i}"):
                        st.code(adhkar, language="text")
            
            if st.button("🗑️ مسح الكل"):
                st.session_state.favorite_adhkar = []
                st.success("تم مسح جميع الأذكار المفضلة")
                st.rerun()
        else:
            st.info("لا توجد أذكار مفضلة حتى الآن")
    
    # Statistics Tab
    with tab3:
        # Overall statistics
        st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">عدد الفئات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">أذكار اليوم</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">المفضلة</div>
            </div>
        </div>
        """.format(
            len(df),
            len(df['category'].unique()),
            st.session_state.daily_adhkar_count,
            len(st.session_state.favorite_adhkar)
        ), unsafe_allow_html=True)
        
        # Category distribution
        st.markdown("### 📈 توزيع الأذكار حسب الفئات")
        category_counts = df['category'].value_counts()
        st.bar_chart(category_counts.head(10))
        
        # Most common categories
        st.markdown("### 🏆 أكثر الفئات شيوعاً")
        for i, (category, count) in enumerate(category_counts.head(5).items(), 1):
            st.write(f"{i}. {category}: {count} ذكر")

if __name__ == "__main__":
    main()
