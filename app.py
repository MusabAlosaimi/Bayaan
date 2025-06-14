import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from collections import Counter
import re

# Try to import scikit-learn, fallback gracefully if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not installed. AI features will be disabled. Install with: pip install scikit-learn")

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="أذكار المسلم - Islamic Adhkar AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .ai-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .adhkar-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-right: 4px solid #667eea;
        direction: rtl;
        text-align: right;
    }
    
    .similar-adhkar-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-right: 4px solid #e17055;
        direction: rtl;
        text-align: right;
    }
    
    .adhkar-text {
        font-size: 1.3rem;
        line-height: 2;
        color: #2c3e50;
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
    }
    
    .similarity-score {
        background: linear-gradient(45deg, #00b894, #00cec9);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .category-tag {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .search-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .ai-search-container {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .time-based-greeting {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .counter-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .counter-number {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
    }
    
    .random-adhkar {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: #333;
    }
    
    .ml-insights {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        min-width: 150px;
    }
    
    .installation-guide {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the adhkar data"""
    try:
        df = pd.read_csv('adhkar_df.csv')
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_tfidf_model():
    """Load and cache the TF-IDF vectorizer model"""
    if not SKLEARN_AVAILABLE:
        return None
    
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.warning("⚠️ ملف النموذج غير موجود: tfidf_vectorizer.pkl")
        return None
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None

@st.cache_data
def get_tfidf_matrix(_vectorizer, texts):
    """Get TF-IDF matrix for texts"""
    if not SKLEARN_AVAILABLE or _vectorizer is None:
        return None
    
    try:
        return _vectorizer.transform(texts)
    except Exception as e:
        st.error(f"خطأ في معالجة النصوص: {e}")
        return None

def semantic_search(query, vectorizer, tfidf_matrix, df, top_k=5):
    """Perform semantic search using TF-IDF similarity"""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame(), []
    
    try:
        if vectorizer is None or tfidf_matrix is None:
            return pd.DataFrame(), []
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
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

def find_similar_adhkar(adhkar_text, vectorizer, tfidf_matrix, df, top_k=3):
    """Find similar adhkar to a given adhkar"""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame(), []
    
    try:
        if vectorizer is None or tfidf_matrix is None:
            return pd.DataFrame(), []
            
        # Find the index of current adhkar
        current_idx = df[df['clean_text'] == adhkar_text].index
        if len(current_idx) == 0:
            return pd.DataFrame(), []
        
        current_idx = current_idx[0]
        
        # Get similarity with all other adhkar
        current_vector = tfidf_matrix[current_idx]
        similarities = cosine_similarity(current_vector, tfidf_matrix).flatten()
        
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

def get_category_insights(df, vectorizer, tfidf_matrix):
    """Get ML insights about categories"""
    if not SKLEARN_AVAILABLE:
        return {}
    
    try:
        if vectorizer is None or tfidf_matrix is None:
            return {}
        
        insights = {}
        categories = df['category'].unique()
        
        # Calculate average TF-IDF scores for each category
        for category in categories:
            category_mask = df['category'] == category
            category_tfidf = tfidf_matrix[category_mask]
            if category_tfidf.shape[0] > 0:
                avg_tfidf = np.mean(category_tfidf.toarray(), axis=0)
                # Get top features for this category
                feature_names = vectorizer.get_feature_names_out()
                top_features_idx = avg_tfidf.argsort()[-5:][::-1]
                top_features = [feature_names[idx] for idx in top_features_idx]
                top_scores = [avg_tfidf[idx] for idx in top_features_idx]
                
                insights[category] = {
                    'top_features': top_features,
                    'scores': top_scores,
                    'count': sum(category_mask)
                }
        
        return insights
    except Exception as e:
        st.error(f"خطأ في تحليل الفئات: {e}")
        return {}

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
pip install scikit-learn

# أو تثبيت جميع المتطلبات
pip install streamlit pandas numpy scikit-learn
    """, language="bash")
    
    st.markdown("""
    **الميزات المتاحة بعد التثبيت:**
    - 🤖 البحث الذكي بالمعنى
    - 🔍 العثور على أذكار مشابهة
    - 📊 تحليلات ذكية للفئات
    - 🎯 توصيات مخصصة
    """)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data and model
    df = load_data()
    vectorizer = load_tfidf_model() if SKLEARN_AVAILABLE else None
    
    if df.empty:
        st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
        return
    
    # Get TF-IDF matrix
    tfidf_matrix = None
    if vectorizer is not None and SKLEARN_AVAILABLE:
        with st.spinner("جاري تحضير النموذج الذكي..."):
            tfidf_matrix = get_tfidf_matrix(vectorizer, df['clean_text'].tolist())
    
    # Main header
    ai_status = "🤖 مفعل" if (SKLEARN_AVAILABLE and vectorizer is not None) else "❌ غير متاح"
    st.markdown(f"""
    <div class="main-header">
        <h1>🕌 أذكار المسلم الذكي - Islamic Adhkar AI</h1>
        <p>اذكروا الله كثيراً لعلكم تفلحون</p>
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
        
        # AI-powered or random adhkar
        st.markdown("""
        <div class="sidebar-content">
            <h3>🎯 ذكر مقترح</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 احصل على ذكر"):
            if SKLEARN_AVAILABLE and vectorizer is not None and tfidf_matrix is not None:
                # Get time-based recommendation
                current_hour = datetime.now().hour
                if 5 <= current_hour < 12:
                    query = "صباح"
                elif 18 <= current_hour < 22:
                    query = "مساء"
                else:
                    query = "نوم"
                
                smart_results, similarities = semantic_search(query, vectorizer, tfidf_matrix, df, top_k=1)
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 البحث الذكي", 
            "🔍 البحث التقليدي", 
            "⭐ المفضلة", 
            "📊 تحليلات ذكية", 
            "🎯 توصيات", 
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
            
            # Semantic search
            semantic_query = st.text_input(
                "🧠 البحث الدلالي", 
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
                        semantic_query, vectorizer, tfidf_matrix, df, top_k=search_depth
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
                            quick_search, vectorizer, tfidf_matrix, df, top_k=3
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
        recommendations_tab = tab5
        about_tab = tab6
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
            if SKLEARN_AVAILABLE and vectorizer is not None and tfidf_matrix is not None:
                if st.button("🤖 اقتراحات ذكية بناءً على المفضلة"):
                    all_suggestions = []
                    for fav_adhkar in st.session_state.favorite_adhkar[:3]:  # Limit to avoid too many results
                        similar_results, similarities = find_similar_adhkar(
                            fav_adhkar, vectorizer, tfidf_matrix, df, top_k=2
                        )
                        if not similar_results.empty:
                            for idx, (_, row) in enumerate(similar_results.iterrows()):
                                if row['clean_text'] not in st.session_state.favorite_adhkar:
                                    all_suggestions.append((row, similarities[idx]))
                    
                    if all_suggestions:
                        st.markdown("### 🤖 اقتراحات ذكية بناءً على مفضلتك:")
                        for idx, (row, sim) in enumerate(all_suggestions[:5]):  # Show top 5
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
                    if SKLEARN_AVAILABLE and vectorizer is not None and tfidf_matrix is not None:
                        if st.button(f"🔍 أذكار مشابهة", key=f"similar_fav_{i}"):
                            similar_results, similarities = find_similar_adhkar(
                                adhkar, vectorizer, tfidf_matrix, df, top_k=3
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
        
        # AI Insights (only if sklearn available)
        if SKLEARN_AVAILABLE and vectorizer is not None and tfidf_matrix is not None:
            st.markdown("### 🤖 تحليلات ذكية للفئات")
            
            with st.spinner("جاري تحليل الفئات..."):
                insights = get_category_insights(df, vectorizer, tfidf_matrix)
            
            if insights:
                selected_category_for_analysis = st.selectbox(
                    "اختر فئة للتحليل الذكي:", 
                    list(insights.keys())
                )
                
                if selected_category_for_analysis in insights:
                    insight = insights[selected_category_for_analysis]
                    st.markdown(f"""
                    <div class="ml-insights">
                        <h4>📊 تحليل فئة: {selected_category_for_analysis}</h4>
                        <p><strong>عدد الأذكار:</strong> {insight['count']}</p>
                        <p><strong>الكلمات الأساسية:</strong></p>
                        <ul>
                    """, unsafe_allow_html=True)
                    
                    for feature, score in zip(insight['top_features'], insight['scores']):
                        st.markdown(f"<li>{feature} (أهمية: {score:.3f})</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        
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
        st.bar_chart(pd.DataFrame({
            'count': hist_data[0]
        }))
    
    # Recommendations Tab (only if sklearn available)
    if SKLEARN_AVAILABLE and vectorizer is not None:
        with recommendations_tab:
            st.markdown("## 🎯 توصيات ذكية")
            
            # Time-based recommendations
            st.markdown("### ⏰ توصيات حسب الوقت")
            current_hour = datetime.now().hour
            
            if 5 <= current_hour < 12:
                recommended_query = "صباح"
                st.info("🌅 الوقت الآن مناسب لأذكار الصباح")
            elif 18 <= current_hour < 22:
                recommended_query = "مساء"
                st.info("🌆 الوقت الآن مناسب لأذكار المساء")
            elif 22 <= current_hour or current_hour < 5:
                recommended_query = "نوم"
                st.info("🌙 الوقت الآن مناسب لأذكار النوم")
            else:
                recommended_query = "دعاء"
                st.info("📿 يمكنك قراءة أي أذكار في هذا الوقت")
            
            time_recommendations, time_similarities = semantic_search(
                recommended_query, vectorizer, tfidf_matrix, df, top_k=3
            )
            
            if not time_recommendations.empty:
                st.markdown("#### 🤖 الأذكار المقترحة لهذا الوقت:")
                for idx, (_, row) in enumerate(time_recommendations.iterrows()):
                    display_adhkar_card(
                        row['clean_text'], 
                        row['category'], 
                        f"time_rec_{idx}",
                        similarity_score=time_similarities[idx],
                        is_similar=True
                    )
            
            # Mood-based recommendations
            st.markdown("### 🎭 توصيات حسب المزاج")
            mood_options = {
                "😊 سعيد وممتن": "حمد شكر",
                "😔 حزين ومهموم": "حزن هم",
                "😰 قلق وخائف": "خوف أمان",
                "🤲 راغب في الدعاء": "دعاء استغفار",
                "🙏 طالب المغفرة": "توبة استغفار",
                "❤️ محب لله": "حب الله"
            }
            
            selected_mood = st.selectbox("اختر حالتك الحالية:", list(mood_options.keys()))
            
            if st.button("🎯 احصل على توصيات للمزاج"):
                mood_query = mood_options[selected_mood]
                mood_recommendations, mood_similarities = semantic_search(
                    mood_query, vectorizer, tfidf_matrix, df, top_k=4
                )
                
                if not mood_recommendations.empty:
                    st.markdown(f"#### توصيات لحالة: {selected_mood}")
                    for idx, (_, row) in enumerate(mood_recommendations.iterrows()):
                        display_adhkar_card(
                            row['clean_text'], 
                            row['category'], 
                            f"mood_rec_{idx}",
                            similarity_score=mood_similarities[idx],
                            is_similar=True
                        )
            
            # Personalized recommendations based on favorites
            if st.session_state.favorite_adhkar:
                st.markdown("### 💝 توصيات شخصية بناءً على مفضلتك")
                
                if st.button("🤖 احصل على توصيات شخصية"):
                    personal_recommendations = []
                    
                    # Analyze favorite adhkar to get personalized recommendations
                    for fav_adhkar in st.session_state.favorite_adhkar[:2]:
                        similar_results, similarities = find_similar_adhkar(
                            fav_adhkar, vectorizer, tfidf_matrix, df, top_k=2
                        )
                        
                        for idx, (_, row) in enumerate(similar_results.iterrows()):
                            if row['clean_text'] not in st.session_state.favorite_adhkar:
                                personal_recommendations.append((row, similarities[idx]))
                    
                    if personal_recommendations:
                        # Sort by similarity and show top recommendations
                        personal_recommendations.sort(key=lambda x: x[1], reverse=True)
                        
                        st.markdown("#### 🎯 توصيات مخصصة لك:")
                        for idx, (row, sim) in enumerate(personal_recommendations[:4]):
                            display_adhkar_card(
                                row['clean_text'], 
                                row['category'], 
                                f"personal_rec_{idx}",
                                similarity_score=sim,
                                is_similar=True
                            )
                    else:
                        st.info("لا توجد توصيات شخصية متاحة حالياً")
    
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
        - 📱 تصميم متجاوب
        """)
        
        if SKLEARN_AVAILABLE and vectorizer is not None:
            vocab_size = len(vectorizer.get_feature_names_out())
            st.markdown(f"""
            #### 🤖 المميزات الذكية (مفعلة):
            - 🧠 بحث ذكي بالمعنى باستخدام TF-IDF
            - 🔍 العثور على أذكار مشابهة
            - 📊 تحليلات ذكية للفئات
            - 🎯 توصيات مخصصة
            - 📈 تحليل النصوص بـ {vocab_size:,} كلمة
            """)
        elif SKLEARN_AVAILABLE:
            st.markdown("""
            #### ⚠️ المميزات الذكية (غير مفعلة):
            - النموذج غير محمل - تأكد من وجود ملف `tfidf_vectorizer.pkl`
            """)
        else:
            st.markdown("""
            #### ❌ المميزات الذكية (غير متاحة):
            - يتطلب تثبيت scikit-learn
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
        scikit-learn>=1.3.0  # للميزات الذكية
        ```
        
        #### 📁 الملفات المطلوبة:
        - `adhkar_df.csv` - بيانات الأذكار
        - `tfidf_vectorizer.pkl` - النموذج الذكي (اختياري)
        
        ---
        ### 🤲 دعاء
        
        *"اللهم اجعل هذا العمل خالصاً لوجهك الكريم، وانفع به المسلمين في كل مكان"*
        
        **تذكر:** المداومة على الأذكار خير من الانقطاع عنها
        
        ---
        
        #### 📞 الدعم التقني:
        - إذا واجهت مشاكل في التثبيت، تأكد من إصدارات المكتبات
        - للمساعدة في النموذج الذكي، تأكد من وجود ملف النموذج
        - النطبيق يعمل بدون الميزات الذكية إذا لم تكن متاحة
        """)

if __name__ == "__main__":
    main()
