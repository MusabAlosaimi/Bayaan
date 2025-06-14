import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pickle
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="أذكار المسلم - Islamic Adhkar",
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
    
    .adhkar-text {
        font-size: 1.3rem;
        line-height: 2;
        color: #2c3e50;
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
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
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
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
    
    .search-container {
        background: #f8f9fa;
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

def display_adhkar_card(adhkar_text, category, index):
    """Display a single adhkar card"""
    with st.container():
        st.markdown(f"""
        <div class="adhkar-card">
            <div class="adhkar-text">{adhkar_text}</div>
            <div class="category-tag">{category}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📖 قراءة", key=f"read_{index}"):
                st.session_state.counter += 1
                st.session_state.daily_adhkar_count += 1
                st.success("✅ تم احتساب القراءة")
        
        with col2:
            if st.button("❤️ إضافة للمفضلة", key=f"fav_{index}"):
                if adhkar_text not in st.session_state.favorite_adhkar:
                    st.session_state.favorite_adhkar.append(adhkar_text)
                    st.success("✅ تم إضافة الذكر للمفضلة")
                else:
                    st.info("هذا الذكر موجود بالفعل في المفضلة")
        
        with col3:
            if st.button("📋 نسخ", key=f"copy_{index}"):
                st.code(adhkar_text)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("لا يمكن تحميل البيانات. يرجى التأكد من وجود ملف البيانات.")
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🕌 أذكار المسلм - Islamic Adhkar</h1>
        <p>اذكروا الله كثيراً لعلكم تفلحون</p>
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
            <h3>🎲 ذكر عشوائي</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 احصل على ذكر عشوائي"):
            random_adhkar = df.sample(1).iloc[0]
            st.markdown(f"""
            <div class="random-adhkar">
                <div class="adhkar-text">{random_adhkar['clean_text']}</div>
                <div class="category-tag">{random_adhkar['category']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 البحث والتصفح", "⭐ المفضلة", "📊 الإحصائيات", "🎯 أذكار مقترحة", "ℹ️ حول التطبيق"])
    
    with tab1:
        st.markdown("""
        <div class="search-container">
            <h3>🔍 البحث في الأذكار</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Search and filter options
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
            display_adhkar_card(row['clean_text'], row['category'], idx)
    
    with tab2:
        st.markdown("## ⭐ الأذكار المفضلة")
        
        if st.session_state.favorite_adhkar:
            st.success(f"لديك {len(st.session_state.favorite_adhkar)} ذكر في المفضلة")
            
            for i, adhkar in enumerate(st.session_state.favorite_adhkar):
                st.markdown(f"""
                <div class="adhkar-card">
                    <div class="adhkar-text">{adhkar}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🗑️ حذف من المفضلة", key=f"del_fav_{i}"):
                    st.session_state.favorite_adhkar.remove(adhkar)
                    st.rerun()
            
            if st.button("🗑️ مسح جميع المفضلة"):
                st.session_state.favorite_adhkar = []
                st.success("تم مسح جميع الأذكار المفضلة")
                st.rerun()
        else:
            st.info("لا توجد أذكار مفضلة حتى الآن. أضف بعض الأذكار من قسم البحث!")
    
    with tab3:
        st.markdown("## 📊 إحصائيات مفصلة")
        
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
    
    with tab4:
        st.markdown("## 🎯 أذكار مقترحة حسب الوقت")
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            recommended_category = "أذكار الصباح والمساء"
            st.info("🌅 الوقت الآن مناسب لأذكار الصباح")
        elif 18 <= current_hour < 22:
            recommended_category = "أذكار الصباح والمساء"
            st.info("🌆 الوقت الآن مناسب لأذكار المساء")
        elif 22 <= current_hour or current_hour < 5:
            recommended_category = "أذكار النوم"
            st.info("🌙 الوقت الآن مناسب لأذكار النوم")
        else:
            recommended_category = None
            st.info("📿 يمكنك قراءة أي أذكار في هذا الوقت")
        
        if recommended_category:
            recommended_adhkar = df[df['category'] == recommended_category]
            if not recommended_adhkar.empty:
                st.markdown(f"### الأذكار المقترحة: {recommended_category}")
                for idx, row in recommended_adhkar.head(3).iterrows():
                    display_adhkar_card(row['clean_text'], row['category'], f"rec_{idx}")
        
        # Random suggestions
        st.markdown("### 🎲 اقتراحات عشوائية")
        if st.button("🔄 اقتراحات جديدة"):
            random_suggestions = df.sample(3)
            for idx, row in random_suggestions.iterrows():
                display_adhkar_card(row['clean_text'], row['category'], f"rand_{idx}")
    
    with tab5:
        st.markdown("## ℹ️ حول التطبيق")
        
        st.markdown("""
        ### 🕌 تطبيق أذكار المسلم
        
        هذا التطبيق يحتوي على مجموعة شاملة من الأذكار والأدعية الإسلامية المأخوذة من القرآن الكريم والسنة النبوية الشريفة.
        
        #### 🌟 المميزات:
        - 📖 أكثر من 260 ذكر ودعاء
        - 🔍 بحث متقدم في الأذكار
        - ⭐ إمكانية حفظ الأذكار المفضلة
        - 📊 تتبع عدد الأذكار المقروءة
        - 🎯 اقتراحات حسب الوقت
        - 🎲 أذكار عشوائية
        - 📱 تصميم متجاوب
        
        #### 📚 الفئات المتاحة:
        """)
        
        categories_list = df['category'].unique()
        for i, category in enumerate(categories_list, 1):
            count = len(df[df['category'] == category])
            st.write(f"{i}. **{category}** ({count} ذكر)")
        
        st.markdown("""
        ---
        ### 🤲 دعاء
        
        *"اللهم اجعل هذا العمل خالصاً لوجهك الكريم، وانفع به المسلمين في كل مكان"*
        
        **تذكر:** المداومة على الأذكار خير من الانقطاع عنها
        """)

if __name__ == "__main__":
    main()
