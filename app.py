import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="تفسير القرآن الذكي - Smart Quran Tafseer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter

# Try to import optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ مكتبة scikit-learn غير مثبتة. للتثبيت: pip install scikit-learn")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    st.error("❌ مكتبة datasets غير مثبتة. للتثبيت: pip install datasets")

import warnings
warnings.filterwarnings('ignore')

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-600: #16a34a;
        --primary-700: #15803d;
        --secondary-600: #2563eb;
        --secondary-700: #1d4ed8;
        --accent-500: #f59e0b;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 50%, #fffbeb 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    .premium-header {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.95) 0%, rgba(37, 99, 235, 0.9) 50%, rgba(217, 119, 6, 0.95) 100%);
        color: white;
        padding: 3rem 0;
        margin: -1rem -1rem 3rem -1rem;
        text-align: center;
        border-radius: 0 0 20px 20px;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Amiri', serif;
    }
    
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .premium-arabic {
        font-family: 'Amiri', serif;
        font-size: 1.8rem;
        line-height: 2;
        color: var(--gray-800);
        text-align: right;
        direction: rtl;
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.05) 0%, rgba(255, 255, 255, 0.8) 100%);
        padding: 2rem;
        border-radius: 15px;
        border-right: 6px solid var(--primary-600);
        margin: 1rem 0;
    }
    
    .tafseer-text {
        font-family: 'Amiri', serif;
        font-size: 1.2rem;
        line-height: 1.8;
        color: var(--gray-700);
        text-align: right;
        direction: rtl;
        background: var(--gray-50);
        padding: 1.5rem;
        border-radius: 10px;
        border-right: 4px solid var(--accent-500);
        margin: 1rem 0;
    }
    
    .ayah-number {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        color: white;
        padding: 6px 12px;
        border-radius: 50%;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0 8px;
        min-width: 30px;
        text-align: center;
    }
    
    .surah-badge {
        background: linear-gradient(135deg, var(--secondary-600), var(--secondary-700));
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 8px 4px;
    }
    
    .model-status {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .model-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .model-active {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        color: white;
    }
    
    .model-inactive {
        background: var(--gray-100);
        color: var(--gray-600);
    }
</style>
""", unsafe_allow_html=True)

def remove_tashkeel(text):
    """Remove Arabic diacritics"""
    if not isinstance(text, str):
        return ""
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

def preprocess_arabic_text(text):
    """Arabic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    text = remove_tashkeel(text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ي', 'ى', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_sample_data():
    """Create sample data"""
    sample_data = {
        'text': [
            'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الرَّحْمَٰنِ الرَّحِيمِ',
            'مَالِكِ يَوْمِ الدِّينِ',
            'إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ',
            'اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ',
            'صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ',
            'الم',
            'ذَٰلِكَ الْكِتَابُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى لِّلْمُتَّقِينَ',
            'الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ وَيُقِيمُونَ الصَّلَاةَ وَمِمَّا رَزَقْنَاهُمْ يُنفِقُونَ'
        ],
        'surah': ['الفاتحة'] * 7 + ['البقرة'] * 3,
        'ayah_number': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
        'tafseer': [
            'بسم الله الرحمن الرحيم: افتتاح كل سورة بحمد الله والثناء عليه',
            'الحمد لله رب العالمين: الثناء على الله بصفاته الجميلة',
            'الرحمن الرحيم: من أسماء الله الحسنى',
            'مالك يوم الدين: الله هو المالك المتصرف يوم القيامة',
            'إياك نعبد وإياك نستعين: التوحيد في العبادة والاستعانة',
            'اهدنا الصراط المستقيم: دعاء بالهداية إلى الطريق المستقيم',
            'صراط الذين أنعمت عليهم: طريق الأنبياء والصالحين',
            'الم: من الحروف المقطعة في أوائل السور',
            'ذلك الكتاب لا ريب فيه: هذا القرآن الكريم لا شك فيه',
            'الذين يؤمنون بالغيب: صفة المتقين أنهم يؤمنون بما غاب عنهم'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df['clean_text'] = df['text'].apply(preprocess_arabic_text)
    df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
    
    return df

@st.cache_data
def load_quran_dataset():
    """Load Quran dataset"""
    if not DATASETS_AVAILABLE:
        st.warning("⚠️ مكتبة datasets غير مثبتة - استخدام بيانات تجريبية")
        return create_sample_data()
    
    try:
        with st.spinner("📖 جاري تحميل بيانات القرآن..."):
            dataset = load_dataset("MohamedRashad/Quran-Tafseer")
            
            if 'train' in dataset:
                df = pd.DataFrame(dataset['train'])
            else:
                split_name = list(dataset.keys())[0]
                df = pd.DataFrame(dataset[split_name])
            
            # Handle column mapping
            if 'ayah' in df.columns and 'text' not in df.columns:
                df['text'] = df['ayah']
            if 'verse' in df.columns and 'text' not in df.columns:
                df['text'] = df['verse']
            
            # Ensure required columns
            if 'text' not in df.columns:
                st.error("❌ لا يمكن العثور على عمود النص")
                return create_sample_data()
            
            if 'surah' not in df.columns:
                df['surah'] = 'غير محدد'
            if 'ayah_number' not in df.columns:
                df['ayah_number'] = range(1, len(df) + 1)
            if 'tafseer' not in df.columns:
                df['tafseer'] = ''
            
            # Clean data
            df['text'] = df['text'].fillna('').astype(str)
            df['tafseer'] = df['tafseer'].fillna('').astype(str)
            df['clean_text'] = df['text'].apply(preprocess_arabic_text)
            df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
            
            # Remove empty rows
            df = df[df['clean_text'].str.strip() != ''].copy()
            df = df.reset_index(drop=True)
            
            st.success(f"✅ تم تحميل {len(df):,} آية من Hugging Face")
            return df
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات: {e}")
        return create_sample_data()

def simple_text_search(query, df, top_k=10):
    """Simple text search"""
    try:
        clean_query = preprocess_arabic_text(query).lower()
        if not clean_query.strip():
            return pd.DataFrame(), []
        
        scores = []
        query_words = set(clean_query.split())
        
        for _, row in df.iterrows():
            text = preprocess_arabic_text(str(row['text'])).lower()
            tafseer = preprocess_arabic_text(str(row.get('tafseer', ''))).lower()
            combined_text = text + ' ' + tafseer
            text_words = set(combined_text.split())
            
            if query_words:
                overlap = len(query_words.intersection(text_words))
                score = overlap / len(query_words)
            else:
                score = 0
            
            scores.append(score)
        
        scores = np.array(scores)
        top_indices = scores.argsort()[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        valid_mask = top_scores > 0
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_scores.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث: {e}")
        return pd.DataFrame(), []

def tfidf_search(query, df, top_k=10):
    """TF-IDF search"""
    try:
        if not SKLEARN_AVAILABLE:
            return simple_text_search(query, df, top_k)
        
        clean_query = preprocess_arabic_text(query)
        if not clean_query.strip():
            return pd.DataFrame(), []
        
        texts = df['clean_text'].fillna('').tolist()
        if not texts:
            return pd.DataFrame(), []
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        corpus = [clean_query] + texts
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[0]
        document_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        valid_mask = top_similarities > 0.01
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث الذكي: {e}")
        return simple_text_search(query, df, top_k)

def display_ayah_card(row, similarity_score=None, model_type=None, card_id=None):
    """Display ayah card"""
    if card_id is None:
        card_id = f"ayah_{hash(str(row.get('text', ''))[:50])}"
    
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            surah_name = row.get('surah', 'غير محدد')
            ayah_number = row.get('ayah_number', '')
            
            st.markdown(f"""
            <div class="surah-badge">📖 سورة {surah_name}</div>
            <span class="ayah-number">{ayah_number}</span>
            """, unsafe_allow_html=True)
        
        with col2:
            if similarity_score is not None:
                percentage = int(similarity_score * 100)
                st.markdown(f"🎯 {percentage}%")
        
        with col3:
            if model_type:
                st.markdown(f"🔍 {model_type}")
        
        # Arabic text
        ayah_text = row.get('text', '')
        if ayah_text:
            st.markdown(f'<div class="premium-arabic">{ayah_text}</div>', unsafe_allow_html=True)
        
        # Tafseer
        tafseer_text = row.get('tafseer', '')
        if tafseer_text and tafseer_text.strip():
            st.markdown("**📚 التفسير:**")
            st.markdown(f'<div class="tafseer-text">{tafseer_text}</div>', unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📚 حفظ", key=f"save_{card_id}", use_container_width=True):
                if 'saved_ayahs' not in st.session_state:
                    st.session_state.saved_ayahs = []
                st.session_state.saved_ayahs.append(ayah_text)
                st.success("✅ تم حفظ الآية!")
        
        with col2:
            if st.button("📋 نسخ", key=f"copy_{card_id}", use_container_width=True):
                st.code(ayah_text, language="text")
                st.success("📋 يمكنك نسخ النص أعلاه!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'search_mode': 'tfidf',
        'saved_ayahs': [],
        'search_history': [],
        'daily_verses_read': 0,
        'total_verses_read': 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_model_status():
    """Show model status"""
    st.markdown('<div class="model-status">', unsafe_allow_html=True)
    
    tfidf_class = "model-active" if SKLEARN_AVAILABLE else "model-inactive"
    tfidf_icon = "📊" if SKLEARN_AVAILABLE else "❌"
    st.markdown(f'<div class="model-badge {tfidf_class}"><span>{tfidf_icon}</span><span>TF-IDF</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="model-badge model-active"><span>🔍</span><span>بحث نصي</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_search_tab(df):
    """Search tab"""
    st.markdown("### 🔍 البحث الذكي في القرآن والتفسير")
    
    # Search mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 TF-IDF (ذكي)", 
                    disabled=not SKLEARN_AVAILABLE,
                    type="primary" if st.session_state.search_mode == 'tfidf' else "secondary",
                    use_container_width=True):
            st.session_state.search_mode = 'tfidf'
    
    with col2:
        if st.button("🔍 بحث نصي", 
                    type="primary" if st.session_state.search_mode == 'simple' else "secondary",
                    use_container_width=True):
            st.session_state.search_mode = 'simple'
    
    # Search input
    col_search, col_button = st.columns([4, 1])
    
    with col_search:
        search_query = st.text_input(
            "",
            placeholder="ابحث في القرآن والتفسير... مثال: 'الصبر'، 'الرحمة'، 'الجنة'",
            key="main_search",
            label_visibility="collapsed"
        )
    
    with col_button:
        search_pressed = st.button("🔍 بحث", use_container_width=True)
    
    # Search options
    with st.expander("⚙️ خيارات البحث", expanded=False):
        max_results = st.slider("عدد النتائج", 5, 50, 10)
        search_in_tafseer = st.checkbox("البحث في التفسير أيضاً", value=True)
    
    # Perform search
    if search_query and (search_pressed or search_query):
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, search_query)
            st.session_state.search_history = st.session_state.search_history[:20]
        
        with st.spinner("🔍 جاري البحث..."):
            search_df = df.copy()
            if search_in_tafseer:
                search_df['combined_text'] = (
                    search_df['clean_text'].fillna('') + ' ' + 
                    search_df.get('clean_tafseer', '').fillna('')
                )
                search_df['clean_text'] = search_df['combined_text']
            
            if st.session_state.search_mode == 'tfidf' and SKLEARN_AVAILABLE:
                results_df, similarities = tfidf_search(search_query, search_df, max_results)
                model_type = 'TF-IDF'
            else:
                results_df, similarities = simple_text_search(search_query, search_df, max_results)
                model_type = 'بحث نصي'
            
            if not results_df.empty:
                st.success(f"🎯 تم العثور على {len(results_df)} نتيجة")
                
                if similarities:
                    avg_similarity = np.mean(similarities) * 100
                    max_similarity = np.max(similarities) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("متوسط الدقة", f"{avg_similarity:.1f}%")
                    with col2:
                        st.metric("أعلى دقة", f"{max_similarity:.1f}%")
                    with col3:
                        unique_surahs = len(results_df['surah'].unique())
                        st.metric("عدد السور", unique_surahs)
                
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    similarity = similarities[idx] if idx < len(similarities) else None
                    display_ayah_card(row, similarity, model_type, f"result_{idx}")
                    
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f9fafb; border-radius: 20px; margin: 2rem 0;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                    <h3>لا توجد نتائج</h3>
                    <p>جرب كلمات مختلفة أو غير نوع البحث</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick suggestions
    if not search_query:
        st.markdown("### 🚀 اقتراحات البحث")
        
        suggestions = ["الصبر", "الرحمة", "الجنة", "التوبة", "الدعاء", "العدل"]
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(f"🔍 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.main_search = suggestion
                    st.rerun()

def show_saved_tab():
    """Saved ayahs tab"""
    st.markdown("### 📚 الآيات المحفوظة")
    
    if not st.session_state.saved_ayahs:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f9fafb; border-radius: 20px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
            <h3>لا توجد آيات محفوظة</h3>
            <p>احفظ آياتك المفضلة من نتائج البحث</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.metric("إجمالي المحفوظة", len(st.session_state.saved_ayahs))
    
    for i, ayah in enumerate(st.session_state.saved_ayahs):
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-arabic">{ayah}</div>
        </div>
        """, unsafe_allow_html=True)

def show_stats_tab(df):
    """Statistics tab"""
    st.markdown("### 📊 الإحصائيات")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("إجمالي الآيات", f"{len(df):,}")
    
    with col2:
        unique_surahs = len(df['surah'].unique())
        st.metric("عدد السور", unique_surahs)
    
    with col3:
        has_tafseer = len(df[df['tafseer'].notna() & (df['tafseer'] != '')])
        st.metric("آيات بتفسير", f"{has_tafseer:,}")

def show_about_tab():
    """About tab"""
    st.markdown("### ℹ️ حول التطبيق")
    
    st.markdown("""
    **🤖 تطبيق متطور للبحث في القرآن الكريم**
    
    يستخدم هذا التطبيق تقنيات متطورة للبحث في القرآن الكريم وتفسيره:
    - **TF-IDF**: البحث الذكي باستخدام تحليل المصطلحات
    - **البحث النصي**: البحث التقليدي بمطابقة الكلمات
    - **معالجة النصوص العربية**: تطبيع النصوص وإزالة التشكيل
    
    **📚 مصدر البيانات:** MohamedRashad/Quran-Tafseer على Hugging Face
    """)

def main():
    """Main application"""
    initialize_session_state()
    
    with st.spinner("🚀 جاري تحضير التطبيق..."):
        df = load_quran_dataset()
    
    if df.empty:
        st.error("❌ لا يمكن تحميل بيانات القرآن")
        st.stop()
    
    # Header
    st.markdown(f"""
    <div class="premium-header">
        <h1 class="header-title">تفسير القرآن الذكي</h1>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">Smart Quran Tafseer with AI Search</p>
        <p>بحث ذكي في {len(df):,} آية قرآنية مع التفسير</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_model_status()
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 البحث", "📚 المحفوظة", "📊 الإحصائيات", "ℹ️ حول التطبيق"])
    
    with tab1:
        show_search_tab(df)
    
    with tab2:
        show_saved_tab()
    
    with tab3:
        show_stats_tab(df)
    
    with tab4:
        show_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: var(--gray-500); padding: 2rem;">
        <p>📖 <strong>تفسير القرآن الذكي</strong> - تم التطوير بـ ❤️ لخدمة كتاب الله</p>
        <p style="font-size: 0.9rem;">آخر تحديث: {datetime.now().strftime('%Y-%m-%d')}</p>
        <p style="font-size: 0.8rem;">البيانات من: MohamedRashad/Quran-Tafseer على Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
