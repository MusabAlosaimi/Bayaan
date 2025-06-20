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
    st.warning("⚠️ مكتبة joblib غير مثبتة. للتثبيت: pip install joblib")

# Try to import advanced ML libraries
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

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.info("ℹ️ مكتبة sentence-transformers غير مثبتة - ميزات BERT معطلة")

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    st.info("ℹ️ مكتبة gensim غير مثبتة - ميزات Word2Vec معطلة")

import warnings
warnings.filterwarnings('ignore')

# Enhanced CSS with new branding for Quran Tafseer
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@300;400;500;600;700&display=swap');
    
    /* Advanced CSS Variables - Quran Theme Colors */
    :root {
        /* Primary Colors - Deep Green & Gold */
        --primary-50: #f0fdf4;
        --primary-100: #dcfce7;
        --primary-200: #bbf7d0;
        --primary-300: #86efac;
        --primary-400: #4ade80;
        --primary-500: #22c55e;
        --primary-600: #16a34a;
        --primary-700: #15803d;
        --primary-800: #166534;
        --primary-900: #14532d;
        
        /* Secondary Colors - Royal Blue */
        --secondary-50: #eff6ff;
        --secondary-100: #dbeafe;
        --secondary-200: #bfdbfe;
        --secondary-300: #93c5fd;
        --secondary-400: #60a5fa;
        --secondary-500: #3b82f6;
        --secondary-600: #2563eb;
        --secondary-700: #1d4ed8;
        --secondary-800: #1e40af;
        --secondary-900: #1e3a8a;
        
        /* Accent Colors - Golden */
        --accent-50: #fffbeb;
        --accent-100: #fef3c7;
        --accent-200: #fde68a;
        --accent-300: #fcd34d;
        --accent-400: #fbbf24;
        --accent-500: #f59e0b;
        --accent-600: #d97706;
        --accent-700: #b45309;
        --accent-800: #92400e;
        --accent-900: #78350f;
        
        /* Neutral Colors */
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        
        /* Spacing & Shadows */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --radius-2xl: 20px;
        --radius-3xl: 24px;
    }
    
    .stApp {
        background: linear-gradient(135deg, 
            var(--primary-50) 0%, 
            var(--secondary-50) 25%,
            var(--accent-50) 50%,
            var(--primary-50) 75%,
            var(--secondary-50) 100%);
        min-height: 100vh;
        font-family: 'Inter', 'Noto Kufi Arabic', sans-serif;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Enhanced Header for Quran Tafseer */
    .premium-header {
        background: linear-gradient(135deg, 
            rgba(22, 163, 74, 0.95) 0%,
            rgba(37, 99, 235, 0.9) 50%,
            rgba(217, 119, 6, 0.95) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        color: white;
        padding: 3rem 0;
        margin: -1rem -1rem 3rem -1rem;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Amiri', serif;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    /* Enhanced Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px) saturate(180%);
        border-radius: var(--radius-2xl);
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 2rem;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .premium-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-2xl);
        border-color: rgba(22, 163, 74, 0.3);
    }
    
    /* Enhanced Arabic Text */
    .premium-arabic {
        font-family: 'Amiri', 'Noto Kufi Arabic', serif;
        font-size: 1.8rem;
        line-height: 2;
        color: var(--gray-800);
        margin: 2rem 0;
        text-align: right;
        direction: rtl;
        background: linear-gradient(135deg, 
            rgba(22, 163, 74, 0.05) 0%,
            rgba(255, 255, 255, 0.8) 50%,
            rgba(37, 99, 235, 0.05) 100%);
        padding: 2rem;
        border-radius: var(--radius-xl);
        border-right: 6px solid var(--primary-600);
        border-left: 2px solid var(--secondary-500);
        box-shadow: var(--shadow-md);
    }
    
    /* Tafseer specific styles */
    .tafseer-text {
        font-family: 'Noto Kufi Arabic', sans-serif;
        font-size: 1.2rem;
        line-height: 1.8;
        color: var(--gray-700);
        text-align: right;
        direction: rtl;
        background: var(--gray-50);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
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
        border-radius: var(--radius-3xl);
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 8px 4px;
    }
    
    /* Model status indicators */
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
        border-radius: var(--radius-3xl);
        font-size: 0.85rem;
        font-weight: 600;
        border: 2px solid transparent;
    }
    
    .model-active {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        color: white;
        border-color: var(--primary-400);
    }
    
    .model-inactive {
        background: var(--gray-200);
        color: var(--gray-600);
        border-color: var(--gray-300);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced utility functions for Arabic text processing
def remove_tashkeel(text):
    """Remove Arabic diacritics for better text processing"""
    if not isinstance(text, str):
        return ""
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

def preprocess_arabic_text(text):
    """Enhanced Arabic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Remove diacritics
    text = remove_tashkeel(text)
    
    # Normalize Arabic letters
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ي', 'ى', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_sample_data():
    """Create comprehensive sample data for demonstration"""
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
            'الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ وَيُقِيمُونَ الصَّلَاةَ وَمِمَّا رَزَقْنَاهُمْ يُنفِقُونَ',
            'وَالَّذِينَ يُؤْمِنُونَ بِمَا أُنزِلَ إِلَيْكَ وَمَا أُنزِلَ مِن قَبْلِكَ وَبِالْآخِرَةِ هُمْ يُوقِنُونَ',
            'أُولَٰئِكَ عَلَىٰ هُدًى مِّن رَّبِّهِمْ ۖ وَأُولَٰئِكَ هُمُ الْمُفْلِحُونَ',
            'إِنَّ الَّذِينَ كَفَرُوا سَوَاءٌ عَلَيْهِمْ أَأَنذَرْتَهُمْ أَمْ لَمْ تُنذِرْهُمْ لَا يُؤْمِنُونَ',
            'خَتَمَ اللَّهُ عَلَىٰ قُلُوبِهِمْ وَعَلَىٰ سَمْعِهِمْ ۖ وَعَلَىٰ أَبْصَارِهِمْ غِشَاوَةٌ ۖ وَلَهُمْ عَذَابٌ عَظِيمٌ',
            'وَمِنَ النَّاسِ مَن يَقُولُ آمَنَّا بِاللَّهِ وَبِالْيَوْمِ الْآخِرِ وَمَا هُم بِمُؤْمِنِينَ',
            'يُخَادِعُونَ اللَّهَ وَالَّذِينَ آمَنُوا وَمَا يَخْدَعُونَ إِلَّا أَنفُسَهُمْ وَمَا يَشْعُرُونَ',
            'فِي قُلُوبِهِم مَّرَضٌ فَزَادَهُمُ اللَّهُ مَرَضًا ۖ وَلَهُمْ عَذَابٌ أَلِيمٌ بِمَا كَانُوا يَكْذِبُونَ',
            'وَإِذَا قِيلَ لَهُمْ لَا تُفْسِدُوا فِي الْأَرْضِ قَالُوا إِنَّمَا نَحْنُ مُصْلِحُونَ',
            'أَلَا إِنَّهُمْ هُمُ الْمُفْسِدُونَ وَلَٰكِن لَّا يَشْعُرُونَ',
            'وَإِذَا قِيلَ لَهُمْ آمِنُوا كَمَا آمَنَ النَّاسُ قَالُوا أَنُؤْمِنُ كَمَا آمَنَ السُّفَهَاءُ ۗ أَلَا إِنَّهُمْ هُمُ السُّفَهَاءُ وَلَٰكِن لَّا يَعْلَمُونَ'
        ],
        'surah': ['الفاتحة'] * 7 + ['البقرة'] * 13,
        'ayah_number': list(range(1, 8)) + list(range(1, 14)),
        'tafseer': [
            'بسم الله الرحمن الرحيم: افتتاح كل سورة بحمد الله والثناء عليه، وهي آية من القرآن في أول كل سورة إلا براءة',
            'الحمد لله رب العالمين: الثناء على الله بصفاته الجميلة وأفعاله الحميدة، وهو سبحانه المستحق للحمد والثناء، رب جميع المخلوقات',
            'الرحمن الرحيم: من أسماء الله الحسنى، الرحمن ذو الرحمة الواسعة التي تشمل جميع المخلوقات، والرحيم ذو الرحمة الخاصة بالمؤمنين',
            'مالك يوم الدين: الله هو المالك المتصرف يوم القيامة، يوم الجزاء والحساب، لا ملك ولا متصرف سواه',
            'إياك نعبد وإياك نستعين: التوحيد في العبادة والاستعانة، فنعبد الله وحده لا شريك له، ونستعين به وحده في جميع أمورنا',
            'اهدنا الصراط المستقيم: دعاء بالهداية إلى الطريق المستقيم وهو الإسلام، الطريق الواضح الذي لا اعوجاج فيه',
            'صراط الذين أنعمت عليهم: طريق الأنبياء والصديقين والشهداء والصالحين الذين أنعم الله عليهم، غير المغضوب عليهم وهم اليهود، ولا الضالين وهم النصارى',
            'الم: من الحروف المقطعة في أوائل السور، والله أعلم بمرادها، وقيل إنها أسماء للسور أو أقسام أقسم الله بها',
            'ذلك الكتاب لا ريب فيه: هذا القرآن الكريم لا شك فيه ولا ريب أنه من عند الله، وهو هداية ونور للمتقين الذين يخافون الله',
            'الذين يؤمنون بالغيب: صفة المتقين أنهم يؤمنون بما غاب عنهم مما أخبر الله به كالملائكة والجنة والنار، ويقيمون الصلاة حق إقامتها، وينفقون مما رزقهم الله في سبيله',
            'والذين يؤمنون بما أنزل إليك: المؤمنون يصدقون بالقرآن المنزل على محمد صلى الله عليه وسلم، وبالكتب السابقة كالتوراة والإنجيل، ويوقنون بالآخرة ويؤمنون بها إيماناً جازماً',
            'أولئك على هدى من ربهم: هؤلاء المتصفون بالصفات المذكورة على نور وبصيرة من ربهم، وهم المفلحون الفائزون في الدنيا والآخرة',
            'إن الذين كفروا سواء عليهم: الكافرون المعاندون لا ينفعهم الإنذار ولا تركه، فقد طبع الله على قلوبهم فلا يؤمنون',
            'ختم الله على قلوبهم: طبع الله على قلوبهم وأسماعهم فلا تعي الحق، وجعل على أبصارهم غشاوة فلا تبصر الهدى، ولهم عذاب عظيم في الآخرة',
            'ومن الناس من يقول آمنا بالله: صفة المنافقين الذين يظهرون الإيمان ويبطنون الكفر، يقولون آمنا بالله واليوم الآخر وما هم بمؤمنين حقيقة',
            'يخادعون الله والذين آمنوا: المنافقون يحسبون أنهم يخدعون الله والمؤمنين بإظهار الإيمان، وما يخدعون إلا أنفسهم وما يشعرون بذلك',
            'في قلوبهم مرض: في قلوب المنافقين مرض الشك والنفاق، فزادهم الله مرضاً بسبب كفرهم وعنادهم، ولهم عذاب أليم بسبب كذبهم',
            'وإذا قيل لهم لا تفسدوا: إذا نُهي المنافقون عن الإفساد في الأرض قالوا إنما نحن مصلحون، وهم في الحقيقة مفسدون ولكن لا يدركون ذلك',
            'ألا إنهم هم المفسدون: تأكيد أن المنافقين هم المفسدون حقاً، ولكنهم لا يشعرون بإفسادهم ولا يدركون خطر أفعالهم',
            'وإذا قيل لهم آمنوا كما آمن الناس: إذا طُلب من المنافقين أن يؤمنوا إيماناً صادقاً كإيمان الصحابة، قالوا أنؤمن كما آمن السفهاء، والحقيقة أنهم هم السفهاء ولكن لا يعلمون'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df['clean_text'] = df['text'].apply(preprocess_arabic_text)
    df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
    
    return df

@st.cache_data
def load_quran_dataset():
    """Load Quran dataset from Hugging Face or use sample data"""
    
    if not DATASETS_AVAILABLE:
        st.warning("⚠️ مكتبة datasets غير مثبتة - استخدام بيانات تجريبية")
        return create_sample_data()
    
    try:
        with st.spinner("📖 جاري تحميل بيانات القرآن والتفسير من Hugging Face..."):
            
            # Load dataset from Hugging Face
            dataset = load_dataset("MohamedRashad/Quran-Tafseer")
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = pd.DataFrame(dataset['train'])
            else:
                # Use first available split
                split_name = list(dataset.keys())[0]
                df = pd.DataFrame(dataset[split_name])
                st.info(f"استخدام قسم البيانات: {split_name}")
            
            # Handle different column names
            column_mapping = {
                'ayah': 'text',
                'verse': 'text',
                'arabic': 'text',
                'surah_name': 'surah',
                'chapter': 'surah',
                'verse_number': 'ayah_number',
                'tafsir': 'tafseer'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                st.error("❌ لا يمكن العثور على عمود النص في البيانات")
                return create_sample_data()
            
            # Handle missing columns
            if 'surah' not in df.columns:
                df['surah'] = 'غير محدد'
            if 'ayah_number' not in df.columns:
                df['ayah_number'] = range(1, len(df) + 1)
            if 'tafseer' not in df.columns:
                df['tafseer'] = ''
            
            # Clean and preprocess
            df['text'] = df['text'].fillna('').astype(str)
            df['tafseer'] = df['tafseer'].fillna('').astype(str)
            df['clean_text'] = df['text'].apply(preprocess_arabic_text)
            df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
            
            # Remove empty rows
            df = df[df['clean_text'].str.strip() != ''].copy()
            df = df.dropna(subset=['text']).copy()
            df = df.reset_index(drop=True)
            
            st.success(f"✅ تم تحميل {len(df):,} آية من Hugging Face")
            return df
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات من Hugging Face: {e}")
        st.info("🔄 استخدام البيانات التجريبية...")
        return create_sample_data()

def initialize_models():
    """Initialize available models with safe fallbacks"""
    models = {
        'bert_available': False,
        'w2v_available': False,
        'word2vec_available': False,
        'tfidf_available': SKLEARN_AVAILABLE,
        'bert_vectors': None,
        'w2v_vectors': None,
        'word2vec_model': None,
        'tfidf_vectorizer': None
    }
    
    # For now, we don't require the model files to exist
    # This allows the app to run with basic functionality
    
    return models

def simple_text_search(query, df, top_k=10):
    """Simple text-based search as fallback"""
    try:
        clean_query = preprocess_arabic_text(query).lower()
        if not clean_query.strip():
            return pd.DataFrame(), []
        
        # Calculate simple text similarity
        scores = []
        for _, row in df.iterrows():
            text = preprocess_arabic_text(str(row['text'])).lower()
            tafseer = preprocess_arabic_text(str(row.get('tafseer', ''))).lower()
            combined_text = text + ' ' + tafseer
            
            # Simple word matching score
            query_words = set(clean_query.split())
            text_words = set(combined_text.split())
            
            if query_words:
                overlap = len(query_words.intersection(text_words))
