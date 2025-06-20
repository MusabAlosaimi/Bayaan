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
                score = overlap / len(query_words)
            else:
                score = 0
            
            scores.append(score)
        
        scores = np.array(scores)
        top_indices = scores.argsort()[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        # Filter by minimum score
        valid_mask = top_scores > 0
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_scores.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث النصي: {e}")
        return pd.DataFrame(), []

def tfidf_search(query, df, top_k=10, vectorizer=None):
    """TF-IDF search with fallback to simple search"""
    try:
        if not SKLEARN_AVAILABLE:
            return simple_text_search(query, df, top_k)
        
        clean_query = preprocess_arabic_text(query)
        if not clean_query.strip():
            return pd.DataFrame(), []
        
        texts = df['clean_text'].fillna('').tolist()
        if not texts:
            return pd.DataFrame(), []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Fit on corpus including query
        corpus = [clean_query] + texts
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[0]
        document_vectors = tfidf_matrix[1:]
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by minimum similarity
        valid_mask = top_similarities > 0.01
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث بـ TF-IDF: {e}")
        return simple_text_search(query, df, top_k)

def display_ayah_card(row, similarity_score=None, model_types=None, card_id=None):
    """Enhanced display for Quran ayah with tafseer"""
    if card_id is None:
        card_id = f"ayah_{hash(str(row.get('text', ''))[:50])}"
    
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div style="padding: 2rem;">', unsafe_allow_html=True)
        
        # Header with surah and ayah info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            surah_name = row.get('surah', 'غير محدد')
            ayah_number = row.get('ayah_number', '')
            
            st.markdown(f"""
            <div class="surah-badge">
                📖 سورة {surah_name}
            </div>
            <span class="ayah-number">{ayah_number}</span>
            """, unsafe_allow_html=True)
        
        with col2:
            if similarity_score is not None:
                percentage = int(similarity_score * 100)
                color = "var(--primary-500)" if percentage > 70 else "var(--accent-500)" if percentage > 40 else "var(--secondary-500)"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}20, {color}10);
                    color: {color};
                    padding: 6px 12px;
                    border-radius: var(--radius-3xl);
                    text-align: center;
                    font-weight: 600;
                    border: 1px solid {color}30;
                ">
                    🎯 {percentage}%
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if model_types:
                models_text = " + ".join(set(model_types)) if isinstance(model_types, list) else str(model_types)
                st.markdown(f"""
                <div style="
                    background: var(--gray-100);
                    color: var(--gray-700);
                    padding: 6px 12px;
                    border-radius: var(--radius-lg);
                    text-align: center;
                    font-size: 0.8rem;
                    font-weight: 500;
                ">
                    {models_text}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Arabic text (Ayah)
        ayah_text = row.get('text', '')
        if ayah_text:
            st.markdown(f"""
            <div class="premium-arabic">
                {ayah_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Tafseer (if available)
        tafseer_text = row.get('tafseer', '')
        if tafseer_text and tafseer_text.strip() and tafseer_text != 'nan':
            st.markdown("**📚 التفسير:**")
            st.markdown(f"""
            <div class="tafseer-text">
                {tafseer_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📚 حفظ", key=f"save_{card_id}", use_container_width=True):
                if 'saved_ayahs' not in st.session_state:
                    st.session_state.saved_ayahs = []
                st.session_state.saved_ayahs.append(ayah_text)
                st.success("✅ تم حفظ الآية!")
        
        with col2:
            if st.button("📋 نسخ الآية", key=f"copy_ayah_{card_id}", use_container_width=True):
                st.code(ayah_text, language="text")
                st.success("📋 يمكنك نسخ النص أعلاه!")
        
        with col3:
            if tafseer_text and st.button("📝 نسخ التفسير", key=f"copy_tafseer_{card_id}", use_container_width=True):
                st.code(tafseer_text, language="text")
                st.success("📝 يمكنك نسخ التفسير أعلاه!")
        
        with col4:
            if st.button("🔍 آيات مشابهة", key=f"similar_{card_id}", use_container_width=True):
                st.session_state.find_similar_ayah = ayah_text
                st.rerun()
        
        st.markdown('</div></div>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'search_mode': 'tfidf',
        'saved_ayahs': [],
        'reading_history': [],
        'search_history': [],
        'daily_verses_read': 0,
        'total_verses_read': 0,
        'last_search_date': datetime.now().date(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_model_status(models):
    """Display current model status"""
    st.markdown('<div class="model-status">', unsafe_allow_html=True)
    
    # TF-IDF Status
    tfidf_class = "model-active" if models['tfidf_available'] else "model-inactive"
    tfidf_icon = "📊" if models['tfidf_available'] else "❌"
    st.markdown(f'''
    <div class="model-badge {tfidf_class}">
        <span>{tfidf_icon}</span>
        <span>TF-IDF متاح</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Simple Search Status
    st.markdown(f'''
    <div class="model-badge model-active">
        <span>🔍</span>
        <span>بحث نصي متاح</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Future Models
    st.markdown(f'''
    <div class="model-badge model-inactive">
        <span>🤖</span>
        <span>BERT (قريباً)</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="model-badge model-inactive">
        <span>🧠</span>
        <span>Word2Vec (قريباً)</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_search_tab(df, models):
    """Enhanced search tab"""
    st.markdown("### 🔍 البحث الذكي في القرآن والتفسير")
    
    # Search mode selection
    st.markdown("**🎯 اختر نوع البحث:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 TF-IDF (ذكي)", 
                    key="mode_tfidf", 
                    use_container_width=True,
                    disabled=not models['tfidf_available'],
                    type="primary" if st.session_state.search_mode == 'tfidf' else "secondary"):
            st.session_state.search_mode = 'tfidf'
    
    with col2:
        if st.button("🔍 بحث نصي (بسيط)", 
                    key="mode_simple", 
                    use_container_width=True,
                    type="primary" if st.session_state.search_mode == 'simple' else "secondary"):
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
        search_pressed = st.button("🔍 بحث", key="search_btn", use_container_width=True)
    
    # Advanced search options
    with st.expander("⚙️ خيارات البحث المتقدمة", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider("عدد النتائج", 5, 50, 10, key="max_results")
        
        with col2:
            search_in_tafseer = st.checkbox("البحث في التفسير أيضاً", value=True, key="search_tafseer")
    
    # Perform search
    if search_query and (search_pressed or search_query):
        # Add to search history
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, search_query)
            st.session_state.search_history = st.session_state.search_history[:20]
        
        with st.spinner(f"🔍 جاري البحث..."):
            results_df = pd.DataFrame()
            similarities = []
            
            # Prepare search dataframe
            search_df = df.copy()
            if search_in_tafseer and 'tafseer' in df.columns:
                # Combine ayah and tafseer for search
                search_df['combined_text'] = (
                    search_df['clean_text'].fillna('') + ' ' + 
                    search_df.get('clean_tafseer', '').fillna('')
                )
                search_df['clean_text'] = search_df['combined_text']
            
            # Execute search based on selected mode
            if st.session_state.search_mode == 'tfidf' and models['tfidf_available']:
                results_df, similarities = tfidf_search(search_query, search_df, max_results)
                model_type = 'TF-IDF'
            else:
                results_df, similarities = simple_text_search(search_query, search_df, max_results)
                model_type = 'بحث نصي'
            
            # Display results
            if not results_df.empty:
                st.success(f"🎯 تم العثور على {len(results_df)} نتيجة")
                
                # Results summary
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
                
                # Display results
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    similarity = similarities[idx] if idx < len(similarities) else None
                    display_ayah_card(row, similarity, model_type, f"result_{idx}")
                    
            else:
                st.markdown("""
                <div style="
                    text-align: center;
                    padding: 3rem 2rem;
                    background: var(--gray-50);
                    border-radius: var(--radius-2xl);
                    border: 2px dashed var(--gray-300);
                    margin: 2rem 0;
                ">
                    <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">🔍</div>
                    <h3 style="color: var(--gray-600); margin-bottom: 0.5rem;">لا توجد نتائج</h3>
                    <p style="color: var(--gray-500);">جرب كلمات مختلفة أو غير نوع البحث</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick search suggestions
    if not search_query:
        st.markdown("### 🚀 اقتراحات البحث")
        
        suggestions = [
            "الصبر", "الرحمة", "الجنة", 
            "التوبة", "الدعاء", "العدل",
            "الصلاة", "الأمانة", "التقوى"
        ]
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(f"🔍 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.main_search = suggestion
                    st.rerun()
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📚 آخر عمليات البحث")
            
            for i, query in enumerate(st.session_state.search_history[:5]):
                if st.button(f"📖 {query}", key=f"history_{i}", use_container_width=True):
                    st.session_state.main_search = query
                    st.rerun()

def show_saved_tab():
    """Show saved ayahs"""
    st.markdown("### 📚 الآيات المحفوظة")
    
    if not st.session_state.saved_ayahs:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            background: var(--gray-50);
            border-radius: var(--radius-2xl);
            border: 2px dashed var(--gray-300);
            margin: 2rem 0;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">📚</div>
            <h3 style="color: var(--gray-600); margin-bottom: 0.5rem;">لا توجد آيات محفوظة</h3>
            <p style="color: var(--gray-500);">احفظ آياتك المفضلة من نتائج البحث لتظهر هنا</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("إجمالي المحفوظة", len(st.session_state.saved_ayahs))
    with col2:
        st.metric("قراءات اليوم", st.session_state.daily_verses_read)
    with col3:
        st.metric("إجمالي القراءات", st.session_state.total_verses_read)
    
    # Display saved ayahs
    st.markdown("### 📖 آياتك المحفوظة")
    
    for i, ayah in enumerate(st.session_state.saved_ayahs):
        st.markdown(f"""
        <div class="premium-card">
            <div style="padding: 2rem;">
                <div class="premium-arabic">{ayah}</div>
                <div style="margin-top: 1rem;">
                    <small style="color: var(--gray-500);">محفوظة في {datetime.now().strftime('%Y-%m-%d')}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_stats_tab(df, models):
    """Statistics tab"""
    st.markdown("### 📊 الإحصائيات والتحليلات")
    
    # Dataset statistics
    st.markdown("#### 📚 إحصائيات المجموعة")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("إجمالي الآيات", f"{len(df):,}")
    
    with col2:
        unique_surahs = len(df['surah'].unique())
        st.metric("عدد السور", unique_surahs)
    
    with col3:
        has_tafseer = len(df[df['tafseer'].notna() & (df['tafseer'] != '')])
        st.metric("آيات بتفسير", f"{has_tafseer:,}")
    
    with col4:
        avg_length = df['text'].str.len().mean()
        st.metric("متوسط طول الآية", f"{avg_length:.0f} حرف")
    
    # Model performance
    st.markdown("#### 🤖 حالة النماذج")
    show_model_status(models)
    
    # User statistics
    st.markdown("#### 👤 إحصائياتك الشخصية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("الآيات المحفوظة", len(st.session_state.saved_ayahs))
    
    with col2:
        st.metric("عمليات البحث", len(st.session_state.search_history))
    
    with col3:
        st.metric("إجمالي القراءات", st.session_state.total_verses_read)

def show_about_tab():
    """About section"""
    st.markdown("### ℹ️ حول تطبيق تفسير القرآن الذكي")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, var(--primary-50), var(--secondary-50));
        padding: 2rem;
        border-radius: var(--radius-2xl);
        border: 1px solid var(--primary-200);
        margin: 2rem 0;
    ">
        <h4 style="color: var(--primary-700); margin-bottom: 1rem;">🤖 تطبيق متطور للبحث في القرآن الكريم</h4>
        <p style="color: var(--gray-700); line-height: 1.6;">
            يستخدم هذا التطبيق تقنيات متطورة للبحث في القرآن الكريم وتفسيره.
            يدعم البحث النصي التقليدي والبحث الذكي باستخدام TF-IDF.
            البيانات محملة من Hugging Face Dataset: MohamedRashad/Quran-Tafseer
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical details
    st.markdown("#### 🛠️ التقنيات المستخدمة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 محركات البحث:**
        - **TF-IDF**: تحليل تكرار المصطلحات والوثائق
        - **البحث النصي**: مطابقة الكلمات المباشرة
        - **معالجة النصوص العربية**: إزالة التشكيل والتطبيع
        """)
    
    with col2:
        st.markdown("""
        **📚 مصادر البيانات:**
        - Hugging Face Dataset: MohamedRashad/Quran-Tafseer
        - تفسير شامل للقرآن الكريم
        - معالجة متقدمة للنصوص العربية
        """)
    
    # Installation guide
    st.markdown("#### 📥 المكتبات المطلوبة")
    
    requirements = """
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
datasets>=2.14.0
scikit-learn>=1.3.0
"""
    
    st.code(requirements, language="text")
    
    st.info("""
    💡 **نصائح لتحسين الأداء:**
    - استخدم كلمات مفتاحية واضحة ومحددة
    - جرب البحث في التفسير للحصول على نتائج أوسع
    - احفظ آياتك المفضلة للرجوع إليها لاحقاً
    - استخدم اقتراحات البحث للاستكشاف
    """)

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Show loading message
    with st.spinner("🚀 جاري تحضير التطبيق..."):
        # Load data and models
        df = load_quran_dataset()
        models = initialize_models()
    
    if df.empty:
        st.error("❌ لا يمكن تحميل بيانات القرآن")
        st.stop()
    
    # Header
    st.markdown(f"""
    <div class="premium-header">
        <div class="header-content">
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📖</div>
            </div>
            <h1 class="header-title">تفسير القرآن الذكي</h1>
            <p style="font-size: 1.4rem; color: rgba(255, 255, 255, 0.95); margin-bottom: 1rem;">
                Smart Quran Tafseer with AI-Powered Search
            </p>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem;">
                بحث ذكي في {len(df):,} آية قرآنية مع التفسير
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model status
    show_model_status(models)
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 البحث الذكي", "📚 المحفوظة", "📊 الإحصائيات", "ℹ️ حول التطبيق"])
    
    with tab1:
        show_search_tab(df, models)
    
    with tab2:
        show_saved_tab()
    
    with tab3:
        show_stats_tab(df, models)
    
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
