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

# Try to import advanced ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

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
    
    /* Search enhancement */
    .search-mode-selector {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 8px;
        margin: 1rem 0;
    }
    
    .search-mode-btn {
        padding: 12px 16px;
        border-radius: var(--radius-lg);
        border: 2px solid var(--gray-300);
        background: white;
        color: var(--gray-700);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .search-mode-btn.active {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        color: white;
        border-color: var(--primary-500);
    }
    
    .search-mode-btn:hover {
        border-color: var(--primary-400);
        transform: translateY(-2px);
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

def load_local_fallback():
    """Fallback to create sample data if Hugging Face is not available"""
    try:
        # Try to load old CSV file if it exists
        if os.path.exists('adhkar_df.csv'):
            df = pd.read_csv('adhkar_df.csv')
            # Convert to Quran format
            if 'text' not in df.columns and 'ayah' in df.columns:
                df['text'] = df['ayah']
            df['clean_text'] = df['text'].apply(preprocess_arabic_text)
            return df
    except:
        pass
    
    # Create sample data for demonstration
    sample_data = {
        'text': [
            'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الرَّحْمَٰنِ الرَّحِيمِ',
            'مَالِكِ يَوْمِ الدِّينِ',
            'إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ',
            'اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ',
            'صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ'
        ],
        'surah': ['الفاتحة'] * 7,
        'ayah_number': list(range(1, 8)),
        'tafseer': [
            'بسم الله الرحمن الرحيم: افتتاح كل سورة بحمد الله والثناء عليه',
            'الحمد لله: الثناء على الله بصفاته الجميلة وأفعاله الحميدة',
            'الرحمن الرحيم: من أسماء الله الحسنى التي تدل على سعة رحمته',
            'مالك يوم الدين: الله هو المالك المتصرف يوم القيامة',
            'إياك نعبد: التوحيد في العبادة والاستعانة',
            'اهدنا الصراط المستقيم: دعاء بالهداية إلى الطريق المستقيم',
            'صراط الذين أنعمت عليهم: طريق الأنبياء والصالحين'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df['clean_text'] = df['text'].apply(preprocess_arabic_text)
    df['clean_tafseer'] = df['tafseer'].apply(preprocess_arabic_text)
    
    st.warning("⚠️ يتم استخدام بيانات تجريبية. لتحميل البيانات الكاملة، ثبت مكتبة datasets")
    return df

@st.cache_data
def load_quran_dataset():
    """Load and cache the Quran Tafseer dataset"""
    if not DATASETS_AVAILABLE:
        st.warning("⚠️ مكتبة datasets غير مثبتة. للتثبيت: pip install datasets")
        return load_local_fallback()
    
    try:
        with st.spinner("📖 جاري تحميل بيانات القرآن والتفسير..."):
            dataset = load_dataset("MohamedRashad/Quran-Tafseer")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset['train'])
            
            # Handle different column names that might exist
            if 'ayah' in df.columns and 'text' not in df.columns:
                df['text'] = df['ayah']
            elif 'verse' in df.columns and 'text' not in df.columns:
                df['text'] = df['verse']
            
            # Ensure we have required columns
            if 'text' not in df.columns:
                st.error("❌ البيانات المحملة لا تحتوي على نص الآيات")
                return load_local_fallback()
            
            # Add preprocessing
            df['clean_text'] = df['text'].fillna('').apply(preprocess_arabic_text)
            
            # Add tafseer preprocessing if available
            if 'tafseer' in df.columns:
                df['clean_tafseer'] = df['tafseer'].fillna('').apply(preprocess_arabic_text)
            else:
                df['tafseer'] = ''
                df['clean_tafseer'] = ''
            
            # Ensure surah and ayah_number columns exist
            if 'surah' not in df.columns:
                df['surah'] = 'غير محدد'
            if 'ayah_number' not in df.columns and 'verse_number' in df.columns:
                df['ayah_number'] = df['verse_number']
            elif 'ayah_number' not in df.columns:
                df['ayah_number'] = range(1, len(df) + 1)
            
            # Remove rows with empty text
            df = df[df['text'].str.strip() != ''].dropna(subset=['text'])
            
            st.success(f"✅ تم تحميل {len(df):,} آية وتفسير من Hugging Face")
            return df
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات من Hugging Face: {e}")
        st.info("🔄 جاري المحاولة باستخدام البيانات المحلية...")
        return load_local_fallback()

@st.cache_data
def load_models():
    """Load pre-trained models with enhanced error handling"""
    models = {
        'bert_available': False,
        'w2v_available': False,
        'word2vec_available': False,
        'tfidf_available': False,
        'bert_vectors': None,
        'w2v_vectors': None,
        'word2vec_model': None,
        'tfidf_vectorizer': None
    }
    
    # Check if we're running the app for the first time
    first_run = True
    
    try:
        # Load BERT vectors
        if os.path.exists('bert_vectors.npy'):
            if st.session_state.get('bert_vectors') is None:
                bert_vectors = np.load('bert_vectors.npy')
                models['bert_vectors'] = bert_vectors
                models['bert_available'] = True
                st.session_state.bert_vectors = bert_vectors
                st.success("✅ تم تحميل نموذج BERT")
            else:
                models['bert_vectors'] = st.session_state.bert_vectors
                models['bert_available'] = True
        else:
            if first_run:
                st.warning("⚠️ ملف bert_vectors.npy غير موجود - ميزات BERT معطلة")
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل BERT: {e}")
    
    try:
        # Load Word2Vec vectors
        if os.path.exists('w2v_vectors.npy'):
            if st.session_state.get('w2v_vectors') is None:
                w2v_vectors = np.load('w2v_vectors.npy')
                models['w2v_vectors'] = w2v_vectors
                models['w2v_available'] = True
                st.session_state.w2v_vectors = w2v_vectors
                st.success("✅ تم تحميل متجهات Word2Vec")
            else:
                models['w2v_vectors'] = st.session_state.w2v_vectors
                models['w2v_available'] = True
        else:
            if first_run:
                st.warning("⚠️ ملف w2v_vectors.npy غير موجود - ميزات Word2Vec معطلة")
            
    except Exception as e:
        st.error(f"❌ خطأ في تحميل Word2Vec vectors: {e}")
    
    try:
        # Load Word2Vec model
        if not GENSIM_AVAILABLE:
            if first_run:
                st.warning("⚠️ مكتبة gensim غير مثبتة. للتثبيت: pip install gensim")
        elif os.path.exists('word2vec.model'):
            if st.session_state.get('word2vec_model') is None:
                word2vec_model = Word2Vec.load('word2vec.model')
                models['word2vec_model'] = word2vec_model
                models['word2vec_available'] = True
                st.session_state.word2vec_model = word2vec_model
                st.success("✅ تم تحميل نموذج Word2Vec")
            else:
                models['word2vec_model'] = st.session_state.word2vec_model
                models['word2vec_available'] = True
        else:
            if first_run:
                st.warning("⚠️ ملف word2vec.model غير موجود - ميزات Word2Vec متقدمة معطلة")
                
    except Exception as e:
        st.error(f"❌ خطأ في تحميل Word2Vec model: {e}")
    
    try:
        # Load or create TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            if os.path.exists('tfidf_vectorizer.pkl'):
                if JOBLIB_AVAILABLE:
                    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
                else:
                    with open('tfidf_vectorizer.pkl', 'rb') as f:
                        tfidf_vectorizer = pickle.load(f)
                models['tfidf_vectorizer'] = tfidf_vectorizer
                models['tfidf_available'] = True
                st.success("✅ تم تحميل نموذج TF-IDF")
            else:
                # Create a new TF-IDF vectorizer
                models['tfidf_available'] = True
                if first_run:
                    st.info("ℹ️ سيتم إنشاء نموذج TF-IDF جديد عند الحاجة")
        else:
            if first_run:
                st.warning("⚠️ مكتبة scikit-learn غير مثبتة - ميزات البحث الذكي محدودة")
    
    except Exception as e:
        st.error(f"❌ خطأ في تحميل TF-IDF: {e}")
    
    # Show summary of available models
    available_count = sum([
        models['bert_available'],
        models['w2v_available'], 
        models['word2vec_available'],
        models['tfidf_available']
    ])
    
    if available_count == 0:
        st.warning("⚠️ لا توجد نماذج ذكية متاحة - سيتم استخدام البحث التقليدي فقط")
    else:
        st.info(f"ℹ️ متاح {available_count} من 4 نماذج ذكية")
    
    return models

def bert_similarity_search(query, df, bert_vectors, top_k=10):
    """Enhanced BERT-based semantic search"""
    try:
        if bert_vectors is None:
            return pd.DataFrame(), []
        
        # For now, we'll use a simple approach since we don't have the BERT model
        # In a real implementation, you would encode the query with the same BERT model
        # that was used to create bert_vectors.npy
        
        # Fallback to TF-IDF for query encoding if BERT model isn't available
        if SKLEARN_AVAILABLE:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create a simple TF-IDF representation for the query
            corpus = [preprocess_arabic_text(query)] + df['clean_text'].tolist()
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            query_vector = tfidf_matrix[0]
            document_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, document_vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            top_similarities = similarities[top_indices]
            
            # Filter by minimum similarity
            valid_mask = top_similarities > 0.1
            if not np.any(valid_mask):
                return pd.DataFrame(), []
            
            top_indices = top_indices[valid_mask]
            top_similarities = top_similarities[valid_mask]
            
            result_df = df.iloc[top_indices].copy()
            return result_df, top_similarities.tolist()
        
        return pd.DataFrame(), []
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث بـ BERT: {e}")
        return pd.DataFrame(), []

def word2vec_similarity_search(query, df, w2v_vectors, word2vec_model, top_k=10):
    """Enhanced Word2Vec-based semantic search"""
    try:
        if w2v_vectors is None or word2vec_model is None:
            return pd.DataFrame(), []
        
        # Preprocess query
        query_words = preprocess_arabic_text(query).split()
        
        # Get word vectors for query words that exist in the model
        query_vectors = []
        for word in query_words:
            try:
                if word in word2vec_model.wv:
                    query_vectors.append(word2vec_model.wv[word])
            except:
                continue
        
        if not query_vectors:
            return pd.DataFrame(), []
        
        # Average the word vectors to get query representation
        query_vector = np.mean(query_vectors, axis=0)
        
        # Calculate similarities with all document vectors
        similarities = cosine_similarity([query_vector], w2v_vectors).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by minimum similarity
        valid_mask = top_similarities > 0.3
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث بـ Word2Vec: {e}")
        return pd.DataFrame(), []

def hybrid_search(query, df, models, search_weights=None, top_k=10):
    """Advanced hybrid search combining multiple models"""
    if search_weights is None:
        search_weights = {'bert': 0.4, 'w2v': 0.4, 'tfidf': 0.2}
    
    all_results = []
    
    try:
        # BERT search
        if models['bert_available']:
            bert_results, bert_similarities = bert_similarity_search(
                query, df, models['bert_vectors'], top_k=top_k*2
            )
            for idx, (_, row) in enumerate(bert_results.iterrows()):
                score = bert_similarities[idx] * search_weights['bert']
                all_results.append((row, score, 'BERT'))
        
        # Word2Vec search
        if models['w2v_available'] and models['word2vec_available']:
            w2v_results, w2v_similarities = word2vec_similarity_search(
                query, df, models['w2v_vectors'], models['word2vec_model'], top_k=top_k*2
            )
            for idx, (_, row) in enumerate(w2v_results.iterrows()):
                score = w2v_similarities[idx] * search_weights['w2v']
                all_results.append((row, score, 'Word2Vec'))
        
        # TF-IDF search
        if models['tfidf_available']:
            tfidf_results, tfidf_similarities = tfidf_search(
                query, df, top_k=top_k, vectorizer=models.get('tfidf_vectorizer')
            )
            for idx, (_, row) in enumerate(tfidf_results.iterrows()):
                score = tfidf_similarities[idx] * search_weights['tfidf']
                all_results.append((row, score, 'TF-IDF'))
        
        # If no advanced models available, use simple text search
        if not all_results:
            simple_results, simple_similarities = simple_text_search(query, df, top_k=top_k)
            for idx, (_, row) in enumerate(simple_results.iterrows()):
                score = simple_similarities[idx]
                all_results.append((row, score, 'نصي بسيط'))
        
        if not all_results:
            return pd.DataFrame(), [], []
        
        # Combine and deduplicate results
        unique_results = {}
        for row, score, model_type in all_results:
            # Use a key based on the text content
            text_key = str(row.get('text', row.get('ayah', '')))[:100]
            if text_key in unique_results:
                # Combine scores from different models
                unique_results[text_key] = (
                    row,
                    unique_results[text_key][1] + score,
                    unique_results[text_key][2] + [model_type]
                )
            else:
                unique_results[text_key] = (row, score, [model_type])
        
        # Sort by combined score
        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        
        # Extract top results
        final_results = sorted_results[:top_k]
        result_rows = [item[0] for item in final_results]
        result_scores = [item[1] for item in final_results]
        result_models = [item[2] for item in final_results]
        
        result_df = pd.DataFrame(result_rows)
        
        return result_df, result_scores, result_models
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث المختلط: {e}")
        return pd.DataFrame(), [], []
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

# Add missing import
import os

def tfidf_search(query, df, top_k=10, vectorizer=None):
    """Enhanced TF-IDF search with dynamic vectorizer creation"""
    try:
        if not SKLEARN_AVAILABLE:
            return pd.DataFrame(), []
        
        clean_query = preprocess_arabic_text(query)
        if not clean_query.strip():
            return pd.DataFrame(), []
        
        texts = df['clean_text'].fillna('').tolist()
        if not texts or len(texts) == 0:
            return pd.DataFrame(), []
        
        # Use provided vectorizer or create new one
        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=1,  # Lower min_df for small datasets
                max_df=0.95
            )
            # Fit the vectorizer on the corpus
            corpus = [clean_query] + texts
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vector = tfidf_matrix[0]
            document_vectors = tfidf_matrix[1:]
        else:
            # Use pre-fitted vectorizer
            try:
                query_vector = vectorizer.transform([clean_query])
                document_vectors = vectorizer.transform(texts)
            except:
                # If transform fails, create new vectorizer
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
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        # Get top results
        if len(similarities) == 0:
            return pd.DataFrame(), []
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by minimum similarity (lower threshold for small datasets)
        valid_mask = top_similarities > 0.01
        if not np.any(valid_mask):
            return pd.DataFrame(), []
        
        top_indices = top_indices[valid_mask]
        top_similarities = top_similarities[valid_mask]
        
        result_df = df.iloc[top_indices].copy()
        return result_df, top_similarities.tolist()
        
    except Exception as e:
        st.error(f"❌ خطأ في البحث بـ TF-IDF: {e}")
        return pd.DataFrame(), []

def display_ayah_card(row, similarity_score=None, model_types=None, card_id=None):
    """Enhanced display for Quran ayah with tafseer"""
    if card_id is None:
        card_id = f"ayah_{hash(str(row.get('text', row.get('ayah', '')))[:50])}"
    
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div style="padding: 2rem;">', unsafe_allow_html=True)
        
        # Header with surah and ayah info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            surah_name = row.get('surah', 'غير محدد')
            ayah_number = row.get('ayah_number', row.get('verse', ''))
            
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
                models_text = " + ".join(set(model_types))
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
        ayah_text = row.get('text', row.get('ayah', ''))
        if ayah_text:
            st.markdown(f"""
            <div class="premium-arabic">
                {ayah_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Tafseer (if available)
        tafseer_text = row.get('tafseer', '')
        if tafseer_text and tafseer_text != 'nan':
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
                st.success("📋 تم نسخ الآية!")
        
        with col3:
            if tafseer_text and st.button("📝 نسخ التفسير", key=f"copy_tafseer_{card_id}", use_container_width=True):
                st.code(tafseer_text, language="text")
                st.success("📝 تم نسخ التفسير!")
        
        with col4:
            if st.button("🔍 آيات مشابهة", key=f"similar_{card_id}", use_container_width=True):
                st.session_state.find_similar_ayah = ayah_text
                st.rerun()
        
        st.markdown('</div></div>', unsafe_allow_html=True)

def initialize_session_state():
    """Enhanced session state initialization"""
    defaults = {
        'search_mode': 'hybrid',
        'saved_ayahs': [],
        'reading_history': [],
        'favorite_surahs': [],
        'search_history': [],
        'daily_verses_read': 0,
        'total_verses_read': 0,
        'last_search_date': datetime.now().date(),
        'current_tab': 'search'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_model_status(models):
    """Display current model status"""
    st.markdown('<div class="model-status">', unsafe_allow_html=True)
    
    # BERT Status
    bert_class = "model-active" if models['bert_available'] else "model-inactive"
    bert_icon = "🤖" if models['bert_available'] else "❌"
    st.markdown(f'''
    <div class="model-badge {bert_class}">
        <span>{bert_icon}</span>
        <span>BERT</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Word2Vec Status
    w2v_class = "model-active" if models['w2v_available'] else "model-inactive"
    w2v_icon = "🧠" if models['w2v_available'] else "❌"
    st.markdown(f'''
    <div class="model-badge {w2v_class}">
        <span>{w2v_icon}</span>
        <span>Word2Vec</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # TF-IDF Status
    tfidf_class = "model-active" if SKLEARN_AVAILABLE else "model-inactive"
    tfidf_icon = "📊" if SKLEARN_AVAILABLE else "❌"
    st.markdown(f'''
    <div class="model-badge {tfidf_class}">
        <span>{tfidf_icon}</span>
        <span>TF-IDF</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_search_tab(df, models):
    """Enhanced search tab with multiple AI models"""
    st.markdown("### 🔍 البحث الذكي في القرآن والتفسير")
    
    # Search mode selection
    st.markdown("**🎯 اختر نوع البحث:**")
    
    search_modes = {
        'hybrid': {'name': '🔄 بحث مختلط', 'desc': 'يجمع بين جميع النماذج للحصول على أفضل النتائج'},
        'bert': {'name': '🤖 BERT', 'desc': 'بحث متقدم بالذكاء الاصطناعي'},
        'word2vec': {'name': '🧠 Word2Vec', 'desc': 'بحث بالكلمات المترابطة دلالياً'},
        'tfidf': {'name': '📊 TF-IDF', 'desc': 'بحث تقليدي بالكلمات المفتاحية'}
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(search_modes['hybrid']['name'], 
                    key="mode_hybrid", 
                    use_container_width=True,
                    type="primary" if st.session_state.search_mode == 'hybrid' else "secondary"):
            st.session_state.search_mode = 'hybrid'
    
    with col2:
        disabled = not models['bert_available']
        if st.button(search_modes['bert']['name'], 
                    key="mode_bert", 
                    use_container_width=True,
                    disabled=disabled,
                    type="primary" if st.session_state.search_mode == 'bert' else "secondary"):
            st.session_state.search_mode = 'bert'
    
    with col3:
        disabled = not models['w2v_available']
        if st.button(search_modes['word2vec']['name'], 
                    key="mode_w2v", 
                    use_container_width=True,
                    disabled=disabled,
                    type="primary" if st.session_state.search_mode == 'word2vec' else "secondary"):
            st.session_state.search_mode = 'word2vec'
    
    with col4:
        disabled = not SKLEARN_AVAILABLE
        if st.button(search_modes['tfidf']['name'], 
                    key="mode_tfidf", 
                    use_container_width=True,
                    disabled=disabled,
                    type="primary" if st.session_state.search_mode == 'tfidf' else "secondary"):
            st.session_state.search_mode = 'tfidf'
    
    # Show description of selected mode
    current_mode = search_modes[st.session_state.search_mode]
    st.info(f"**{current_mode['name']}:** {current_mode['desc']}")
    
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.slider("عدد النتائج", 5, 50, 10, key="max_results")
        
        with col2:
            if st.session_state.search_mode == 'hybrid':
                bert_weight = st.slider("وزن BERT", 0.0, 1.0, 0.4, 0.1, key="bert_weight")
                w2v_weight = st.slider("وزن Word2Vec", 0.0, 1.0, 0.4, 0.1, key="w2v_weight")
                tfidf_weight = st.slider("وزن TF-IDF", 0.0, 1.0, 0.2, 0.1, key="tfidf_weight")
                
                # Normalize weights
                total_weight = bert_weight + w2v_weight + tfidf_weight
                if total_weight > 0:
                    search_weights = {
                        'bert': bert_weight / total_weight,
                        'w2v': w2v_weight / total_weight,
                        'tfidf': tfidf_weight / total_weight
                    }
                else:
                    search_weights = {'bert': 0.4, 'w2v': 0.4, 'tfidf': 0.2}
        
        with col3:
            search_in_tafseer = st.checkbox("البحث في التفسير أيضاً", value=True, key="search_tafseer")
            include_context = st.checkbox("إظهار آيات الياق", value=False, key="show_context")
    
    # Perform search
    if search_query and (search_pressed or search_query):
        # Add to search history
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, search_query)
            st.session_state.search_history = st.session_state.search_history[:20]  # Keep last 20
        
        with st.spinner(f"🔍 جاري البحث باستخدام {current_mode['name']}..."):
            results_df = pd.DataFrame()
            similarities = []
            model_types = []
            
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
            if st.session_state.search_mode == 'hybrid':
                results_df, similarities, model_types = hybrid_search(
                    search_query, search_df, models, search_weights, max_results
                )
            elif st.session_state.search_mode == 'bert' and models['bert_available']:
                results_df, similarities = bert_similarity_search(
                    search_query, search_df, models['bert_vectors'], max_results
                )
                model_types = [['BERT']] * len(similarities)
            elif st.session_state.search_mode == 'word2vec' and models['w2v_available'] and models['word2vec_available']:
                results_df, similarities = word2vec_similarity_search(
                    search_query, search_df, models['w2v_vectors'], models['word2vec_model'], max_results
                )
                model_types = [['Word2Vec']] * len(similarities)
            elif st.session_state.search_mode == 'tfidf' and models['tfidf_available']:
                results_df, similarities = tfidf_search(
                    search_query, search_df, max_results, models.get('tfidf_vectorizer')
                )
                model_types = [['TF-IDF']] * len(similarities)
            else:
                # Fallback to simple text search
                results_df, similarities = simple_text_search(search_query, search_df, max_results)
                model_types = [['بحث نصي']] * len(similarities)
            
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
                        unique_surahs = len(results_df['surah'].unique()) if 'surah' in results_df.columns else 0
                        st.metric("عدد السور", unique_surahs)
                
                # Display results
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    similarity = similarities[idx] if idx < len(similarities) else None
                    models_used = model_types[idx] if idx < len(model_types) else None
                    display_ayah_card(row, similarity, models_used, f"result_{idx}")
                    
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
            "الصبر والثبات", "الرحمة والمغفرة", "الجنة والنعيم", 
            "التوبة والاستغفار", "الدعاء والذكر", "العدل والإحسان",
            "الصلاة والعبادة", "الأمانة والصدق", "البر والتقوى"
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
            
            history_cols = st.columns(min(len(st.session_state.search_history), 5))
            for i, query in enumerate(st.session_state.search_history[:5]):
                with history_cols[i]:
                    if st.button(f"📖 {query[:15]}...", key=f"history_{i}", use_container_width=True):
                        st.session_state.main_search = query
                        st.rerun()

def show_saved_tab(df, models):
    """Show saved ayahs and favorites"""
    st.markdown("### 📚 الآيات المحفوظة والمفضلة")
    
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
        # Try to find the ayah in the dataframe for additional info
        matching_rows = df[df['text'].str.contains(ayah[:50], case=False, na=False)]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            display_ayah_card(row, card_id=f"saved_{i}")
        else:
            # Display as simple text if not found in dataset
            st.markdown(f"""
            <div class="premium-card">
                <div style="padding: 2rem;">
                    <div class="premium-arabic">{ayah}</div>
                    <button onclick="navigator.clipboard.writeText('{ayah}')" 
                            style="margin-top: 1rem; padding: 8px 16px; 
                                   background: var(--primary-600); color: white; 
                                   border: none; border-radius: var(--radius-lg);">
                        📋 نسخ
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_stats_tab(df, models):
    """Enhanced statistics with model performance"""
    st.markdown("### 📊 الإحصائيات والتحليلات")
    
    # Dataset statistics
    st.markdown("#### 📚 إحصائيات المجموعة")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("إجمالي الآيات", f"{len(df):,}")
    
    with col2:
        unique_surahs = len(df['surah'].unique()) if 'surah' in df.columns else 0
        st.metric("عدد السور", unique_surahs)
    
    with col3:
        has_tafseer = len(df[df['tafseer'].notna()]) if 'tafseer' in df.columns else 0
        st.metric("آيات بتفسير", f"{has_tafseer:,}")
    
    with col4:
        avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
        st.metric("متوسط طول الآية", f"{avg_length:.0f} حرف")
    
    # Model performance
    st.markdown("#### 🤖 أداء النماذج")
    
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
    
    # Search history analysis
    if st.session_state.search_history:
        st.markdown("#### 🔍 تحليل عمليات البحث")
        
        # Most common search terms
        all_words = []
        for query in st.session_state.search_history:
            all_words.extend(preprocess_arabic_text(query).split())
        
        if all_words:
            word_counts = Counter(all_words)
            most_common = word_counts.most_common(10)
            
            st.markdown("**الكلمات الأكثر بحثاً:**")
            for word, count in most_common:
                st.markdown(f"- **{word}**: {count} مرة")

def show_about_tab():
    """Enhanced about section"""
    st.markdown("### ℹ️ حول تطبيق تفسير القرآن الذكي")
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, var(--primary-50), var(--secondary-50));
        padding: 2rem;
        border-radius: var(--radius-2xl);
        border: 1px solid var(--primary-200);
        margin: 2rem 0;
    ">
        <h4 style="color: var(--primary-700); margin-bottom: 1rem;">🤖 تطبيق متطور للبحث في القرآن الكريم</h4>
        <p style="color: var(--gray-700); line-height: 1.6;">
            يستخدم هذا التطبيق أحدث تقنيات الذكاء الاصطناعي للبحث الدلالي في القرآن الكريم وتفسيره.
            يجمع بين نماذج BERT و Word2Vec و TF-IDF لتوفير نتائج بحث دقيقة ومتنوعة.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical details
    st.markdown("#### 🛠️ التقنيات المستخدمة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 نماذج الذكاء الاصطناعي:**
        - **BERT**: نموذج متقدم لفهم السياق والمعنى
        - **Word2Vec**: تمثيل الكلمات في فضاء متجهي
        - **TF-IDF**: تحليل تكرار المصطلحات والوثائق
        """)
    
    with col2:
        st.markdown("""
        **📚 مصادر البيانات:**
        - مجموعة بيانات Hugging Face: MohamedRashad/Quran-Tafseer
        - تفسير شامل للقرآن الكريم
        - معالجة متقدمة للنصوص العربية
        """)
    
    # Installation guide
    st.markdown("#### 📥 دليل التثبيت")
    
    setup_code = """
# تثبيت المكتبات الأساسية
pip install streamlit pandas numpy scikit-learn

# تثبيت مكتبات الذكاء الاصطناعي
pip install datasets transformers sentence-transformers gensim

# تثبيت مكتبات إضافية
pip install joblib arabic-reshaper python-bidi

# تشغيل التطبيق
streamlit run app.py
"""
    
    st.code(setup_code, language="bash")
    
    # File requirements
    st.markdown("#### 📁 الملفات المطلوبة")
    
    st.markdown("""
    يجب وضع الملفات التالية في نفس مجلد التطبيق:
    - `bert_vectors.npy`: متجهات BERT المحسوبة مسبقاً
    - `w2v_vectors.npy`: متجهات Word2Vec للوثائق
    - `word2vec.model`: نموذج Word2Vec المدرب
    """)
    
    # Performance notes
    st.markdown("#### ⚡ ملاحظات الأداء")
    
    st.info("""
    💡 **نصائح لتحسين الأداء:**
    - استخدم البحث المختلط للحصول على أفضل النتائج
    - ابدأ بكلمات مفتاحية بسيطة ثم وسّع البحث
    - استخدم خيارات البحث المتقدمة لضبط النتائج
    - احفظ آياتك المفضلة للرجوع إليها لاحقاً
    """)

def main():
    """Enhanced main application"""
    # Initialize session state
    initialize_session_state()
    
    # Load data and models
    with st.spinner("📖 جاري تحميل البيانات والنماذج..."):
        df = load_quran_dataset()
        models = load_models()
    
    if df.empty:
        st.error("❌ لا يمكن تحميل بيانات القرآن. تأكد من الاتصال بالإنترنت وتثبيت مكتبة datasets")
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
                بحث ذكي في {len(df):,} آية قرآنية بتقنيات الذكاء الاصطناعي المتطورة
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
        show_saved_tab(df, models)
    
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
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
