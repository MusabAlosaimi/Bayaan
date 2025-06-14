import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="أذكار المسلم الذكي - Smart Islamic Adhkar",
    page_icon="🕌",
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

# Try to import scikit-learn, fallback gracefully if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Modern React-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Modern React-style Variables */
    :root {
        --emerald-50: #ecfdf5;
        --emerald-100: #d1fae5;
        --emerald-500: #10b981;
        --emerald-600: #059669;
        --emerald-700: #047857;
        --emerald-800: #065f46;
        --teal-50: #f0fdfa;
        --teal-500: #14b8a6;
        --teal-600: #0d9488;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --blue-500: #3b82f6;
        --purple-500: #8b5cf6;
        --yellow-100: #fef3c7;
        --yellow-800: #92400e;
        --red-100: #fee2e2;
        --red-500: #ef4444;
        --white: #ffffff;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--emerald-50) 0%, var(--white) 50%, var(--teal-50) 100%);
        min-height: 100vh;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, var(--emerald-600) 0%, var(--teal-600) 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Amiri', serif;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-family: 'Inter', sans-serif;
        font-weight: 300;
    }
    
    /* Modern Tabs */
    .modern-tabs {
        display: flex;
        background: var(--white);
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-100);
    }
    
    .tab-button {
        flex: 1;
        padding: 12px 20px;
        background: transparent;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        color: var(--gray-600);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .tab-button.active {
        background: var(--emerald-600);
        color: white;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }
    
    .tab-button:hover:not(.active) {
        background: var(--emerald-50);
        color: var(--emerald-700);
    }
    
    /* Modern Search */
    .search-container {
        position: relative;
        margin-bottom: 2rem;
    }
    
    .search-icon {
        position: absolute;
        left: 16px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--gray-400);
        font-size: 1.2rem;
        z-index: 2;
    }
    
    /* Modern Cards */
    .modern-card {
        background: var(--white);
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-100);
        margin-bottom: 1.5rem;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px rgba(0, 0, 0, 0.1);
        border-color: var(--emerald-200);
    }
    
    .featured-card {
        border: 2px solid var(--emerald-500);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    
    .card-header {
        padding: 1.5rem 1.5rem 1rem 1.5rem;
    }
    
    .card-content {
        padding: 0 1.5rem 1.5rem 1.5rem;
    }
    
    .card-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--gray-100);
    }
    
    /* Category Badges */
    .category-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: capitalize;
    }
    
    .badge-morning { background: var(--yellow-100); color: var(--yellow-800); }
    .badge-evening { background: #f3e8ff; color: #7c3aed; }
    .badge-general { background: #dbeafe; color: #1d4ed8; }
    .badge-istighfar { background: var(--emerald-100); color: var(--emerald-800); }
    .badge-protection { background: var(--red-100); color: #dc2626; }
    
    /* Arabic Text */
    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 1.8rem;
        line-height: 1.8;
        color: var(--emerald-800);
        margin-bottom: 1rem;
        text-align: right;
        direction: rtl;
    }
    
    .transliteration {
        font-style: italic;
        color: var(--gray-600);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .translation {
        color: var(--gray-700);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Modern Buttons */
    .modern-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 8px;
        border: none;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .btn-primary {
        background: var(--emerald-600);
        color: white;
    }
    
    .btn-primary:hover {
        background: var(--emerald-700);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    .btn-ghost {
        background: transparent;
        color: var(--gray-500);
        border: 1px solid var(--gray-200);
    }
    
    .btn-ghost:hover {
        background: var(--gray-50);
        color: var(--gray-700);
    }
    
    .btn-ghost.active {
        color: var(--red-500);
        border-color: var(--red-200);
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, var(--emerald-500) 0%, var(--teal-500) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stat-card.blue {
        background: linear-gradient(135deg, var(--blue-500) 0%, #6366f1 100%);
    }
    
    .stat-card.purple {
        background: linear-gradient(135deg, var(--purple-500) 0%, #ec4899 100%);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Reward Box */
    .reward-box {
        background: var(--emerald-50);
        border: 1px solid var(--emerald-100);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .reward-text {
        color: var(--emerald-800);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: var(--gray-500);
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .arabic-text {
            font-size: 1.5rem;
        }
        
        .modern-tabs {
            flex-direction: column;
            gap: 4px;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0;
    }
    
    .stButton > button {
        background: var(--emerald-600) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: var(--emerald-700) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Active tab styling */
    .stButton > button:focus {
        background: var(--emerald-700) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid var(--gray-200) !important;
        padding: 12px 16px 12px 40px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--emerald-500) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sample Adhkar data (enhanced)
@st.cache_data
def load_adhkar_data():
    return [
        {
            "id": 1,
            "arabic": "سُبْحَانَ اللَّهِ وَبِحَمْدِهِ",
            "transliteration": "Subhan Allah wa bihamdihi",
            "translation": "Glory is to Allah and praise is to Him",
            "category": "morning",
            "source": "Sahih Bukhari",
            "reward": "Whoever says this 100 times, his sins will be forgiven even if they are like the foam of the sea",
            "count": 100,
        },
        {
            "id": 2,
            "arabic": "لَا إِلَهَ إِلَّا اللَّهُ وَحْدَهُ لَا شَرِيكَ لَهُ، لَهُ الْمُلْكُ وَلَهُ الْحَمْدُ وَهُوَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ",
            "transliteration": "La ilaha illa Allah wahdahu la sharika lahu, lahu al-mulku wa lahu al-hamdu wa huwa 'ala kulli shay'in qadir",
            "translation": "There is no god but Allah alone, with no partner. His is the dominion and His is the praise, and He is able to do all things",
            "category": "evening",
            "source": "Sahih Muslim",
            "reward": "Whoever says this 10 times, it is as if he freed four slaves from the children of Isma'il",
            "count": 10,
        },
        {
            "id": 3,
            "arabic": "اللَّهُمَّ أَعِنِّي عَلَى ذِكْرِكَ وَشُكْرِكَ وَحُسْنِ عِبَادَتِكَ",
            "transliteration": "Allahumma a'inni 'ala dhikrika wa shukrika wa husni 'ibadatika",
            "translation": "O Allah, help me to remember You, thank You, and worship You in the best manner",
            "category": "general",
            "source": "Abu Dawud",
            "reward": "A comprehensive du'a for spiritual improvement",
            "count": 1,
        },
        {
            "id": 4,
            "arabic": "أَسْتَغْفِرُ اللَّهَ الْعَظِيمَ الَّذِي لَا إِلَهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ وَأَتُوبُ إِلَيْهِ",
            "transliteration": "Astaghfir Allah al-'Azeem alladhi la ilaha illa huwa al-Hayy al-Qayyum wa atubu ilayhi",
            "translation": "I seek forgiveness from Allah the Mighty, whom there is no god but He, the Living, the Eternal, and I repent to Him",
            "category": "istighfar",
            "source": "At-Tirmidhi",
            "reward": "Whoever says this, Allah will forgive him even if he fled from battle",
            "count": 3,
        },
        {
            "id": 5,
            "arabic": "بِسْمِ اللَّهِ الَّذِي لَا يَضُرُّ مَعَ اسْمِهِ شَيْءٌ فِي الْأَرْضِ وَلَا فِي السَّمَاءِ وَهُوَ السَّمِيعُ الْعَلِيمُ",
            "transliteration": "Bismillah alladhi la yadurru ma'a ismihi shay'un fi al-ardi wa la fi as-sama'i wa huwa as-Sami' al-'Alim",
            "translation": "In the name of Allah, with whose name nothing on earth or in heaven can cause harm, and He is the All-Hearing, All-Knowing",
            "category": "protection",
            "source": "Abu Dawud",
            "reward": "Protection from harm when said 3 times in morning and evening",
            "count": 3,
        },
    ]

# Simple cosine similarity for search
def calculate_similarity(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words or not text_words:
        return 0
    
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    return len(intersection) / len(union) if union else 0

def initialize_session_state():
    """Initialize session state variables"""
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'read_counts' not in st.session_state:
        st.session_state.read_counts = {}
    if 'daily_adhkar' not in st.session_state:
        adhkar_data = load_adhkar_data()
        st.session_state.daily_adhkar = random.choice(adhkar_data)
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'search'

def get_category_class(category):
    category_classes = {
        'morning': 'badge-morning',
        'evening': 'badge-evening',
        'general': 'badge-general',
        'istighfar': 'badge-istighfar',
        'protection': 'badge-protection',
    }
    return category_classes.get(category, 'badge-general')

def display_adhkar_card(adhkar, featured=False):
    """Display a modern adhkar card"""
    card_class = "modern-card featured-card" if featured else "modern-card"
    category_class = get_category_class(adhkar['category'])
    
    is_favorite = adhkar['id'] in st.session_state.favorites
    read_count = st.session_state.read_counts.get(adhkar['id'], 0)
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <span class="category-badge {category_class}">{adhkar['category']}</span>
                <div style="display: flex; gap: 8px;">
                    {'❤️' if is_favorite else '🤍'}
                </div>
            </div>
            
            <div class="arabic-text">{adhkar['arabic']}</div>
            <div class="transliteration">{adhkar['transliteration']}</div>
            <div class="translation">{adhkar['translation']}</div>
        </div>
        
        <div class="card-content">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--gray-600); margin-bottom: 1rem;">
                <span>المصدر: {adhkar['source']}</span>
                <span>العدد المستحب: {adhkar['count']}</span>
            </div>
            
            <div class="reward-box">
                <div class="reward-text">
                    <strong>الفضل:</strong> {adhkar['reward']}
                </div>
            </div>
            
            <div class="card-actions">
                <div style="display: flex; gap: 8px;">
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("📖 قرأت هذا الذكر", key=f"read_{adhkar['id']}"):
            st.session_state.read_counts[adhkar['id']] = st.session_state.read_counts.get(adhkar['id'], 0) + 1
            st.success("✅ تم احتساب القراءة!")
            st.rerun()
    
    with col2:
        fav_text = "💔 إزالة من المفضلة" if is_favorite else "❤️ إضافة للمفضلة"
        if st.button(fav_text, key=f"fav_{adhkar['id']}"):
            if is_favorite:
                st.session_state.favorites.remove(adhkar['id'])
                st.success("تم إزالة الذكر من المفضلة")
            else:
                st.session_state.favorites.append(adhkar['id'])
                st.success("✅ تم إضافة الذكر للمفضلة!")
            st.rerun()
    
    with col3:
        if st.button("📋 نسخ", key=f"copy_{adhkar['id']}"):
            copy_text = f"{adhkar['arabic']}\n\n{adhkar['transliteration']}\n\n{adhkar['translation']}"
            st.code(copy_text, language="text")
    
    with col4:
        if st.button("🔗 مشاركة", key=f"share_{adhkar['id']}"):
            share_text = f"{adhkar['arabic']}\n\n{adhkar['transliteration']}\n\n{adhkar['translation']}"
            st.text_area("النص للمشاركة:", value=share_text, height=100, key=f"share_text_{adhkar['id']}")
    
    # Display read count if any
    if read_count > 0:
        st.markdown(f"""
                <div style="margin-top: 8px;">
                    <span style="background: var(--emerald-100); color: var(--emerald-700); padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                        قُرئ {read_count} مرة
                    </span>
                </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div></div>", unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load data
    adhkar_data = load_adhkar_data()
    
    # Modern Header
    st.markdown("""
    <div class="modern-header">
        <div class="header-content">
            <h1 class="header-title">أذكار المسلم الذكي</h1>
            <p class="header-subtitle">Muslim Adhkar AI - Your Intelligent Islamic Remembrance Companion</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab selection buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 البحث", key="tab_search", use_container_width=True):
            st.session_state.active_tab = 'search'
    
    with col2:
        if st.button("❤️ المفضلة", key="tab_favorites", use_container_width=True):
            st.session_state.active_tab = 'favorites'
    
    with col3:
        if st.button("⭐ اليومي", key="tab_daily", use_container_width=True):
            st.session_state.active_tab = 'daily'
    
    with col4:
        if st.button("📊 الإحصائيات", key="tab_stats", use_container_width=True):
            st.session_state.active_tab = 'stats'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display content based on active tab
    if st.session_state.active_tab == 'search':
        # Search Tab Content
        st.markdown("### 🔍 البحث في الأذكار")
        
        # Modern Search Bar
        search_query = st.text_input(
            "",
            placeholder="ابحث في الأذكار... (عربي أو إنجليزي)",
            label_visibility="collapsed",
            key="search_input"
        )
        
        # Filter adhkar based on search
        if search_query.strip():
            filtered_adhkar = []
            query_lower = search_query.lower().strip()
            
            for adhkar in adhkar_data:
                # Create searchable text including all fields
                searchable_text = f"{adhkar['arabic']} {adhkar['transliteration']} {adhkar['translation']} {adhkar['category']} {adhkar['source']}".lower()
                
                # Simple search - check if query words are in the text
                query_words = query_lower.split()
                matches = 0
                for word in query_words:
                    if word in searchable_text:
                        matches += 1
                
                # Calculate similarity score
                if matches > 0:
                    similarity = matches / len(query_words)
                    filtered_adhkar.append((adhkar, similarity))
            
            # Sort by similarity (best matches first)
            filtered_adhkar.sort(key=lambda x: x[1], reverse=True)
            adhkar_to_display = [adhkar for adhkar, _ in filtered_adhkar]
            
            if adhkar_to_display:
                st.success(f"🎯 تم العثور على {len(adhkar_to_display)} نتيجة")
            else:
                st.info("❌ لم يتم العثور على نتائج مطابقة. جرب كلمات أخرى.")
        else:
            adhkar_to_display = adhkar_data
            st.info("💡 اكتب في مربع البحث للعثور على أذكار معينة")
        
        # Display adhkar cards
        for adhkar in adhkar_to_display:
            display_adhkar_card(adhkar)
    
    elif st.session_state.active_tab == 'favorites':
        # Favorites Tab Content
        st.markdown("### ❤️ الأذكار المفضلة")
        
        favorite_adhkar = [adhkar for adhkar in adhkar_data if adhkar['id'] in st.session_state.favorites]
        
        if not favorite_adhkar:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">❤️</div>
                <h3>لا توجد أذكار مفضلة</h3>
                <p>أضف أذكارك المفضلة من قسم البحث لتظهر هنا</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"لديك {len(favorite_adhkar)} ذكر في المفضلة")
            for adhkar in favorite_adhkar:
                display_adhkar_card(adhkar)
    
    elif st.session_state.active_tab == 'daily':
        # Daily Tab Content
        st.markdown("### ⭐ ذكر اليوم")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎲 احصل على ذكر عشوائي", key="random_adhkar", use_container_width=True):
                st.session_state.daily_adhkar = random.choice(adhkar_data)
                st.success("✨ تم اختيار ذكر جديد!")
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.daily_adhkar:
            display_adhkar_card(st.session_state.daily_adhkar, featured=True)
    
    elif st.session_state.active_tab == 'stats':
        # Statistics Tab Content
        st.markdown("### 📊 الإحصائيات والتحليلات")
        
        total_reads = sum(st.session_state.read_counts.values())
        favorite_count = len(st.session_state.favorites)
        total_adhkar = len(adhkar_data)
        
        # Stats Cards
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_reads}</div>
                <div class="stat-label">إجمالي القراءات</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-number">{favorite_count}</div>
                <div class="stat-label">الأذكار المفضلة</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-number">{total_adhkar}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Most Read Adhkar
        if st.session_state.read_counts:
            st.markdown("#### 📈 الأذكار الأكثر قراءة")
            
            # Sort by read count
            sorted_reads = sorted(st.session_state.read_counts.items(), key=lambda x: x[1], reverse=True)
            
            for adhkar_id, count in sorted_reads[:5]:
                adhkar = next((a for a in adhkar_data if a['id'] == adhkar_id), None)
                if adhkar:
                    st.markdown(f"""
                    <div class="modern-card" style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem;">
                            <div style="flex: 1;">
                                <p class="arabic-text" style="font-size: 1.3rem; margin-bottom: 0.5rem;">{adhkar['arabic']}</p>
                                <p style="color: var(--gray-600); font-size: 0.9rem; margin: 0;">{adhkar['translation']}</p>
                            </div>
                            <div style="margin-left: 1rem;">
                                <span style="background: var(--emerald-100); color: var(--emerald-700); padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                                    {count} مرة
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📖 ابدأ بقراءة بعض الأذكار لترى الإحصائيات هنا")

if __name__ == "__main__":
    main()
