import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import base64
from collections import defaultdict
import re

# Set page configuration
st.set_page_config(
    page_title="أذكار المسلم - Islamic Adhkar",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Sample Adhkar data (would normally come from a CSV)
adhkar_data = [
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
    {
        "id": 6,
        "arabic": "رَبَّنَا آتِنَا فِي الدُّنْيَا حَسَنَةً وَفِي الْآخِرَةِ حَسَنَةً وَقِنَا عَذَابَ النَّارِ",
        "transliteration": "Rabbana atina fid-dunya hasanatan wa fil-akhirati hasanatan wa qina 'adhaban-nar",
        "translation": "Our Lord, give us in this world [that which is] good and in the Hereafter [that which is] good and protect us from the punishment of the Fire",
        "category": "prayer",
        "source": "Quran 2:201",
        "reward": "A comprehensive prayer for good in both worlds",
        "count": 1,
    },
    {
        "id": 7,
        "arabic": "حَسْبِيَ اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ ۖ عَلَيْهِ تَوَكَّلْتُ ۖ وَهُوَ رَبُّ الْعَرْشِ الْعَظِيمِ",
        "transliteration": "Hasbiyallahu la ilaha illa huwa, alayhi tawakkaltu, wa huwa Rabbul-arshil-azeem",
        "translation": "Sufficient for me is Allah; there is no deity except Him. On Him I have relied, and He is the Lord of the Great Throne",
        "category": "trust",
        "source": "Quran 9:129",
        "reward": "Reliance on Allah in all matters",
        "count": 7,
    }
]

# Convert to DataFrame
df = pd.DataFrame(adhkar_data)

# Initialize session state
def init_session_state():
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'read_counts' not in st.session_state:
        st.session_state.read_counts = defaultdict(int)
    if 'daily_adhkar' not in st.session_state:
        st.session_state.daily_adhkar = df.sample(1).iloc[0].to_dict()
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "ai"
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

# CSS styling
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
    
    body {{
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }}
    
    .container {{
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 1rem;
    }}
    
    .header {{
        background: linear-gradient(135deg, #0c4b6e, #1c6ea4);
        padding: 1.5rem 0;
        text-align: center;
        box-shadow: var(--card-shadow);
        position: sticky;
        top: 0;
        z-index: 100;
        color: white;
        border-radius: 0 0 20px 20px;
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        padding: 0.5rem 0;
    }}
    
    .app-title {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.25rem 0;
        font-family: 'Amiri', serif;
    }}
    
    .app-subtitle {{
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.25rem 0;
        opacity: 0.9;
    }}
    
    .nav-container {{
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        box-shadow: var(--card-shadow);
    }}
    
    .nav-button {{
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 600;
        border: none;
        background: white;
        color: var(--text);
        font-size: 1rem;
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
    
    .featured-card {{
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-left: 4px solid var(--primary);
    }}
    
    .adhkar-text {{
        font-size: 1.8rem;
        line-height: 1.8;
        color: var(--dark);
        font-family: 'Amiri', serif;
        margin-bottom: 1rem;
        font-weight: 400;
        text-align: center;
    }}
    
    .badge {{
        background: var(--accent);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
    }}
    
    .tag {{
        background: var(--primary);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
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
        margin: 5px;
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
        font-size: 0.9rem;
    }}
    
    .stats-container {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
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
    
    .icon-button {{
        background: transparent;
        border: none;
        color: #666;
        font-size: 1.2rem;
        cursor: pointer;
        padding: 8px;
        border-radius: 50%;
        transition: all 0.2s ease;
    }}
    
    .icon-button:hover {{
        background: #f0f0f0;
        color: var(--primary);
    }}
    
    .favorite {{
        color: #e53e3e !important;
    }}
    
    @media (max-width: 768px) {{
        .stats-container {{
            grid-template-columns: 1fr;
        }}
        
        .nav-container {{
            flex-wrap: wrap;
        }}
        
        .adhkar-text {{
            font-size: 1.5rem;
        }}
    }}
</style>

<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Helper functions
def increment_read_count(adhkar_id):
    st.session_state.read_counts[adhkar_id] += 1

def toggle_favorite(adhkar_id):
    if adhkar_id in st.session_state.favorites:
        st.session_state.favorites.remove(adhkar_id)
    else:
        st.session_state.favorites.append(adhkar_id)

def get_category_color(category):
    colors = {
        "morning": "#f6e05e",
        "evening": "#9f7aea",
        "general": "#4299e1",
        "istighfar": "#48bb78",
        "protection": "#f56565",
        "prayer": "#ed64a6",
        "trust": "#0bc5ea"
    }
    return colors.get(category, "#a0aec0")

# Search function
def search_adhkar(query):
    if not query:
        return df
    
    query = query.lower()
    results = []
    
    for idx, row in df.iterrows():
        # Check if query exists in any of the text fields
        if (query in row['arabic'].lower() or 
            query in row['transliteration'].lower() or 
            query in row['translation'].lower() or 
            query in row['category'].lower() or 
            query in row['source'].lower() or 
            query in row['reward'].lower()):
            results.append(row)
    
    return pd.DataFrame(results)

# UI Components
def render_adhkar_card(adhkar, featured=False):
    arabic = adhkar['arabic']
    transliteration = adhkar['transliteration']
    translation = adhkar['translation']
    category = adhkar['category']
    source = adhkar['source']
    reward = adhkar['reward']
    count = adhkar['count']
    adhkar_id = adhkar['id']
    
    read_count = st.session_state.read_counts.get(adhkar_id, 0)
    is_favorite = adhkar_id in st.session_state.favorites
    
    card_class = "card featured-card" if featured else "card"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div class="tag" style="background: {get_category_color(category)};">
                {category.capitalize()}
            </div>
            <div>
                <button class="icon-button {'favorite' if is_favorite else ''}" onclick="toggleFavorite({adhkar_id})" title="{'Remove from favorites' if is_favorite else 'Add to favorites'}">
                    ♥
                </button>
                <button class="icon-button" onclick="copyToClipboard('{arabic}')" title="Copy">
                    ⎘
                </button>
            </div>
        </div>
        
        <div class="adhkar-text">{arabic}</div>
        
        <div style="text-align: center; margin: 1rem 0; color: #4a5568;">
            <div style="font-style: italic; margin-bottom: 0.5rem;">{transliteration}</div>
            <div>{translation}</div>
        </div>
        
        <div style="margin: 1.5rem 0; padding: 1rem; background: #f7fafc; border-radius: 8px;">
            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>Reward:</strong> {reward}</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <div><strong>Source:</strong> {source}</div>
                <div><strong>Recommended count:</strong> {count}</div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <button class="action-button" onclick="incrementRead({adhkar_id})">
                Mark as read
            </button>
            
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">Read count:</span>
                <span class="badge">{read_count}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("""
    <div class="header">
        <div class="app-title">أذكار المسلم الذكي</div>
        <div class="app-subtitle">Muslim Adhkar AI - Your Intelligent Islamic Remembrance Companion</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tabs = {
        "ai": "الذكاء الاصطناعي",
        "daily": "العرضي",
        "favorites": "المفضلة",
        "search": "البحث"
    }
    
    tab_html = '<div class="nav-container">'
    for tab_id, tab_name in tabs.items():
        active = "active" if st.session_state.active_tab == tab_id else ""
        tab_html += f'<button class="nav-button {active}" onclick="setActiveTab(\'{tab_id}\')">{tab_name}</button>'
    tab_html += '</div>'
    
    st.markdown(tab_html, unsafe_allow_html=True)
    
    # Tab content
    if st.session_state.active_tab == "ai":
        st.markdown("### الذكاء الاصطناعي الموصي بالأذكار")
        
        # Featured Adhkar
        daily_adhkar = st.session_state.daily_adhkar
        render_adhkar_card(daily_adhkar, featured=True)
        
        # Refresh button
        if st.button("احصل على ذكر جديد", key="refresh_daily"):
            st.session_state.daily_adhkar = df.sample(1).iloc[0].to_dict()
            st.experimental_rerun()
        
        # AI-powered recommendations
        st.markdown("### توصيات ذكية")
        recommendations = df.sample(2)
        for idx, row in recommendations.iterrows():
            render_adhkar_card(row.to_dict())
    
    elif st.session_state.active_tab == "daily":
        st.markdown("### أذكار حسب الوقت")
        
        # Time-based greeting
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            greeting = "🌅 أذكار الصباح"
        elif 12 <= current_hour < 18:
            greeting = "☀️ أذكار الظهر"
        elif 18 <= current_hour < 22:
            greeting = "🌆 أذكار المساء"
        else:
            greeting = "🌙 أذكار الليل"
        
        st.markdown(f"""
        <div class="section">
            <h3 style="text-align: center;">{greeting}</h3>
            <p style="text-align: center; color: #4a5568;">الأذكار المناسبة لهذا الوقت من اليوم</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display time-based adhkar
        if 5 <= current_hour < 12:
            morning_adhkar = df[df['category'] == 'morning']
            for idx, row in morning_adhkar.iterrows():
                render_adhkar_card(row.to_dict())
        else:
            evening_adhkar = df[df['category'] == 'evening']
            for idx, row in evening_adhkar.iterrows():
                render_adhkar_card(row.to_dict())
    
    elif st.session_state.active_tab == "favorites":
        if not st.session_state.favorites:
            st.markdown("""
            <div class="section" style="text-align: center; padding: 3rem;">
                <h3>لا توجد أذكار مفضلة</h3>
                <p style="color: #4a5568;">أضف أذكارك المفضلة لتظهر هنا</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"### أذكارك المفضلة ({len(st.session_state.favorites)})")
            for adhkar_id in st.session_state.favorites:
                adhkar = df[df['id'] == adhkar_id].iloc[0].to_dict()
                render_adhkar_card(adhkar)
    
    elif st.session_state.active_tab == "search":
        st.markdown("### البحث في الأذكار")
        
        # Search input
        search_query = st.text_input("ابحث في الأذكار (عربي أو إنجليزي)", value=st.session_state.search_query, 
                                    placeholder="اكتب للبحث...", key="search_input")
        
        if search_query:
            st.session_state.search_query = search_query
            results = search_adhkar(search_query)
            
            if results.empty:
                st.markdown("""
                <div class="section" style="text-align: center; padding: 3rem;">
                    <h3>لم يتم العثور على نتائج</h3>
                    <p style="color: #4a5568;">جرب كلمات بحث مختلفة</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"### نتائج البحث ({len(results)})")
                for idx, row in results.iterrows():
                    render_adhkar_card(row.to_dict())
        else:
            # Show all adhkar when no search query
            for idx, row in df.iterrows():
                render_adhkar_card(row.to_dict())
    
    # Stats section
    if st.session_state.active_tab != "search":
        total_reads = sum(st.session_state.read_counts.values())
        
        st.markdown("""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">إجمالي القراءات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">الأذكار المفضلة</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">إجمالي الأذكار</div>
            </div>
        </div>
        """.format(total_reads, len(st.session_state.favorites), len(df)), unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>أذكار المسلم الذكي - إحياء سنة الذكر</p>
        <p>جميع الحقوق محفوظة © {}</p>
    </div>
    """.format(datetime.now().year), unsafe_allow_html=True)
    
    # JavaScript functions
    st.markdown("""
    <script>
    function setActiveTab(tab) {
        Streamlit.setComponentValue(tab);
    }
    
    function incrementRead(adhkarId) {
        Streamlit.setComponentValue("read_" + adhkarId);
    }
    
    function toggleFavorite(adhkarId) {
        Streamlit.setComponentValue("fav_" + adhkarId);
    }
    
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            alert("تم نسخ الذكر: " + text);
        });
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Handle component values
    if 'component_value' not in st.session_state:
        st.session_state.component_value = None
    
    component_value = st.session_state.get('component_value', None)
    
    if component_value:
        if component_value.startswith("read_"):
            adhkar_id = int(component_value.split("_")[1])
            increment_read_count(adhkar_id)
        elif component_value.startswith("fav_"):
            adhkar_id = int(component_value.split("_")[1])
            toggle_favorite(adhkar_id)
        elif component_value in ["ai", "daily", "favorites", "search"]:
            st.session_state.active_tab = component_value
        
        st.session_state.component_value = None
        st.experimental_rerun()
    
    main()
