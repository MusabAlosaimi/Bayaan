import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Dua Similarity Finder",
    page_icon="📿",
    layout="centered"
)

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

# Function to load model and data
@st.cache_resource
def load_model_and_data():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = pd.read_csv("adhkar_df.csv")
    return vectorizer, df

# Function to find the most similar dua
def find_similar_dua(user_dua, vectorizer, adhkar_df):
    clean_dua = remove_tashkeel(user_dua.strip())
    if not clean_dua:
        return "❗ الرجاء إدخال دعاء صحيح", ""
    
    user_vector = vectorizer.transform([clean_dua])
    tfidf_matrix = vectorizer.transform(adhkar_df['clean_text'])
    similarities = manual_cosine_similarity(user_vector, tfidf_matrix)
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < 0.1:
        return "❌ لم يتم العثور على دعاء مشابه", ""
    
    return adhkar_df.iloc[best_idx]['category'], adhkar_df.iloc[best_idx]['text'], best_score

# Main app
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .header {
        font-size: 2.5em;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 20px;
    }
    .subheader {
        color: #138D75;
        font-weight: bold;
    }
    .result-box {
        background-color: #EBF5FB;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .similarity {
        color: #E74C3C;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="header">📿 البحث عن دعاء مشابه</p>', unsafe_allow_html=True)
    st.write("أدخل دعاءً في الحقل أدناه للعثور على أقرب دعاء مشابه من مجموعة الأدعية")

    # Load model and data
    vectorizer, adhkar_df = load_model_and_data()

    # Input area
    user_input = st.text_area("أدخل نص الدعاء هنا:", height=150, key="dua_input")
    
    if st.button("ابحث عن الدعاء المشابه"):
        if user_input.strip() == "":
            st.warning("❗ الرجاء إدخال نص الدعاء أولاً")
        else:
            with st.spinner("جاري البحث في قاعدة الأدعية..."):
                category, dua, score = find_similar_dua(user_input, vectorizer, adhkar_df)
                
                if "❌" in category or "❗" in category:
                    st.error(category)
                else:
                    st.success("تم العثور على دعاء مشابه!")
                    with st.container():
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f'<p class="subheader">فئة الدعاء:</p><p>{category}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="subheader">نص الدعاء:</p><p>{dua}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="similarity">درجة التشابه: {score:.2%}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
