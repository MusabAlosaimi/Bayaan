import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# Remove tashkeel function
def remove_tashkeel(text):
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

# Load data and model
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("adhkar_df.csv")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = vectorizer.transform(df['clean_text'])
    return df, vectorizer, tfidf_matrix

adhkar_df, vectorizer, tfidf_matrix = load_model_and_data()

# App UI
st.title("🕌 محرك بحث للأذكار باستخدام TF-IDF")
user_dua = st.text_area("أدخل الدعاء", placeholder="مثال: اللهم اجعلني من المتقين")

if st.button("🔍 بحث"):
    clean_input = remove_tashkeel(user_dua.strip())

    if not clean_input:
        st.warning("الرجاء إدخال دعاء صالح")
    else:
        user_vec = vectorizer.transform([clean_input])
        similarities = cosine_similarity(user_vec, tfidf_matrix)
        best_idx = similarities.argmax()
        score = similarities[0, best_idx]

        if score < 0.1:
            st.error("❌ لم يتم العثور على دعاء مشابه")
        else:
            st.success(f"✅ الفئة: {adhkar_df.iloc[best_idx]['category']}")
            st.write(f"📜 الدعاء الأقرب: {adhkar_df.iloc[best_idx]['text']}")
