
!pip install joblib

import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# ✅ Remove Arabic diacritics
def remove_tashkeel(text):
    tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    return tashkeel_pattern.sub('', text)

# ✅ Manual cosine similarity
def manual_cosine_similarity(a, b):
    a_dense = a.toarray().flatten()
    b_dense = b.toarray()
    dot_products = np.dot(b_dense, a_dense)
    a_norm = np.linalg.norm(a_dense)
    b_norms = np.linalg.norm(b_dense, axis=1)
    return dot_products / (a_norm * b_norms + 1e-10)

# ✅ Load model + data and build clean_text
@st.cache_resource
def load_model_and_data():
    vectorizer = joblib.load("/mnt/data/tfidf_vectorizer.pkl")
    df = pd.read_csv("/mnt/data/adhkar_df.csv")
    df['clean_text'] = df['text'].apply(lambda x: remove_tashkeel(str(x)))
    return vectorizer, df

# ✅ Search function
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
    
    return adhkar_df.iloc[best_idx]['category'], adhkar_df.iloc[best_idx]['text']

# 🟢 UI
st.set_page_config(page_title="محرك بحث الأذكار", layout="centered")
st.title("🕌 محرك بحث الأذكار بالذكاء الاصطناعي")

vectorizer, adhkar_df = load_model_and_data()

user_input = st.text_area("✏️ أدخل الدعاء هنا", placeholder="مثال: اللهم اجعلني من المتقين")

if st.button("🔍 بحث"):
    category, match = find_similar_dua(user_input, vectorizer, adhkar_df)
    st.markdown(f"**📚 الفئة:** {category}")
    st.markdown(f"**🕊️ الدعاء الأقرب:**\n\n{match}")
